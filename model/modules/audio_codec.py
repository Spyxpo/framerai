"""Neural audio codec: waveform to discrete acoustic tokens and back.

A causal convolutional encoder strides the waveform down to a low frame rate
(24 kHz / 320 = 75 Hz by default), residual vector quantization turns each frame
into a small stack of integers, and a decoder reconstructs the waveform. The
integers are what the language model predicts, which is the point: acoustics
become a sequence-modelling problem rather than a spectrogram-regression one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rvq import ResidualVQ


class CausalConv1d(nn.Module):
    """1D convolution padded only on the left, so no frame sees its future."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.pad = kernel_size - stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, 7, dilation=dilation, padding=3 * dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(F.elu(self.conv1(F.elu(x))))
        return x + h


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.units = nn.Sequential(
            ResidualUnit(in_channels, 1),
            ResidualUnit(in_channels, 3),
            ResidualUnit(in_channels, 9),
        )
        self.down = CausalConv1d(in_channels, out_channels, 2 * stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.elu(self.units(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, out_channels, 2 * stride, stride=stride, padding=stride // 2
        )
        self.units = nn.Sequential(
            ResidualUnit(out_channels, 1),
            ResidualUnit(out_channels, 3),
            ResidualUnit(out_channels, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.units(self.up(F.elu(x)))


def _strides_for(hop: int) -> list:
    """Factor the total hop into per-block strides.

    Prefers a few moderate strides over one large one: a stride-320 convolution
    would have to learn everything at once, and each block gets residual units
    at increasing dilation to widen its receptive field.
    """
    strides, remaining = [], hop
    for candidate in (8, 5, 4, 2):
        while remaining % candidate == 0 and remaining > 1:
            strides.append(candidate)
            remaining //= candidate
    if remaining != 1:
        raise ValueError(
            f"codec_hop ({hop}) must factor into strides of 8, 5, 4, and 2; "
            f"{remaining} is left over"
        )
    return strides


class AudioCodec(nn.Module):
    """Waveform to acoustic tokens and back."""

    def __init__(
        self,
        base_channels: int = 64,
        latent_dim: int = 256,
        hop: int = 320,
        n_quantizers: int = 8,
        codebook_size: int = 1024,
        sample_rate: int = 24000,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        self.hop = hop
        self.sample_rate = sample_rate
        strides = _strides_for(hop)

        channels = base_channels
        encoder = [CausalConv1d(1, channels, 7)]
        for stride in strides:
            out_channels = min(channels * 2, base_channels * 16)
            encoder.append(EncoderBlock(channels, out_channels, stride))
            channels = out_channels
        encoder.append(CausalConv1d(channels, latent_dim, 3))
        self.encoder = nn.Sequential(*encoder)

        self.quantizer = ResidualVQ(
            n_quantizers=n_quantizers,
            codebook_size=codebook_size,
            codebook_dim=latent_dim,
            quantizer_dropout=quantizer_dropout,
        )

        decoder = [nn.Conv1d(latent_dim, channels, 7, padding=3)]
        for stride in reversed(strides):
            out_channels = max(channels // 2, base_channels)
            decoder.append(DecoderBlock(channels, out_channels, stride))
            channels = out_channels
        decoder.append(nn.Conv1d(channels, 1, 7, padding=3))
        self.decoder = nn.Sequential(*decoder)

    def encode_latent(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, 1, N)
        return self.encoder(waveform)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, n_quantizers: int = None) -> torch.Tensor:
        """Acoustic token ids, ``(B, n_q, frames)``."""
        return self.quantizer.encode(self.encode_latent(waveform), n_quantizers)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct a waveform from token ids, ``(B, samples)``."""
        return self.decoder(self.quantizer.decode(codes)).squeeze(1)

    def forward(self, waveform: torch.Tensor) -> tuple:
        """Returns ``(reconstruction, commit_loss, codes)``."""
        latent = self.encode_latent(waveform)
        quantized, codes, commit_loss = self.quantizer(latent)
        recon = self.decoder(quantized).squeeze(1)
        return recon, commit_loss, codes

    def frames_for(self, samples: int) -> int:
        return max(1, samples // self.hop)
