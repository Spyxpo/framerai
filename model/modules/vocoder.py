"""Neural vocoder with an inverse-STFT head.

Griffin-Lim reconstructs phase by iterating between the magnitude spectrogram
and the waveform, converging on something plausible rather than correct. The
result has a characteristic metallic quality that no amount of model training
removes, because the information was discarded before the vocoder ran.

Predicting magnitude *and* phase and inverting the STFT once is both faster than
32 Griffin-Lim iterations and not subject to that ceiling.
"""

import math

import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    """Depthwise convolution, layer norm, and an inverted bottleneck."""

    def __init__(self, channels: int, mlp_ratio: float = 3.0, layer_scale: float = 1e-6):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        hidden = int(channels * mlp_ratio)
        self.pointwise1 = nn.Linear(channels, hidden)
        self.pointwise2 = nn.Linear(hidden, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels))

    @torch.no_grad()
    def reset_parameters(self):
        self.gamma.fill_(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).transpose(1, 2)
        x = self.pointwise2(nn.functional.gelu(self.pointwise1(self.norm(x))))
        return residual + (self.gamma * x).transpose(1, 2)


class ISTFTHead(nn.Module):
    """Predict log-magnitude and phase, then invert the STFT once.

    Phase is predicted as an unconstrained pair and converted with ``atan2``,
    which is well-behaved everywhere, rather than as an angle directly, which
    would have a discontinuity at the wrap point.
    """

    def __init__(self, channels: int, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq = n_fft // 2 + 1
        self.proj = nn.Linear(channels, self.n_freq * 3)
        self.register_buffer("window", torch.hann_window(n_fft))

    @torch.no_grad()
    def reset_buffers(self, device=None):
        device = device or self.window.device
        self.window = torch.hann_window(self.n_fft, device=device)

    def forward(self, features: torch.Tensor, length: int = None) -> torch.Tensor:
        """``features`` is ``(B, C, frames)``; returns ``(B, samples)``."""
        parts = self.proj(features.transpose(1, 2))  # (B, frames, n_freq * 3)
        log_magnitude, phase_x, phase_y = parts.chunk(3, dim=-1)
        # Clamped so an untrained head cannot produce inf on the first step.
        magnitude = torch.exp(log_magnitude.clamp(max=math.log(1e3)))
        phase = torch.atan2(phase_y, phase_x)

        spec = torch.polar(magnitude, phase).transpose(1, 2)  # (B, n_freq, frames)
        waveform = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=length,
        )
        return waveform


class NeuralVocoder(nn.Module):
    """Acoustic features to a waveform."""

    def __init__(
        self,
        in_channels: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.embed = nn.Conv1d(in_channels, d_model, 7, padding=3)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtBlock(d_model) for _ in range(n_layers)])
        self.head = ISTFTHead(d_model, n_fft, hop_length)

    def forward(self, features: torch.Tensor, length: int = None) -> torch.Tensor:
        x = self.embed(features)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.head(x, length=length)
