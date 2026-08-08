"""Audio generation module.

Generates a log-mel spectrogram with a text-conditioned diffusion decoder
(reusing the image U-Net over a single-channel mel "image"), then reconstructs a
waveform with a pure-torch Griffin-Lim inversion. This keeps audio output
self-contained with no external vocoder dependency.
"""

import math

import torch
import torch.nn as nn

from .audio_encoder import mel_filterbank
from .diffusion import DiffusionModule


class AudioGenerator(nn.Module):
    """Text-conditioned mel diffusion with Griffin-Lim waveform reconstruction."""

    def __init__(
        self,
        n_mels: int = 80,
        n_frames: int = 128,
        base_channels: int = 128,
        context_dim: int = 1024,
        num_steps: int = 1000,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Treat the mel spectrogram as a single-channel image for diffusion.
        self.diffusion = DiffusionModule(
            in_channels=1,
            base_channels=base_channels,
            context_dim=context_dim,
            num_steps=num_steps,
        )

        # Pseudo-inverse of the mel filterbank maps mel power back to linear power.
        fb = mel_filterbank(sample_rate, n_fft, n_mels)  # (n_mels, n_freq)
        self.register_buffer("mel_pinv", torch.linalg.pinv(fb))  # (n_freq, n_mels)
        self.register_buffer("gl_window", torch.hann_window(n_fft))

    @torch.no_grad()
    def reset_buffers(self, device=None):
        """Recompute the derived buffers, for use after a meta-device build."""
        device = device or self.gl_window.device
        fb = mel_filterbank(self.sample_rate, self.n_fft, self.n_mels)
        self.mel_pinv = torch.linalg.pinv(fb).to(device)
        self.gl_window = torch.hann_window(self.n_fft, device=device)

    def forward(self, target_mel: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training forward: diffusion loss over the target mel (B, 1, n_mels, n_frames)."""
        return self.diffusion(target_mel, context)

    @torch.no_grad()
    def sample(self, context: torch.Tensor = None, device: str = "cpu", batch_size: int = 1) -> torch.Tensor:
        """Generate a normalized mel spectrogram in [-1, 1]."""
        shape = (batch_size, 1, self.n_mels, self.n_frames)
        return self.diffusion.sample(shape, context, device)

    def _denormalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Undo the [-1, 1] log-mel normalization used during training."""
        log_mel = mel * 4.0 - 4.0
        return torch.exp(log_mel)

    @torch.no_grad()
    def mel_to_waveform(self, mel: torch.Tensor, n_iter: int = 32) -> torch.Tensor:
        """Reconstruct a waveform from a normalized mel via Griffin-Lim."""
        if mel.dim() == 4:
            mel = mel[0]
        if mel.dim() == 3:
            mel = mel[0]  # (n_mels, n_frames)

        mel_power = self._denormalize_mel(mel)  # (n_mels, n_frames)
        lin_power = torch.matmul(self.mel_pinv.to(mel.device), mel_power).clamp(min=0.0)
        mag = torch.sqrt(lin_power)  # (n_freq, n_frames)

        window = self.gl_window.to(mel.device)
        angles = torch.exp(2j * math.pi * torch.rand_like(mag))
        spec = mag * angles
        waveform = None
        for _ in range(n_iter):
            waveform = torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
            rebuilt = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
            )
            angles = rebuilt / (rebuilt.abs() + 1e-8)
            spec = mag * angles

        waveform = torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        return waveform
