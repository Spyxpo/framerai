"""Audio encoder for processing audio/speech inputs.

Turns a raw waveform into a sequence of embeddings in the language model space,
mirroring the vision encoder. The mel front-end is implemented with torch.stft
and a pure-torch mel filterbank so the model stays importable without any
external audio library.
"""

import math

import torch
import torch.nn as nn

from .vision_encoder import VisionBlock


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """Build a triangular mel filterbank of shape (n_mels, n_fft // 2 + 1)."""
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)

    min_mel = _hz_to_mel(0.0)
    max_mel = _hz_to_mel(sample_rate / 2)
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        rising = (freqs - left) / (center - left + 1e-8)
        falling = (right - freqs) / (right - center + 1e-8)
        fb[m - 1] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb


class AudioFrontEnd(nn.Module):
    """Convert a waveform into a log-mel spectrogram using torch primitives."""

    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160, n_mels: int = 80):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("fb", mel_filterbank(sample_rate, n_fft, n_mels))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, num_samples) -> log-mel (B, n_mels, frames)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(waveform.device),
            return_complex=True,
        )
        power = spec.abs() ** 2  # (B, freq, frames)
        mel = torch.matmul(self.fb.to(waveform.device), power)  # (B, n_mels, frames)
        return torch.log(mel.clamp(min=1e-5))


class AudioEncoder(nn.Module):
    """ViT-style encoder over mel frames for audio understanding."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 12,
        max_frames: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.frontend = AudioFrontEnd(sample_rate, n_fft, hop_length, n_mels)
        self.input_proj = nn.Linear(n_mels, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            VisionBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: waveform (B, N) / (B, 1, N), or precomputed log-mel (B, n_mels, T)
        if audio.dim() == 3 and audio.shape[1] == self.n_mels:
            mel = audio
        else:
            mel = self.frontend(audio)

        x = mel.transpose(1, 2)  # (B, T, n_mels)
        if x.shape[1] > self.max_frames:
            x = x[:, : self.max_frames]
        x = self.input_proj(x)  # (B, T, d_model)

        B, T, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : T + 1]

        for block in self.blocks:
            x = block(x)
        return self.norm(x)
