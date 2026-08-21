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
from .vocoder import NeuralVocoder


class AudioGenerator(nn.Module):
    """Text-conditioned mel diffusion with neural vocoder or Griffin-Lim waveform reconstruction."""

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
        vocoder_arch: str = "griffin_lim",
        vocoder_d_model: int = 512,
        vocoder_n_layers: int = 8,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.vocoder_arch = vocoder_arch

        # Treat the mel spectrogram as a single-channel image for diffusion.
        self.diffusion = DiffusionModule(
            in_channels=1,
            base_channels=base_channels,
            context_dim=context_dim,
            num_steps=num_steps,
        )

        self.vocoder = (
            NeuralVocoder(
                in_channels=n_mels,
                d_model=vocoder_d_model,
                n_layers=vocoder_n_layers,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            if vocoder_arch != "griffin_lim"
            else None
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

    def forward(
        self,
        target_mel: torch.Tensor,
        context: torch.Tensor = None,
        target_waveform: torch.Tensor = None,
    ) -> torch.Tensor:
        """Training forward: diffusion loss over the target mel (+ vocoder loss if configured)."""
        loss = self.diffusion(target_mel, context)
        if self.vocoder is not None and target_waveform is not None:
            loss = loss + self.vocoder_loss(target_mel, target_waveform)
        return loss

    def vocoder_loss(
        self, target_mel: torch.Tensor, target_waveform: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruction loss for training/fine-tuning the neural vocoder."""
        if self.vocoder is None:
            return torch.tensor(0.0, device=target_mel.device)

        if target_mel.dim() == 4:
            features = target_mel.squeeze(1)
        elif target_mel.dim() == 2:
            features = target_mel.unsqueeze(0)
        else:
            features = target_mel

        frames = features.shape[-1]
        target_length = (frames - 1) * self.hop_length
        recon_wav = self.vocoder(features, length=target_length)

        if target_waveform.dim() == 1:
            target_wav = target_waveform.unsqueeze(0)
        else:
            target_wav = target_waveform

        length = min(recon_wav.shape[-1], target_wav.shape[-1])
        recon_sub = recon_wav[..., :length]
        target_sub = target_wav[..., :length]

        time_loss = torch.nn.functional.l1_loss(recon_sub, target_sub)

        window = self.gl_window.to(target_mel.device)
        recon_spec = torch.stft(
            recon_sub.squeeze(0) if recon_sub.shape[0] == 1 else recon_sub,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        ).abs()
        target_spec = torch.stft(
            target_sub.squeeze(0) if target_sub.shape[0] == 1 else target_sub,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        ).abs()
        stft_loss = torch.nn.functional.l1_loss(recon_spec, target_spec)

        return time_loss + stft_loss

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
    def mel_to_waveform(
        self,
        mel: torch.Tensor,
        n_iter: int = 32,
        use_griffin_lim: bool = False,
    ) -> torch.Tensor:
        """Reconstruct a waveform from a normalized mel via neural vocoder or Griffin-Lim."""
        if self.vocoder is not None and not use_griffin_lim:
            if mel.dim() == 4:
                features = mel.squeeze(1)
            elif mel.dim() == 2:
                features = mel.unsqueeze(0)
            else:
                features = mel

            frames = features.shape[-1]
            target_length = (frames - 1) * self.hop_length
            waveform = self.vocoder(features, length=target_length)
            if mel.dim() in (2, 3) or (mel.dim() == 4 and mel.shape[0] == 1):
                waveform = waveform[0]
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
            return waveform

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
