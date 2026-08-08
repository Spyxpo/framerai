"""Causal 3D variational autoencoder for video.

Video diffusion in pixel space is the image problem multiplied by frame count.
This compresses 4x along time and 8x along each spatial dimension before the
denoiser sees anything: a 16-frame 512x512 clip becomes a 4x64x64 latent, a 256x
reduction in the volume being denoised.

Every temporal convolution is *causal* - frame t is padded only on its left, so
it never sees frame t+1. That is what makes variable duration and streaming
decode possible: a clip can be extended without recomputing what came before,
and the first frames can be emitted before the last are generated. It also lets
a single image be encoded as a one-frame clip, which is how image and video
training share a latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import DiagonalGaussian


class CausalConv3d(nn.Module):
    """3D convolution with causal padding along time.

    Spatial dimensions are padded symmetrically as usual; the temporal dimension
    is padded ``kernel_t - 1`` frames on the left and none on the right.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.time_pad = (kernel_size[0] - 1) * dilation
        self.spatial_pad = (kernel_size[1] // 2, kernel_size[2] // 2)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(0, self.spatial_pad[0], self.spatial_pad[1]),
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (left, right) per dimension, innermost first: W, H, T.
        x = F.pad(x, (0, 0, 0, 0, self.time_pad, 0))
        return self.conv(x)


class CausalGroupNorm(nn.Module):
    """GroupNorm applied independently to each frame.

    ``nn.GroupNorm`` on a ``(B, C, T, H, W)`` tensor normalizes over the whole
    volume, so every frame's statistics depend on every other frame - including
    later ones. That silently defeats causal convolutions. Normalizing per frame
    keeps the guarantee.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).reshape(B * T, C, H, W)
        x = self.norm(x)
        return x.reshape(B, T, C, H, W).transpose(1, 2)


class CausalResnetBlock3d(nn.Module):
    """Pre-norm residual block built from causal convolutions."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = CausalGroupNorm(min(groups, in_channels), in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3)
        self.norm2 = CausalGroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


def _levels(factor: int, name: str) -> int:
    if factor < 1 or (factor & (factor - 1)):
        raise ValueError(f"{name} ({factor}) must be a power of two")
    return factor.bit_length() - 1


class CausalVideoVAE(nn.Module):
    """Encode a clip to a latent volume and back.

    Temporal downsampling happens on the earliest levels and spatial
    downsampling continues past it, so ``temporal_downsample`` may be smaller
    than ``spatial_downsample`` (4 and 8 by default).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 8,
        base_channels: int = 128,
        temporal_downsample: int = 4,
        spatial_downsample: int = 8,
        scale_factor: float = 0.18215,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample

        time_levels = _levels(temporal_downsample, "video_vae_temporal_downsample")
        space_levels = _levels(spatial_downsample, "video_vae_spatial_downsample")
        if time_levels > space_levels:
            raise ValueError(
                f"video_vae_temporal_downsample ({temporal_downsample}) cannot exceed "
                f"video_vae_spatial_downsample ({spatial_downsample})"
            )

        self.encoder_in = CausalConv3d(in_channels, base_channels, 3)
        encoder = []
        channels = base_channels
        for level in range(space_levels):
            out_channels = min(base_channels * (2 ** (level + 1)), base_channels * 8)
            encoder.append(CausalResnetBlock3d(channels, out_channels))
            # Halve time only while temporal levels remain.
            stride = (2 if level < time_levels else 1, 2, 2)
            encoder.append(CausalConv3d(out_channels, out_channels, 3, stride=stride))
            channels = out_channels
        self.encoder = nn.Sequential(*encoder)
        self.encoder_mid = CausalResnetBlock3d(channels, channels)
        self.encoder_norm = CausalGroupNorm(min(32, channels), channels)
        self.encoder_out = CausalConv3d(channels, latent_channels * 2, 3)

        self.decoder_in = CausalConv3d(latent_channels, channels, 3)
        self.decoder_mid = CausalResnetBlock3d(channels, channels)
        decoder = []
        for level in reversed(range(space_levels)):
            next_channels = (
                min(base_channels * (2 ** level), base_channels * 8) if level else base_channels
            )
            scale = (2 if level < time_levels else 1, 2, 2)
            decoder.append(nn.Upsample(scale_factor=scale, mode="nearest"))
            decoder.append(CausalResnetBlock3d(channels, next_channels))
            channels = next_channels
        self.decoder = nn.Sequential(*decoder)
        self.decoder_norm = CausalGroupNorm(min(32, channels), channels)
        self.decoder_out = CausalConv3d(channels, in_channels, 3)

        self.register_buffer("scale_factor", torch.tensor(float(scale_factor)))

    @torch.no_grad()
    def reset_buffers(self, device=None):
        device = device or self.scale_factor.device
        self.scale_factor = torch.tensor(0.18215, device=device)

    def encode(self, video: torch.Tensor) -> DiagonalGaussian:
        """``video`` is ``(B, C, T, H, W)``."""
        h = self.encoder_mid(self.encoder(self.encoder_in(video)))
        return DiagonalGaussian(self.encoder_out(F.silu(self.encoder_norm(h))))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_mid(self.decoder_in(z / self.scale_factor))
        h = self.decoder(h)
        return self.decoder_out(F.silu(self.decoder_norm(h)))

    def encode_to_latent(self, video: torch.Tensor, sample: bool = True) -> torch.Tensor:
        posterior = self.encode(video)
        latent = posterior.sample() if sample else posterior.mode()
        return latent * self.scale_factor

    def forward(self, video: torch.Tensor) -> tuple:
        posterior = self.encode(video)
        return self.decode(posterior.sample() * self.scale_factor), posterior.kl()

    def latent_shape(self, batch: int, frames: int, height: int, width: int) -> tuple:
        for value, factor, name in (
            (height, self.spatial_downsample, "height"),
            (width, self.spatial_downsample, "width"),
        ):
            if value % factor:
                raise ValueError(f"{name} ({value}) must be divisible by {factor}")
        # Temporal length rounds up: a clip shorter than the compression factor
        # still yields one latent frame.
        latent_frames = max(1, -(-frames // self.temporal_downsample))
        return (
            batch,
            self.latent_channels,
            latent_frames,
            height // self.spatial_downsample,
            width // self.spatial_downsample,
        )
