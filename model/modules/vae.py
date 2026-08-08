"""KL-regularized variational autoencoder for latent-space diffusion.

Pixel-space diffusion is quadratic in pixel count at every attention site, which
is why the U-Net path cannot run at the resolutions it is configured for. This
VAE compresses an image by ``downsample`` in each spatial dimension before the
denoiser ever sees it, so a 512x512 image becomes a 64x64 latent - a 64x
reduction in the sequence the denoiser attends over.

Attention appears only at the bottleneck, where the spatial extent is already
divided by ``downsample``. Putting it at full resolution is exactly the mistake
this module exists to avoid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussian:
    """A diagonal Gaussian posterior parameterized by concatenated mean/logvar."""

    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        # Clamped so a diverging encoder produces a large loss rather than inf.
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: torch.Generator = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """KL against a standard normal, averaged over the batch."""
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar,
            dim=[1, 2, 3],
        ).mean()


class ResnetBlock(nn.Module):
    """Pre-norm residual block with a projection shortcut when widths differ."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class BottleneckAttention(nn.Module):
    """Self-attention at the compressed bottleneck only.

    Safe here precisely because the spatial extent has already been divided by
    ``downsample``: at 512px with 8x compression this attends over 64x64 = 4096
    positions, not 262144.
    """

    def __init__(self, channels: int, n_heads: int = 8, groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out = nn.Conv1d(channels, channels, 1)
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h).reshape(B, 3, self.n_heads, self.head_dim, H * W)
        q, k, v = (t.transpose(-2, -1) for t in (qkv[:, 0], qkv[:, 1], qkv[:, 2]))
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(-2, -1).reshape(B, C, H * W)
        return x + self.out(attended).view(B, C, H, W)


class Encoder(nn.Module):
    """Image to a diagonal-Gaussian latent, downsampling by a power of two."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        downsample: int = 8,
        n_heads: int = 8,
    ):
        super().__init__()
        levels = _levels(downsample)
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        blocks = []
        channels = base_channels
        for level in range(levels):
            out_channels = min(base_channels * (2 ** (level + 1)), base_channels * 8)
            blocks.append(ResnetBlock(channels, out_channels))
            blocks.append(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))
            channels = out_channels
        self.down = nn.Sequential(*blocks)

        self.mid_block1 = ResnetBlock(channels, channels)
        self.mid_attn = BottleneckAttention(channels, n_heads)
        self.mid_block2 = ResnetBlock(channels, channels)

        self.norm_out = nn.GroupNorm(min(32, channels), channels)
        # Two latent channels' worth of output: mean and log-variance.
        self.conv_out = nn.Conv2d(channels, latent_channels * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> DiagonalGaussian:
        h = self.down(self.conv_in(x))
        h = self.mid_block2(self.mid_attn(self.mid_block1(h)))
        return DiagonalGaussian(self.conv_out(F.silu(self.norm_out(h))))


class Decoder(nn.Module):
    """Latent back to an image, mirroring the encoder."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        downsample: int = 8,
        n_heads: int = 8,
    ):
        super().__init__()
        levels = _levels(downsample)
        channels = min(base_channels * (2 ** levels), base_channels * 8)
        self.conv_in = nn.Conv2d(latent_channels, channels, 3, padding=1)

        self.mid_block1 = ResnetBlock(channels, channels)
        self.mid_attn = BottleneckAttention(channels, n_heads)
        self.mid_block2 = ResnetBlock(channels, channels)

        blocks = []
        for level in reversed(range(levels)):
            next_channels = (
                min(base_channels * (2 ** level), base_channels * 8) if level else base_channels
            )
            blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            blocks.append(ResnetBlock(channels, next_channels))
            channels = next_channels
        self.up = nn.Sequential(*blocks)

        self.norm_out = nn.GroupNorm(min(32, channels), channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block2(self.mid_attn(self.mid_block1(h)))
        h = self.up(h)
        return self.conv_out(F.silu(self.norm_out(h)))


def _levels(downsample: int) -> int:
    """Number of halvings, validating that ``downsample`` is a power of two."""
    if downsample < 1 or (downsample & (downsample - 1)):
        raise ValueError(f"vae_downsample ({downsample}) must be a power of two")
    return downsample.bit_length() - 1


class KLVAE(nn.Module):
    """KL-regularized autoencoder mapping images to and from a latent grid.

    ``scale_factor`` normalizes the latent to roughly unit variance so the
    diffusion noise schedule sees a well-conditioned target. It is a property of
    the trained encoder; the default is a reasonable starting value and should be
    re-measured once the VAE is trained (``latents.std()`` over a data sample).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        downsample: int = 8,
        n_heads: int = 8,
        scale_factor: float = 0.18215,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.downsample = downsample
        self.encoder = Encoder(in_channels, latent_channels, base_channels, downsample, n_heads)
        self.decoder = Decoder(in_channels, latent_channels, base_channels, downsample, n_heads)
        self.register_buffer("scale_factor", torch.tensor(float(scale_factor)))

    @torch.no_grad()
    def reset_buffers(self, device=None):
        device = device or self.scale_factor.device
        self.scale_factor = torch.tensor(0.18215, device=device)

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z / self.scale_factor)

    def encode_to_latent(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Encode straight to a scaled latent, the form the denoiser trains on."""
        posterior = self.encode(x)
        latent = posterior.sample() if sample else posterior.mode()
        return latent * self.scale_factor

    def forward(self, x: torch.Tensor) -> tuple:
        """Reconstruction and KL, for training the autoencoder itself."""
        posterior = self.encode(x)
        recon = self.decoder(posterior.sample())
        return recon, posterior.kl()

    def latent_shape(self, batch: int, height: int, width: int) -> tuple:
        if height % self.downsample or width % self.downsample:
            raise ValueError(
                f"image size {height}x{width} must be divisible by "
                f"vae_downsample ({self.downsample})"
            )
        return (batch, self.latent_channels, height // self.downsample, width // self.downsample)
