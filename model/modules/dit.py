"""Diffusion transformer denoiser with adaLN-zero conditioning.

Replaces the convolutional U-Net for latent-space image generation. Three
properties matter here and none of them hold for the U-Net:

* Positions come from an on-the-fly 2D sin-cos grid rather than a learned table,
  so the same weights denoise any resolution and aspect ratio.
* Timestep and pooled-text conditioning modulate every block through adaLN-zero.
  The modulation projection is zero-initialized, so each block starts as the
  identity and the residual stream is undisturbed at step zero - which is what
  makes deep denoisers trainable.
* Attention is the fused SDPA kernel over a latent grid, not a materialized
  matrix over pixels.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_pos_embed_2d(d_model: int, height: int, width: int, device=None) -> torch.Tensor:
    """2D sin-cos positional embedding, computed per call.

    A learned table would fix the resolution at construction time. Computing the
    grid means one set of weights serves every size the VAE can produce.
    """
    if d_model % 4:
        raise ValueError(f"dit_d_model ({d_model}) must be divisible by 4 for 2D sin-cos")
    quarter = d_model // 4
    omega = torch.exp(
        -math.log(10000.0) * torch.arange(quarter, device=device, dtype=torch.float32) / quarter
    )

    def axis(length):
        pos = torch.arange(length, device=device, dtype=torch.float32)
        angles = pos[:, None] * omega[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (length, d_model // 2)

    emb_h = axis(height)[:, None, :].expand(height, width, d_model // 2)
    emb_w = axis(width)[None, :, :].expand(height, width, d_model // 2)
    return torch.cat([emb_h, emb_w], dim=-1).reshape(height * width, d_model)


class TimestepEmbedder(nn.Module):
    """Scalar timestep to a conditioning vector."""

    def __init__(self, d_model: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Self-attention, text cross-attention, and SwiGLU, all adaLN-zero gated."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.q_cross = nn.Linear(d_model, d_model, bias=False)
        self.kv_cross = nn.Linear(context_dim, 2 * d_model, bias=False)
        self.cross_out = nn.Linear(d_model, d_model, bias=False)

        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(d_model, int(d_model * mlp_ratio))

        # adaLN-zero: shift/scale/gate for each of the three sub-layers.
        self.modulation = nn.Linear(d_model, 9 * d_model)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Zero the modulation projection so the block starts as the identity."""
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor, context: torch.Tensor = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        params = self.modulation(F.silu(conditioning)).chunk(9, dim=-1)
        shift_attn, scale_attn, gate_attn = params[0], params[1], params[2]
        shift_cross, scale_cross, gate_cross = params[3], params[4], params[5]
        shift_mlp, scale_mlp, gate_mlp = params[6], params[7], params[8]

        h = modulate(self.norm1(x), shift_attn, scale_attn)
        q, k, v = (self._heads(t) for t in self.qkv(h).chunk(3, dim=-1))
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(B, T, C)
        x = x + gate_attn.unsqueeze(1) * self.attn_out(attended)

        if context is not None:
            h = modulate(self.norm2(x), shift_cross, scale_cross)
            q = self._heads(self.q_cross(h))
            k, v = (self._heads(t) for t in self.kv_cross(context).chunk(2, dim=-1))
            crossed = F.scaled_dot_product_attention(q, k, v)
            crossed = crossed.transpose(1, 2).reshape(B, T, C)
            x = x + gate_cross.unsqueeze(1) * self.cross_out(crossed)

        h = modulate(self.norm3(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(h)


class PatchEmbed2D(nn.Module):
    """Patchify a latent grid into a token sequence."""

    def __init__(self, in_channels: int, d_model: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple:
        _, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"latent {height}x{width} must be divisible by "
                f"dit_patch_size ({self.patch_size})"
            )
        tokens = self.proj(x)
        grid = (tokens.shape[-2], tokens.shape[-1])
        return tokens.flatten(2).transpose(1, 2), grid


class DiT(nn.Module):
    """Diffusion transformer over a latent grid.

    ``forward`` predicts the flow velocity (or the noise, depending on the
    objective the caller trains with); it is a plain vector field over latents
    and carries no assumption about the schedule.
    """

    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 1152,
        n_layers: int = 28,
        n_heads: int = 16,
        patch_size: int = 2,
        context_dim: int = 1024,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"dit_d_model ({d_model}) must be divisible by dit_n_heads ({n_heads})")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_embed = PatchEmbed2D(in_channels, d_model, patch_size)
        self.time_embed = TimestepEmbedder(d_model)
        # Pooled text, so conditioning reaches the modulation as well as the
        # cross-attention. Without it adaLN sees only the timestep.
        self.context_pool = nn.Linear(context_dim, d_model)

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, context_dim, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.modulation_out = nn.Linear(d_model, 2 * d_model)
        self.proj_out = nn.Linear(d_model, patch_size * patch_size * in_channels)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Zero the output head so the model predicts nothing before training."""
        nn.init.zeros_(self.modulation_out.weight)
        nn.init.zeros_(self.modulation_out.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def _unpatchify(self, tokens: torch.Tensor, grid: tuple) -> torch.Tensor:
        B = tokens.shape[0]
        h, w = grid
        p, c = self.patch_size, self.in_channels
        x = tokens.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, c, h * p, w * p)

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None
    ) -> torch.Tensor:
        tokens, grid = self.patch_embed(z)
        tokens = tokens + sincos_pos_embed_2d(
            self.d_model, grid[0], grid[1], device=tokens.device
        ).to(tokens.dtype)

        conditioning = self.time_embed(t)
        if context is not None:
            conditioning = conditioning + self.context_pool(context.mean(dim=1))

        for block in self.blocks:
            tokens = block(tokens, conditioning, context)

        shift, scale = self.modulation_out(F.silu(conditioning)).chunk(2, dim=-1)
        tokens = modulate(self.norm_out(tokens), shift, scale)
        return self._unpatchify(self.proj_out(tokens), grid)
