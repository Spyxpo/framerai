"""Spacetime diffusion transformer for video.

Attention is factorised into a spatial pass and a temporal pass. Full 3D
attention over T*H*W positions is quadratic in the whole volume, which at any
useful clip length is not affordable; factorising costs
``O(T * HW^2 + HW * T^2)`` instead of ``O((T * HW)^2)`` while still letting
information reach every position in two hops.

Both passes are single batched reshapes. That is what removes the per-frame
Python loop in the old 3D U-Net, which ran one convolution call per frame per
block and made throughput fall linearly with clip length.

As with the image transformer, positions come from a computed 3D sin-cos grid
rather than a learned table, so the same weights denoise any duration,
resolution, and aspect ratio.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import SwiGLU, TimestepEmbedder, modulate


def sincos_pos_embed_3d(d_model: int, frames: int, height: int, width: int, device=None):
    """3D sin-cos positional embedding over ``(T, H, W)``, computed per call."""
    if d_model % 6:
        raise ValueError(
            f"video_dit_d_model ({d_model}) must be divisible by 6 for 3D sin-cos"
        )
    per_axis = d_model // 3
    half = per_axis // 2
    omega = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )

    def axis(length):
        pos = torch.arange(length, device=device, dtype=torch.float32)
        angles = pos[:, None] * omega[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (length, per_axis)

    emb_t = axis(frames)[:, None, None, :].expand(frames, height, width, per_axis)
    emb_h = axis(height)[None, :, None, :].expand(frames, height, width, per_axis)
    emb_w = axis(width)[None, None, :, :].expand(frames, height, width, per_axis)
    return torch.cat([emb_t, emb_h, emb_w], dim=-1).reshape(frames * height * width, d_model)


class SpacetimeDiTBlock(nn.Module):
    """Spatial attention, temporal attention, text cross-attention, SwiGLU.

    All four sub-layers are adaLN-zero gated, so the block is the identity at
    initialization and a deep stack stays trainable.
    """

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

        self.norm_spatial = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv_spatial = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_spatial = nn.Linear(d_model, d_model, bias=False)

        self.norm_temporal = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv_temporal = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_temporal = nn.Linear(d_model, d_model, bias=False)

        self.norm_cross = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.q_cross = nn.Linear(d_model, d_model, bias=False)
        self.kv_cross = nn.Linear(context_dim, 2 * d_model, bias=False)
        self.out_cross = nn.Linear(d_model, d_model, bias=False)

        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(d_model, int(d_model * mlp_ratio))

        # shift/scale/gate for each of the four sub-layers.
        self.modulation = nn.Linear(d_model, 12 * d_model)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def _attend(self, qkv_proj, out_proj, x, dropout_p):
        batch, length, channels = x.shape
        q, k, v = (
            t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            for t in qkv_proj(x).chunk(3, dim=-1)
        )
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        return out_proj(attended.transpose(1, 2).reshape(batch, length, channels))

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        grid: tuple,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """``x`` is ``(B, T*H*W, C)`` with ``grid = (T, H, W)``."""
        B, N, C = x.shape
        T, H, W = grid
        S = H * W
        dropout_p = self.dropout_p if self.training else 0.0
        params = self.modulation(F.silu(conditioning)).chunk(12, dim=-1)

        # Spatial: every position attends within its own frame.
        h = modulate(self.norm_spatial(x), params[0], params[1])
        h = h.view(B, T, S, C).reshape(B * T, S, C)
        h = self._attend(self.qkv_spatial, self.out_spatial, h, dropout_p)
        h = h.view(B, T, S, C).reshape(B, N, C)
        x = x + params[2].unsqueeze(1) * h

        # Temporal: every position attends across frames at its own location.
        h = modulate(self.norm_temporal(x), params[3], params[4])
        h = h.view(B, T, S, C).transpose(1, 2).reshape(B * S, T, C)
        h = self._attend(self.qkv_temporal, self.out_temporal, h, dropout_p)
        h = h.view(B, S, T, C).transpose(1, 2).reshape(B, N, C)
        x = x + params[5].unsqueeze(1) * h

        if context is not None:
            h = modulate(self.norm_cross(x), params[6], params[7])
            q = self.q_cross(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            k, v = (
                t.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
                for t in self.kv_cross(context).chunk(2, dim=-1)
            )
            crossed = F.scaled_dot_product_attention(q, k, v)
            crossed = self.out_cross(crossed.transpose(1, 2).reshape(B, N, C))
            x = x + params[8].unsqueeze(1) * crossed

        h = modulate(self.norm_mlp(x), params[9], params[10])
        return x + params[11].unsqueeze(1) * self.mlp(h)


class PatchEmbed3D(nn.Module):
    """Patchify a latent volume into a token sequence."""

    def __init__(self, in_channels: int, d_model: int, patch_size=(1, 2, 2)):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv3d(in_channels, d_model, self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> tuple:
        _, _, frames, height, width = x.shape
        pt, ph, pw = self.patch_size
        if frames % pt or height % ph or width % pw:
            raise ValueError(
                f"latent {frames}x{height}x{width} must be divisible by "
                f"video_dit_patch_size {self.patch_size}"
            )
        tokens = self.proj(x)
        grid = (tokens.shape[2], tokens.shape[3], tokens.shape[4])
        return tokens.flatten(2).transpose(1, 2), grid


class SpacetimeDiT(nn.Module):
    """Diffusion transformer over a video latent volume."""

    def __init__(
        self,
        in_channels: int = 8,
        d_model: int = 1536,
        n_layers: int = 24,
        n_heads: int = 12,
        patch_size=(1, 2, 2),
        context_dim: int = 1024,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(
                f"video_dit_d_model ({d_model}) must be divisible by "
                f"video_dit_n_heads ({n_heads})"
            )
        self.in_channels = in_channels
        self.patch_size = tuple(patch_size)
        self.d_model = d_model

        self.patch_embed = PatchEmbed3D(in_channels, d_model, patch_size)
        self.time_embed = TimestepEmbedder(d_model)
        self.context_pool = nn.Linear(context_dim, d_model)
        # Frame rate is a conditioning signal, not a fixed property: the same
        # weights should produce a slow pan and a fast action shot.
        self.fps_embed = TimestepEmbedder(d_model)

        self.blocks = nn.ModuleList([
            SpacetimeDiTBlock(d_model, n_heads, context_dim, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.modulation_out = nn.Linear(d_model, 2 * d_model)
        pt, ph, pw = self.patch_size
        self.proj_out = nn.Linear(d_model, pt * ph * pw * in_channels)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.modulation_out.weight)
        nn.init.zeros_(self.modulation_out.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def _unpatchify(self, tokens: torch.Tensor, grid: tuple) -> torch.Tensor:
        B = tokens.shape[0]
        t, h, w = grid
        pt, ph, pw = self.patch_size
        c = self.in_channels
        x = tokens.reshape(B, t, h, w, pt, ph, pw, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, c, t * pt, h * ph, w * pw)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor = None,
        fps: torch.Tensor = None,
    ) -> torch.Tensor:
        tokens, grid = self.patch_embed(z)
        tokens = tokens + sincos_pos_embed_3d(
            self.d_model, grid[0], grid[1], grid[2], device=tokens.device
        ).to(tokens.dtype)

        conditioning = self.time_embed(t)
        if context is not None:
            conditioning = conditioning + self.context_pool(context.mean(dim=1))
        if fps is not None:
            if fps.dim() == 0:
                fps = fps.expand(z.shape[0])
            conditioning = conditioning + self.fps_embed(fps)

        for block in self.blocks:
            tokens = block(tokens, conditioning, grid, context)

        shift, scale = self.modulation_out(F.silu(conditioning)).chunk(2, dim=-1)
        tokens = modulate(self.norm_out(tokens), shift, scale)
        return self._unpatchify(self.proj_out(tokens), grid)
