"""Vision encoder for processing image inputs."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""

    def __init__(self, image_size: int = 256, patch_size: int = 16, in_channels: int = 3, d_model: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model) * 0.02)

    @torch.no_grad()
    def reset_parameters(self):
        self.cls_token.normal_(std=0.02)
        self.pos_embed.normal_(std=0.02)

    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        """Resample the learned position table to a different patch grid.

        The table is trained at one resolution. Bicubic resampling of its 2D
        reshape lets the same weights accept a different one, which is what a
        tile of unusual shape needs. The class-token position is not part of the
        grid and passes through untouched.
        """
        patches = height * width
        if patches == self.num_patches:
            return self.pos_embed

        cls_pos, grid_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        side = int(self.num_patches ** 0.5)
        grid = grid_pos.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(height, width), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, patches, -1)
        return torch.cat([cls_pos, grid], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x)
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.interpolate_pos_encoding(grid_h, grid_w)


class DynamicTiler(nn.Module):
    """Split a high-resolution image into encoder-sized tiles.

    A fixed-resolution encoder either downsamples a large image until detail is
    gone or runs at a resolution whose attention cost is prohibitive. Tiling
    keeps the encoder at its trained size and processes several tiles instead:
    ``max_tiles`` of ``tile_size`` covers roughly ``sqrt(max_tiles)`` times the
    linear resolution at a fraction of a single large forward's cost, because
    attention is quadratic within a tile but only linear across them.

    A downscaled thumbnail is prepended so the model still sees global layout,
    which tiles alone destroy.
    """

    def __init__(self, tile_size: int, max_tiles: int = 12, thumbnail: bool = True):
        super().__init__()
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        self.thumbnail = thumbnail

    def choose_grid(self, height: int, width: int) -> tuple:
        """The (rows, cols) tiling whose aspect ratio best matches the image."""
        target = width / height
        best, best_error = (1, 1), float("inf")
        for rows in range(1, self.max_tiles + 1):
            for cols in range(1, self.max_tiles // rows + 1):
                if rows * cols > self.max_tiles:
                    continue
                # Compared in log space so 2:1 and 1:2 are equally far from square.
                error = abs(math.log((cols / rows) / target))
                # Prefer more tiles on a tie: more coverage at the same shape.
                if error < best_error - 1e-9 or (
                    abs(error - best_error) < 1e-9 and rows * cols > best[0] * best[1]
                ):
                    best, best_error = (rows, cols), error
        return best

    def forward(self, image: torch.Tensor) -> tuple:
        """``image`` is ``(B, C, H, W)``; returns ``(tiles, grid)``.

        ``tiles`` is ``(B, n_tiles, C, tile, tile)``, thumbnail first when
        enabled, so the caller can flatten it into one encoder batch.
        """
        B, C, height, width = image.shape
        rows, cols = self.choose_grid(height, width)

        resized = F.interpolate(
            image,
            size=(rows * self.tile_size, cols * self.tile_size),
            mode="bicubic",
            align_corners=False,
        )
        tiles = (
            resized.unfold(2, self.tile_size, self.tile_size)
            .unfold(3, self.tile_size, self.tile_size)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(B, rows * cols, C, self.tile_size, self.tile_size)
        )

        if self.thumbnail:
            global_view = F.interpolate(
                image, size=(self.tile_size, self.tile_size),
                mode="bicubic", align_corners=False,
            ).unsqueeze(1)
            tiles = torch.cat([global_view, tiles], dim=1)

        return tiles, (rows, cols)


class VisionAttention(nn.Module):
    """Multi-head self-attention for vision transformer.

    Uses the fused SDPA kernel rather than materializing a ``(B, heads, T, T)``
    matrix. The explicit form made the encoder's memory cost quadratic in the
    patch count and retained for backward across every layer, which put the
    large presets' configured resolutions out of reach on any hardware.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class VisionBlock(nn.Module):
    """Vision transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = VisionAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Full vision encoder (ViT-style) for image understanding."""

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, d_model)
        self.blocks = nn.ModuleList([
            VisionBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
