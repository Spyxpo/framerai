"""Video generation module with temporal attention (Sora-like architecture)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Attention across temporal (frame) dimension."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H*W, C) where T is frames
        B, T, S, C = x.shape
        h = self.norm(x)

        # Reshape to attend across time for each spatial position
        h = h.permute(0, 2, 1, 3).reshape(B * S, T, C)

        qkv = self.qkv(h).reshape(B * S, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B * S, T, C)
        out = self.out(out)
        out = out.reshape(B, S, T, C).permute(0, 2, 1, 3)

        return x + out


class SpatialTemporalBlock(nn.Module):
    """Combined spatial and temporal attention block for video."""

    def __init__(self, channels: int, n_heads: int = 8, time_dim: int = None, context_dim: int = 1024):
        super().__init__()
        time_dim = time_dim or channels * 4

        # Spatial processing
        self.spatial_norm = nn.GroupNorm(32, channels)
        self.spatial_conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.spatial_conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        # Time embedding
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels))

        # Temporal attention (operates per-pixel across frames)
        self.temporal_attn = TemporalAttention(channels, n_heads)

        # Cross attention for text conditioning
        self.cross_norm = nn.LayerNorm(channels)
        self.cross_q = nn.Linear(channels, channels, bias=False)
        self.cross_kv = nn.Linear(context_dim, channels * 2, bias=False)
        self.cross_out = nn.Linear(channels, channels, bias=False)
        self.cross_heads = n_heads
        self.cross_head_dim = channels // n_heads

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Spatial processing per-frame
        frames = []
        for i in range(T):
            frame = x[:, :, i]
            h = self.spatial_norm(frame)
            h = F.silu(self.spatial_conv1(h))
            h = h + self.time_proj(t_emb)[:, :, None, None]
            h = self.spatial_conv2(h)
            frames.append(frame + h)
        x_spatial = torch.stack(frames, dim=2)

        # Temporal attention
        x_flat = x_spatial.reshape(B, C, T, H * W).permute(0, 2, 3, 1)  # (B, T, H*W, C)
        x_temporal = self.temporal_attn(x_flat)
        x_out = x_temporal.permute(0, 3, 1, 2).reshape(B, C, T, H, W)

        # Cross attention with text
        if context is not None:
            x_cross = x_out.reshape(B, C, T * H * W).permute(0, 2, 1)  # (B, T*H*W, C)
            h = self.cross_norm(x_cross)
            q = self.cross_q(h).reshape(B, -1, self.cross_heads, self.cross_head_dim).transpose(1, 2)
            kv = self.cross_kv(context).reshape(B, -1, 2, self.cross_heads, self.cross_head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            scale = math.sqrt(self.cross_head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            cross_out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(B, T * H * W, C)
            cross_out = self.cross_out(cross_out)
            x_out = x_out + cross_out.permute(0, 2, 1).reshape(B, C, T, H, W)

        return x_out


class VideoUNet(nn.Module):
    """3D U-Net for video diffusion with temporal layers."""

    def __init__(self, in_channels: int = 3, base_channels: int = 128, context_dim: int = 1024):
        super().__init__()
        ch = base_channels
        time_dim = ch * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input
        self.conv_in = nn.Conv3d(in_channels, ch, 3, padding=1)

        # Encoder
        self.enc1 = SpatialTemporalBlock(ch, 8, time_dim, context_dim)
        self.down1 = nn.Conv3d(ch, ch * 2, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.enc2 = SpatialTemporalBlock(ch * 2, 8, time_dim, context_dim)
        self.down2 = nn.Conv3d(ch * 2, ch * 4, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Bottleneck
        self.mid = SpatialTemporalBlock(ch * 4, 8, time_dim, context_dim)

        # Decoder
        self.up2 = nn.ConvTranspose3d(ch * 4, ch * 2, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dec2 = SpatialTemporalBlock(ch * 4, 8, time_dim, context_dim)  # ch*4 due to skip

        self.up1 = nn.ConvTranspose3d(ch * 2, ch, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dec1 = SpatialTemporalBlock(ch * 2, 8, time_dim, context_dim)  # ch*2 due to skip

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv3d(ch, in_channels, 3, padding=1),
        )

    def _sinusoidal_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # x: (B, C, T, H, W)
        t_emb = self.time_embed(self._sinusoidal_embedding(t, x.shape[1] * 4 // 4))
        # Adjust time embed dim
        t_emb_ch = self._sinusoidal_embedding(t, self.conv_in.out_channels)
        t_emb = self.time_embed(t_emb_ch)

        h = self.conv_in(x)

        h1 = self.enc1(h, t_emb, context)
        h1_down = self.down1(h1)

        h2 = self.enc2(h1_down, t_emb, context)
        h2_down = self.down2(h2)

        mid = self.mid(h2_down, t_emb, context)

        up = self.up2(mid)
        up = torch.cat([up, h2], dim=1)
        up = self.dec2(up, t_emb, context)

        # Reduce channels back before up1
        up = up[:, :up.shape[1] // 2]
        up = self.up1(up)
        up = torch.cat([up, h1], dim=1)
        up = self.dec1(up, t_emb, context)

        # Reduce channels
        up = up[:, :up.shape[1] // 2]
        return self.conv_out(up)


class VideoGenerator(nn.Module):
    """Video generation via diffusion with spatial-temporal architecture."""

    def __init__(
        self,
        frames: int = 16,
        resolution: int = 256,
        base_channels: int = 128,
        context_dim: int = 1024,
        num_steps: int = 1000,
    ):
        super().__init__()
        self.frames = frames
        self.resolution = resolution
        self.num_steps = num_steps
        self.unet = VideoUNet(3, base_channels, context_dim)

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1 - alpha_cumprod))

    def forward(self, video: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training: predict noise added to video."""
        B = video.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=video.device)
        noise = torch.randn_like(video)

        sqrt_a = self.sqrt_alpha_cumprod[t].view(B, 1, 1, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alpha_cumprod[t].view(B, 1, 1, 1, 1)
        noisy = sqrt_a * video + sqrt_1ma * noise

        pred = self.unet(noisy, t, context)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, batch_size: int = 1, context: torch.Tensor = None, device: str = "cpu") -> torch.Tensor:
        """Generate video frames via reverse diffusion."""
        shape = (batch_size, 3, self.frames, self.resolution, self.resolution)
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t_batch, context)

            alpha = self.alphas[t]
            alpha_cp = self.alpha_cumprod[t]

            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cp)) * pred_noise)

            if t > 0:
                x = x + torch.sqrt(self.betas[t]) * torch.randn_like(x)

        return x.clamp(-1, 1)
