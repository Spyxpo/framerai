"""Diffusion model for image generation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseScheduler:
    """Linear noise scheduler for diffusion process."""

    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1).to(x.device)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1).to(x.device)
        return sqrt_alpha * x + sqrt_one_minus * noise

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    """Self-attention over spatial dimensions."""

    def __init__(self, channels: int, n_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out = nn.Conv1d(channels, channels, 1)
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h).reshape(B, 3, self.n_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        scale = math.sqrt(self.head_dim)
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) / scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(B, C, H * W)
        out = self.out(out).view(B, C, H, W)
        return x + out


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""

    def __init__(self, channels: int, context_dim: int, n_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.out = nn.Conv1d(channels, channels, 1)
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        q = self.q(h).reshape(B, self.n_heads, self.head_dim, H * W)

        kv = self.kv(context).reshape(B, -1, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 4, 1)
        k, v = kv[0], kv[1]

        scale = math.sqrt(self.head_dim)
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) / scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(B, C, H * W)
        out = self.out(out).view(B, C, H, W)
        return x + out


class UNet(nn.Module):
    """U-Net with text conditioning for diffusion image generation."""

    def __init__(self, in_channels: int = 3, base_channels: int = 256, context_dim: int = 1024):
        super().__init__()
        ch = base_channels
        time_dim = ch * 4
        self.time_embed = SinusoidalTimeEmbedding(ch)
        self.time_mlp = nn.Sequential(nn.Linear(ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        # Encoder
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down1 = nn.ModuleList([ResBlock(ch, ch, time_dim), SpatialAttention(ch), CrossAttention(ch, context_dim)])
        self.down_sample1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

        self.down2 = nn.ModuleList([ResBlock(ch, ch * 2, time_dim), SpatialAttention(ch * 2), CrossAttention(ch * 2, context_dim)])
        self.down_sample2 = nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1)

        self.down3 = nn.ModuleList([ResBlock(ch * 2, ch * 4, time_dim), SpatialAttention(ch * 4), CrossAttention(ch * 4, context_dim)])
        self.down_sample3 = nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.mid = nn.ModuleList([
            ResBlock(ch * 4, ch * 4, time_dim),
            SpatialAttention(ch * 4),
            CrossAttention(ch * 4, context_dim),
            ResBlock(ch * 4, ch * 4, time_dim),
        ])

        # Decoder
        self.up_sample3 = nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1)
        self.up3 = nn.ModuleList([ResBlock(ch * 8, ch * 2, time_dim), SpatialAttention(ch * 2), CrossAttention(ch * 2, context_dim)])

        self.up_sample2 = nn.ConvTranspose2d(ch * 2, ch * 2, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([ResBlock(ch * 4, ch, time_dim), SpatialAttention(ch), CrossAttention(ch, context_dim)])

        self.up_sample1 = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([ResBlock(ch * 2, ch, time_dim), SpatialAttention(ch), CrossAttention(ch, context_dim)])

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_embed(t))

        # Encoder
        h = self.conv_in(x)

        h1 = self.down1[0](h, t_emb)
        h1 = self.down1[1](h1)
        if context is not None:
            h1 = self.down1[2](h1, context)
        h1_down = self.down_sample1(h1)

        h2 = self.down2[0](h1_down, t_emb)
        h2 = self.down2[1](h2)
        if context is not None:
            h2 = self.down2[2](h2, context)
        h2_down = self.down_sample2(h2)

        h3 = self.down3[0](h2_down, t_emb)
        h3 = self.down3[1](h3)
        if context is not None:
            h3 = self.down3[2](h3, context)
        h3_down = self.down_sample3(h3)

        # Bottleneck
        mid = self.mid[0](h3_down, t_emb)
        mid = self.mid[1](mid)
        if context is not None:
            mid = self.mid[2](mid, context)
        mid = self.mid[3](mid, t_emb)

        # Decoder
        up = self.up_sample3(mid)
        up = torch.cat([up, h3], dim=1)
        up = self.up3[0](up, t_emb)
        up = self.up3[1](up)
        if context is not None:
            up = self.up3[2](up, context)

        up = self.up_sample2(up)
        up = torch.cat([up, h2], dim=1)
        up = self.up2[0](up, t_emb)
        up = self.up2[1](up)
        if context is not None:
            up = self.up2[2](up, context)

        up = self.up_sample1(up)
        up = torch.cat([up, h1], dim=1)
        up = self.up1[0](up, t_emb)
        up = self.up1[1](up)
        if context is not None:
            up = self.up1[2](up, context)

        return self.conv_out(up)


class DiffusionModule(nn.Module):
    """Complete diffusion module wrapping UNet and noise scheduler."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 256,
        context_dim: int = 1024,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.unet = UNet(in_channels, base_channels, context_dim)
        self.scheduler = NoiseScheduler(num_steps, beta_start, beta_end)
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training forward: add noise and predict it."""
        B = x.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=x.device)
        noise = torch.randn_like(x)
        noisy = self.scheduler.add_noise(x, noise, t)
        pred_noise = self.unet(noisy, t, context)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, shape: tuple, context: torch.Tensor = None, device: str = "cpu") -> torch.Tensor:
        """Generate images via reverse diffusion."""
        self.scheduler.to(device)
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t_batch, context)

            alpha = self.scheduler.alphas[t]
            alpha_cumprod = self.scheduler.alpha_cumprod[t]
            alpha_cumprod_prev = self.scheduler.alpha_cumprod_prev[t]

            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
            pred_x0 = pred_x0.clamp(-1, 1)

            mean = (torch.sqrt(alpha_cumprod_prev) * (1 - alpha) / (1 - alpha_cumprod)) * pred_x0
            mean += (torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * x

            if t > 0:
                noise = torch.randn_like(x)
                beta = self.scheduler.betas[t]
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

        return x.clamp(-1, 1)
