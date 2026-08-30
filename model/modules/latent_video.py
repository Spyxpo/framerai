"""Latent-space video generation: causal 3D VAE, spacetime DiT, rectified flow.

A drop-in sibling of :class:`~model.modules.video_generator.VideoGenerator`,
exposing the same ``forward(video, context) -> loss`` and
``sample(batch_size, context, device)`` surface so ``FramerModel`` and
``FramerGenerator`` swap through a factory and need no other change.

Underneath it is the video counterpart of the latent image path: diffusion runs
in a 4x-temporal, 8x-spatial compressed latent, the denoiser is a transformer
with factorised spacetime attention rather than a 3D U-Net with a per-frame
Python loop, and sampling is a 20-50 step ODE solve with classifier-free
guidance rather than a 1000-step ancestral chain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import ODESampler, RectifiedFlow
from .spacetime_dit import SpacetimeDiT
from .video_vae import CausalVideoVAE


class LatentVideoGenerator(nn.Module):
    """Text-conditioned latent video diffusion with guidance."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frames = config.video_frames
        self.resolution = config.video_resolution
        self.fps = config.video_fps
        self.cfg_dropout_prob = config.cfg_dropout_prob
        self.cfg_scale = config.cfg_scale

        self.vae = CausalVideoVAE(
            in_channels=3,
            latent_channels=config.video_vae_latent_channels,
            base_channels=config.video_vae_base_channels,
            temporal_downsample=config.video_vae_temporal_downsample,
            spatial_downsample=config.video_vae_spatial_downsample,
        )
        self.denoiser = SpacetimeDiT(
            in_channels=config.video_vae_latent_channels,
            d_model=config.video_dit_d_model,
            n_layers=config.video_dit_n_layers,
            n_heads=config.video_dit_n_heads,
            patch_size=config.video_dit_patch_size,
            context_dim=config.d_model,
            dropout=config.dropout,
        )
        self.flow = RectifiedFlow()
        self.sampler = ODESampler(config.sampler_steps, config.sampler_method)
        self.null_context = nn.Parameter(torch.zeros(1, 1, config.d_model))

    @torch.no_grad()
    def reset_parameters(self):
        self.null_context.zero_()

    def _drop_context(self, context: torch.Tensor) -> torch.Tensor:
        if context is None or not self.training or self.cfg_dropout_prob <= 0:
            return context
        keep = torch.rand(context.shape[0], device=context.device) >= self.cfg_dropout_prob
        return torch.where(keep.view(-1, 1, 1), context, self.null_context.to(context.dtype))

    def forward(self, video: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training loss over a batch of clips, ``(B, C, T, H, W)``."""
        with torch.no_grad():
            latents = self.vae.encode_to_latent(video)

        context = self._drop_context(context)
        noise = torch.randn_like(latents)
        t = self.flow.sample_t(latents.shape[0], device=latents.device)
        noisy = self.flow.interpolate(noise, latents, t)
        target = self.flow.target(noise, latents)

        fps = torch.full((latents.shape[0],), float(self.fps), device=latents.device)
        velocity = self.denoiser(noisy, t, context, fps=fps)
        return self.flow.loss(velocity, target)

    def autoencoder_loss(self, video: torch.Tensor, kl_weight: float = 1e-6) -> torch.Tensor:
        recon, kl = self.vae(video)
        return F.mse_loss(recon, video) + kl_weight * kl

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        context: torch.Tensor = None,
        device: str = "cpu",
        frames: int = None,
        height: int = None,
        width: int = None,
        fps: int = None,
        cfg_scale: float = None,
        steps: int = None,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """Generate a clip. Duration and resolution are per request."""
        frames = frames or self.frames
        height = height or self.resolution
        width = width or self.resolution
        fps_value = float(fps or self.fps)

        latent_shape = self.vae.latent_shape(batch_size, frames, height, width)
        fps_batch = torch.full((batch_size,), fps_value, device=device)

        def velocity_fn(z, t, ctx):
            return self.denoiser(z, t, ctx, fps=fps_batch[: z.shape[0]])

        latents = self.sampler.sample(
            velocity_fn,
            latent_shape,
            context=context,
            null_context=self.null_context,
            cfg_scale=self.cfg_scale if cfg_scale is None else cfg_scale,
            device=device,
            steps=steps,
            generator=generator,
        )
        video = self.vae.decode(latents).clamp(-1, 1)
        # The VAE's temporal upsampling works in powers of two, so a requested
        # duration that is not a multiple of the compression factor comes back
        # slightly long. Trim rather than surprise the caller.
        return video[:, :, :frames]


    @torch.no_grad()
    def sample_long(
        self,
        batch_size: int = 1,
        context: torch.Tensor = None,
        device: str = "cpu",
        frames: int = None,
        height: int = None,
        width: int = None,
        fps: int = None,
        cfg_scale: float = None,
        steps: int = None,
        generator: torch.Generator = None,
        window_frames: int = None,
        overlap_frames: int = None,
    ) -> torch.Tensor:
        """Generate a clip longer than one denoising window.

        A single window bounds duration by whatever fits in memory at once, so
        a long shot could only be made as separate clips, and separate clips do
        not join: the seam between them is a visible cut, which is exactly what
        the temporal consistency metric exists to catch.

        Here the windows overlap. Each one after the first has its opening
        latent frames held to the closing frames of the one before, through the
        solver's clamp hook, so content carries across the join instead of being
        reinvented. The causal VAE decodes the assembled latents in one pass,
        which is what it was built for: frame t never sees frame t+1, so a
        longer sequence is not a different computation.

        Memory now scales with the window rather than with the duration.
        """
        # Absence and zero are distinguished: `x or default` would turn an
        # explicit 0 into a default and skip the guards below entirely.
        frames = self.frames if frames is None else frames
        height = self.resolution if height is None else height
        width = self.resolution if width is None else width
        window_frames = self.frames if window_frames is None else window_frames
        overlap_frames = self.frames // 4 if overlap_frames is None else overlap_frames

        if window_frames < 1:
            raise ValueError(f"window_frames must be positive, got {window_frames}")
        if not 0 <= overlap_frames < window_frames:
            raise ValueError(
                f"overlap_frames ({overlap_frames}) must be at least 0 and "
                f"less than window_frames ({window_frames})"
            )

        # Everything below is in latent frames: that is what the solver works
        # in, and what the carry-over has to line up with.
        _, latent_channels, window_latents, latent_h, latent_w = self.vae.latent_shape(
            batch_size, window_frames, height, width
        )
        window_shape = (batch_size, latent_channels, window_latents, latent_h, latent_w)
        overlap_latents = (
            0 if overlap_frames == 0
            else min(window_latents - 1, self.vae.latent_shape(batch_size, overlap_frames, height, width)[2])
        )
        total_latents = self.vae.latent_shape(batch_size, frames, height, width)[2]

        fps_value = float(fps or self.fps)
        fps_batch = torch.full((batch_size,), fps_value, device=device)

        def velocity_fn(z, t, ctx):
            return self.denoiser(z, t, ctx, fps=fps_batch[: z.shape[0]])

        scale = self.cfg_scale if cfg_scale is None else cfg_scale
        assembled, carried = None, None

        while assembled is None or assembled.shape[2] < total_latents:
            clamp_fn = None
            if carried is not None and overlap_latents:
                held = carried

                def clamp_fn(x, t, held=held):
                    # The opening frames are already decided, so the solver is
                    # told so at every step rather than being asked to rediscover
                    # them and then corrected once at the end.
                    x = x.clone()
                    x[:, :, :overlap_latents] = held
                    return x

            window = self.sampler.sample(
                velocity_fn, window_shape, context=context, null_context=self.null_context,
                cfg_scale=scale, device=device, steps=steps, generator=generator,
                clamp_fn=clamp_fn,
            )

            if assembled is None:
                assembled = window
            else:
                # The overlap is already in the assembled sequence.
                assembled = torch.cat([assembled, window[:, :, overlap_latents:]], dim=2)

            if overlap_latents:
                carried = assembled[:, :, -overlap_latents:]
            if window_latents - overlap_latents <= 0:  # pragma: no cover - guarded above
                break

        video = self.vae.decode(assembled[:, :, :total_latents]).clamp(-1, 1)
        return video[:, :, :frames]


def build_video_generator(config):
    """Return the video generator the config selects."""
    from .video_generator import VideoGenerator

    if config.video_gen_arch == "spacetime_dit":
        return LatentVideoGenerator(config)
    return VideoGenerator(
        frames=config.video_frames,
        resolution=config.video_resolution,
        base_channels=config.diffusion_channels // 2,
        context_dim=config.d_model,
        num_steps=config.diffusion_steps,
    )
