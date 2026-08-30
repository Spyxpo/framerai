"""Latent-space image generation: VAE, diffusion transformer, rectified flow.

A drop-in sibling of :class:`~model.modules.diffusion.DiffusionModule`. It
exposes the same ``forward(images, context) -> loss`` and
``sample(shape, context, device)`` surface, so ``FramerModel`` and
``FramerGenerator`` swap between the two through a factory and need no other
change.

What differs is everything underneath. Diffusion runs in an 8x-compressed latent
rather than on pixels, the denoiser is a transformer with adaLN-zero rather than
a convolutional U-Net, the objective is rectified flow solved in tens of ODE
steps rather than a 1000-step ancestral chain, and classifier-free guidance is
implemented against a learned null-context embedding.

That last item is worth stating plainly: the README advertised classifier-free
diffusion from the beginning and no such code existed. This is it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import DiT
from .flow import ODESampler, RectifiedFlow
from .vae import KLVAE


class LatentImageGenerator(nn.Module):
    """Text-conditioned latent diffusion with classifier-free guidance."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cfg_dropout_prob = config.cfg_dropout_prob
        self.cfg_scale = config.cfg_scale
        self.objective = config.diffusion_objective

        self.vae = KLVAE(
            in_channels=3,
            latent_channels=config.vae_latent_channels,
            base_channels=config.vae_base_channels,
            downsample=config.vae_downsample,
        )
        self.denoiser = DiT(
            in_channels=config.vae_latent_channels,
            d_model=config.dit_d_model,
            n_layers=config.dit_n_layers,
            n_heads=config.dit_n_heads,
            patch_size=config.dit_patch_size,
            context_dim=config.d_model,
            dropout=config.dropout,
        )
        self.flow = RectifiedFlow()
        # A distilled student runs in single-digit steps and needs no guidance
        # pair, so both savings are selected together rather than separately.
        self.flow_distilled = getattr(config, "flow_distilled", False)
        self.sampler = ODESampler(
            config.flow_distilled_steps if self.flow_distilled else config.sampler_steps,
            config.sampler_method,
            guidance_distilled=self.flow_distilled,
        )

        # The learned unconditional embedding. Guidance extrapolates away from
        # this, so it has to be trained alongside the conditional path - which is
        # what cfg_dropout_prob does.
        self.null_context = nn.Parameter(torch.zeros(1, 1, config.d_model))

    @torch.no_grad()
    def reset_parameters(self):
        self.null_context.zero_()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _drop_context(self, context: torch.Tensor) -> torch.Tensor:
        """Replace whole examples' conditioning with the null embedding."""
        if context is None or not self.training or self.cfg_dropout_prob <= 0:
            return context
        keep = torch.rand(context.shape[0], device=context.device) >= self.cfg_dropout_prob
        keep = keep.view(-1, 1, 1)
        return torch.where(keep, context, self.null_context.to(context.dtype))

    def forward(self, images: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training loss over a batch of images.

        The VAE encodes under ``no_grad``: it is trained separately by
        :meth:`autoencoder_loss`, and letting the diffusion loss reshape the
        latent space is how latent diffusion training goes wrong.
        """
        with torch.no_grad():
            latents = self.vae.encode_to_latent(images)

        context = self._drop_context(context)
        noise = torch.randn_like(latents)
        t = self.flow.sample_t(latents.shape[0], device=latents.device)

        if self.objective == "rectified_flow":
            noisy = self.flow.interpolate(noise, latents, t)
            target = self.flow.target(noise, latents)
        else:  # ddpm-style epsilon prediction over the same latent
            noisy = self.flow.interpolate(noise, latents, t)
            target = noise

        velocity = self.denoiser(noisy, t, context)
        return self.flow.loss(velocity, target)

    def autoencoder_loss(self, images: torch.Tensor, kl_weight: float = 1e-6) -> torch.Tensor:
        """Reconstruction plus KL, for pretraining the VAE on its own."""
        recon, kl = self.vae(images)
        return F.mse_loss(recon, images) + kl_weight * kl

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        context: torch.Tensor = None,
        device: str = "cpu",
        cfg_scale: float = None,
        steps: int = None,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """Generate images. ``shape`` is in pixel space, ``(B, C, H, W)``.

        Matches ``DiffusionModule.sample``'s signature so callers do not branch.
        """
        batch, _, height, width = shape
        latent_shape = self.vae.latent_shape(batch, height, width)
        scale = self.cfg_scale if cfg_scale is None else cfg_scale

        latents = self.sampler.sample(
            self.denoiser,
            latent_shape,
            context=context,
            null_context=self.null_context,
            cfg_scale=scale,
            device=device,
            steps=steps,
            generator=generator,
        )
        return self.vae.decode(latents).clamp(-1, 1)


def build_image_generator(config):
    """Return the image generator the config selects.

    ``unet`` keeps the original pixel-space path, which is what the small presets
    run on. ``latent_dit`` is the latent transformer above.
    """
    from .diffusion import DiffusionModule

    if config.image_gen_arch == "latent_dit":
        return LatentImageGenerator(config)
    return DiffusionModule(
        in_channels=3,
        base_channels=config.diffusion_channels,
        context_dim=config.d_model,
        num_steps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
