"""Utility functions for FramerAI model."""

import os

import torch
import torch.nn as nn


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_params(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def _count_on_meta(factory) -> int:
    """Parameter count of a module built on the meta device (allocates nothing).

    ``torch.device("meta")`` materializes shapes without storage, so a tower with
    billions of parameters can be counted on a laptop in milliseconds.
    """
    with torch.device("meta"):
        module = factory()
    return sum(p.numel() for p in module.parameters())


# The multimodal towers FramerModel builds, in report order. Single source of
# truth: build.py's breakdown and the estimator tests both import this rather
# than repeating the tuple.
MULTIMODAL_TOWERS = (
    "vision_encoder", "vision_projector", "audio_encoder", "audio_projector",
    "image_vae", "image_diffusion", "video_vae", "video_diffusion", "audio_diffusion",
)


def estimate_multimodal_params(config, strict: bool = False) -> dict:
    """Count the multimodal towers of a config without allocating memory.

    Mirrors what :class:`model.framer.FramerModel` builds when
    ``config.text_only`` is False: the vision and audio encoders with their
    projectors, and the image / video / audio diffusion decoders. Returns a
    per-tower breakdown plus ``total``.

    ``AudioGenerator`` is counted through its inner ``DiffusionModule``: every
    other tensor it owns is a registered buffer (mel pseudo-inverse, windows),
    which carries no trainable parameters.

    Returns all-zero counts for a text-only config, and ``None`` if a tower
    cannot be built on the meta device, so callers can degrade to a text-only
    estimate instead of failing. Pass ``strict=True`` to re-raise instead: a typo
    in a new tower should fail loudly in tests rather than quietly downgrade
    ``--estimate`` to "could not be sized".
    """
    if config.text_only:
        return {**{name: 0 for name in MULTIMODAL_TOWERS}, "total": 0}

    from ..modules.audio_encoder import AudioEncoder
    from ..modules.diffusion import DiffusionModule
    from ..modules.dit import DiT
    from ..modules.multimodal_projector import MultimodalProjector
    from ..modules.spacetime_dit import SpacetimeDiT
    from ..modules.vae import KLVAE
    from ..modules.video_generator import VideoGenerator
    from ..modules.video_vae import CausalVideoVAE
    from ..modules.vision_encoder import VisionEncoder

    d = config.d_model
    factories = {
        "vision_encoder": lambda: VisionEncoder(
            config.image_size, config.patch_size, config.vision_d_model,
            config.vision_n_heads, config.vision_n_layers, config.dropout,
        ),
        "vision_projector": lambda: MultimodalProjector(config.vision_d_model, d),
        "audio_encoder": lambda: AudioEncoder(
            sample_rate=config.audio_sample_rate, n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length, n_mels=config.audio_n_mels,
            d_model=config.audio_d_model, n_heads=config.audio_n_heads,
            n_layers=config.audio_n_layers, max_frames=config.audio_max_frames,
            dropout=config.dropout,
        ),
        "audio_projector": lambda: MultimodalProjector(config.audio_d_model, d),
        # The image decoder is arch-selected. Under "latent_dit" the autoencoder
        # is counted separately from the denoiser, since they are trained
        # separately and sized independently; under "unet" there is no VAE and
        # the slot is zero.
        "image_vae": lambda: (
            KLVAE(
                in_channels=3, latent_channels=config.vae_latent_channels,
                base_channels=config.vae_base_channels, downsample=config.vae_downsample,
            )
            if config.image_gen_arch == "latent_dit"
            else nn.Identity()
        ),
        "image_diffusion": lambda: (
            DiT(
                in_channels=config.vae_latent_channels, d_model=config.dit_d_model,
                n_layers=config.dit_n_layers, n_heads=config.dit_n_heads,
                patch_size=config.dit_patch_size, context_dim=d,
            )
            if config.image_gen_arch == "latent_dit"
            else DiffusionModule(
                in_channels=3, base_channels=config.diffusion_channels, context_dim=d,
                num_steps=config.diffusion_steps,
            )
        ),
        "video_vae": lambda: (
            CausalVideoVAE(
                in_channels=3, latent_channels=config.video_vae_latent_channels,
                base_channels=config.video_vae_base_channels,
                temporal_downsample=config.video_vae_temporal_downsample,
                spatial_downsample=config.video_vae_spatial_downsample,
            )
            if config.video_gen_arch == "spacetime_dit"
            else nn.Identity()
        ),
        "video_diffusion": lambda: (
            SpacetimeDiT(
                in_channels=config.video_vae_latent_channels,
                d_model=config.video_dit_d_model, n_layers=config.video_dit_n_layers,
                n_heads=config.video_dit_n_heads, patch_size=config.video_dit_patch_size,
                context_dim=d,
            )
            if config.video_gen_arch == "spacetime_dit"
            else VideoGenerator(
                frames=config.video_frames, resolution=config.video_resolution,
                base_channels=config.diffusion_channels // 2, context_dim=d,
                num_steps=config.diffusion_steps,
            )
        ),
        # AudioGenerator's parameters all live in this inner diffusion module.
        "audio_diffusion": lambda: DiffusionModule(
            in_channels=1, base_channels=config.audio_gen_channels, context_dim=d,
            num_steps=config.diffusion_steps,
        ),
    }

    assert tuple(factories) == MULTIMODAL_TOWERS, "tower registry drifted from MULTIMODAL_TOWERS"

    counts = {}
    try:
        for name, factory in factories.items():
            counts[name] = _count_on_meta(factory)
    except Exception:  # noqa: BLE001 - any tower that resists meta construction
        if strict:
            raise
        return None
    counts["total"] = sum(counts.values())
    return counts


def estimate_params(config, strict: bool = False) -> dict:
    """Estimate a config's parameter budget without instantiating the model.

    ``total`` and ``active`` cover the **text backbone**: total parameters versus
    the parameters a single text token actually routes through (they differ only
    for MoE). ``multimodal`` breaks down the encoders and diffusion decoders, and
    ``model_total`` is the whole model - the number that describes FramerAI as a
    multimodal system.

    "Active" has two senses worth keeping straight: ``active`` is per-text-token
    compute, while the vision and audio encoders run only on requests carrying an
    image or audio, and each diffusion decoder runs only when generating that
    modality. None of them are counted in ``active``.

    Nothing here allocates memory, so a trillion-parameter preset can be sized on
    a laptop.
    """
    d = config.d_model
    n_layers = config.n_layers
    head_dim = d // config.n_heads
    kv_dim = config.kv_heads * head_dim

    # Attention block: q(d*d) + out(d*d) + k(d*kv) + v(d*kv)
    attn = 2 * d * d + 2 * d * kv_dim
    if config.use_qk_norm:
        attn += 2 * head_dim  # per-head RMSNorm on q and k
    # Two RMSNorms per block.
    norm = 2 * d

    def dense_ffn(width):
        return 3 * d * width  # SwiGLU: w1, w2, w3

    total = 0
    active = 0
    for i in range(n_layers):
        layer_total = attn + norm
        layer_active = attn + norm
        if config.is_moe_layer(i):
            ew = config.expert_d_ff or config.d_ff
            router = d * config.n_experts
            experts = config.n_experts * dense_ffn(ew)
            shared = config.n_shared_experts * dense_ffn(ew)
            layer_total += router + experts + shared
            layer_active += router + config.n_experts_per_tok * dense_ffn(ew) + shared
        else:
            layer_total += dense_ffn(config.d_ff)
            layer_active += dense_ffn(config.d_ff)
        total += layer_total
        active += layer_active

    embed = config.vocab_size * d  # tied embedding / lm head (counted once)
    total += embed + d  # + final norm
    active += embed + d

    multimodal = estimate_multimodal_params(config, strict=strict)
    mm_total = multimodal["total"] if multimodal else 0
    model_total = total + mm_total

    return {
        "total": total,
        "active": active,
        "total_h": human_params(total),
        "active_h": human_params(active),
        "multimodal": multimodal,
        "multimodal_total": mm_total,
        "multimodal_h": human_params(mm_total),
        "model_total": model_total,
        "model_total_h": human_params(model_total),
        # Weight bytes in bf16, and the full AdamW training state: bf16 weights,
        # an fp32 master copy, and two fp32 moments (2 + 4 + 4 + 4 = 14 bytes,
        # rounded to 16 to leave room for gradients).
        "bf16_bytes": 2 * model_total,
        "adamw_bytes": 16 * model_total,
        "summary": (
            f"{config.preset or 'custom'}: {human_params(total)} text params, "
            f"{human_params(active)} active/token"
            + (" (MoE)" if config.use_moe else " (dense)")
            + (f" + {human_params(mm_total)} multimodal = {human_params(model_total)} total" if mm_total else "")
        ),
    }


def save_checkpoint(model: nn.Module, optimizer, step: int, loss: float, path: str):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("step", 0), checkpoint.get("loss", float("inf"))
