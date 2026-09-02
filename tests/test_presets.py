"""Preset registry + parameter estimator tests (incl. the 3T flagship)."""

import pytest

from conftest import tiny_config
from model.configs import PRESETS, FramerConfig, list_presets, resolve_preset_name
from model.framer import FramerModel
from model.utils.helpers import estimate_params

# The presets that must be multimodal at scale, not a big LLM with default towers.
LARGE_MULTIMODAL = [
    "framer-160b-a16b",
    "framer-200b-a20b",
    "framer-1t-a32b",
    "framer-2t-a49b",
    "framer-3t-a64b",
]

# Expert counts have to divide evenly across every mesh width the expert-parallel
# placement supports, or the largest presets cannot be sharded at all.
EP_MESHES = (8, 16, 32, 64, 128)


def test_all_presets_build_configs():
    for name in list_presets():
        cfg = FramerConfig.from_preset(name)
        assert cfg.d_model > 0 and cfg.n_layers > 0
        assert cfg.preset == name


def test_size_aliases():
    assert resolve_preset_name("tiny") == "framer-tiny"
    assert resolve_preset_name("large") == "framer-large"
    assert FramerConfig.from_preset("small").preset == "framer-small"


def test_preset_image_train_resolutions():
    assert FramerConfig.from_preset("framer-tiny").image_train_resolution == 64
    assert FramerConfig.from_preset("framer-tiny-moe").image_train_resolution == 64
    assert FramerConfig.from_preset("framer-small").image_train_resolution == 64
    assert FramerConfig.from_preset("framer-medium").image_train_resolution == 64
    assert FramerConfig.from_preset("framer-large").image_train_resolution == 64
    # Mid-size presets migrated to latent_dit for Issue #206
    assert FramerConfig.from_preset("framer-3b").image_train_resolution == 256
    assert FramerConfig.from_preset("framer-8b").image_train_resolution == 512
    assert FramerConfig.from_preset("framer-30b-a3b").image_train_resolution == 256
    # Large presets already on latent_dit
    assert FramerConfig.from_preset("framer-160b-a16b").image_train_resolution == 512
    assert FramerConfig.from_preset("framer-2t-a49b").image_train_resolution == 512


def test_validate_rejects_high_resolution_pixel_unet():
    with pytest.raises(ValueError, match="image_train_resolution.*cannot exceed 64"):
        FramerConfig(image_gen_arch="unet", image_train_resolution=128).validate()


def test_pixel_unet_forward_backward_pass():
    import torch
    cfg = FramerConfig.from_preset("framer-small")
    model = FramerModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    target_images = torch.randn(1, 3, cfg.image_train_resolution, cfg.image_train_resolution)
    out = model(input_ids=input_ids, target_images=target_images)
    assert "image_loss" in out
    assert torch.isfinite(out["image_loss"])
    out["image_loss"].backward()
    assert model.diffusion.unet.conv_in.weight.grad is not None


def test_trillion_preset_is_about_1t_without_instantiation():
    cfg = FramerConfig.from_preset("framer-1t-a32b")
    est = estimate_params(cfg)
    # ~1T total, far smaller active budget — the whole point of MoE.
    assert 8e11 < est["total"] < 1.2e12
    assert est["active"] < est["total"] / 10


def test_moe_active_less_than_total():
    dense = estimate_params(FramerConfig.from_preset("framer-8b"))
    assert dense["active"] == dense["total"]  # dense: active == total
    moe = estimate_params(FramerConfig.from_preset("framer-30b-a3b"))
    assert moe["active"] < moe["total"]


def test_estimator_matches_instantiated_text_only_model():
    cfg = tiny_config(vocab_size=256, d_model=64, n_layers=2, n_heads=8, n_kv_heads=2, d_ff=128)
    model = FramerModel(cfg)
    actual = sum(p.numel() for p in model.parameters())
    est = estimate_params(cfg)["total"]
    # Tied embedding is counted once in both; allow a small relative gap.
    assert abs(actual - est) / actual < 0.02


def test_trillion_preset_counts_every_modality():
    """The flagship's trillion is text + code + image + video + audio, not text alone."""
    est = estimate_params(FramerConfig.from_preset("framer-1t-a32b"))
    assert 9.5e11 < est["model_total"] < 1.05e12
    assert est["active"] < 4e10  # sparse: a text token routes through ~32B
    assert est["multimodal_total"] > 1e10  # towers scaled with the backbone
    assert est["model_total"] == est["total"] + est["multimodal_total"]


def test_flagship_preset_is_about_3t_across_every_modality():
    """The 3T flagship: three trillion of text, image, video, and audio together."""
    cfg = FramerConfig.from_preset("framer-3t-a64b")
    est = estimate_params(cfg)
    assert 2.95e12 < est["model_total"] < 3.1e12
    assert 6e10 < est["active"] < 7e10  # top-6 routing, ~64B per text token
    assert est["multimodal_total"] > 7e10  # towers scaled with the backbone
    assert est["model_total"] == est["total"] + est["multimodal_total"]


def test_flagship_towers_outgrow_the_2t_preset():
    """The point of the preset is the towers, not only the expert count."""
    two = estimate_params(FramerConfig.from_preset("framer-2t-a49b"))["multimodal"]
    three = estimate_params(FramerConfig.from_preset("framer-3t-a64b"))["multimodal"]
    for tower in ("vision_encoder", "audio_encoder", "image_diffusion", "video_diffusion"):
        assert three[tower] > two[tower], tower


@pytest.mark.parametrize("name", ["framer-2t-a49b", "framer-3t-a64b"])
def test_flagship_experts_divide_across_every_mesh(name):
    cfg = FramerConfig.from_preset(name)
    for mesh in EP_MESHES:
        assert cfg.n_experts % mesh == 0, f"{cfg.n_experts} experts do not shard {mesh} ways"


def test_200b_preset():
    est = estimate_params(FramerConfig.from_preset("framer-200b-a20b"))
    assert 1.9e11 < est["model_total"] < 2.15e11
    assert est["active"] < 2.5e10
    assert est["multimodal_total"] > 5e9


@pytest.mark.parametrize("name", LARGE_MULTIMODAL)
def test_large_presets_scale_their_multimodal_towers(name):
    """Guards against a large preset silently falling back to default tower sizes."""
    default = FramerConfig()
    cfg = FramerConfig.from_preset(name)
    assert not cfg.text_only
    assert cfg.vision_d_model > default.vision_d_model
    assert cfg.vision_n_layers > default.vision_n_layers
    assert cfg.audio_d_model > default.audio_d_model
    assert cfg.audio_n_layers > default.audio_n_layers
    assert cfg.diffusion_channels > default.diffusion_channels
    assert cfg.audio_gen_channels > default.audio_gen_channels


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_satisfies_module_shape_invariants(name):
    cfg = FramerConfig.from_preset(name)  # raises via validate() on a bad shape
    assert cfg.d_model % cfg.n_heads == 0
    assert cfg.n_heads % cfg.kv_heads == 0
    assert cfg.image_size % cfg.patch_size == 0
    # GroupNorm(32) in the image U-Net and in the half-width video U-Net.
    assert cfg.diffusion_channels % 64 == 0
    assert cfg.audio_gen_channels % 32 == 0
    if cfg.use_moe:
        assert cfg.n_experts_per_tok <= cfg.n_experts
        assert cfg.first_dense_layers < cfg.n_layers


def test_validate_rejects_broken_shapes():
    with pytest.raises(ValueError, match="divisible by n_heads"):
        FramerConfig(d_model=100, n_heads=8).validate()
    with pytest.raises(ValueError, match="multiple of 64"):
        FramerConfig(diffusion_channels=100).validate()
    with pytest.raises(ValueError, match="exceeds"):
        FramerConfig(use_moe=True, n_experts=4, n_experts_per_tok=8).validate()


def test_mid_size_presets_use_latent_dit():
    """Issue #206: framer-3b, framer-8b, framer-30b-a3b migrated from 1000-step pixel U-Net."""
    for name in ["framer-3b", "framer-8b", "framer-30b-a3b"]:
        cfg = FramerConfig.from_preset(name)
        assert cfg.image_gen_arch == "latent_dit", \
            f"{name} must use latent_dit, not {cfg.image_gen_arch}"
        assert cfg.sampler_steps == 50, \
            f"{name} must use 50 sampler steps, got {cfg.sampler_steps}"
        assert cfg.cfg_scale == 3.0, \
            f"{name} must use cfg_scale=3.0, got {cfg.cfg_scale}"
        # Verify latent diffusion config is present
        assert cfg.vae_latent_channels > 0
        assert cfg.vae_base_channels > 0
        assert cfg.vae_downsample > 0
        assert cfg.dit_d_model > 0
        assert cfg.dit_n_layers > 0
        assert cfg.dit_n_heads > 0


def test_mid_size_presets_build_latent_diffusion_modules():
    """Verify the migrated presets would instantiate latent diffusion, not pixel U-Net."""
    # Use framer-3b only for instantiation test (smallest of the three)
    # Other two are verified via config in test_mid_size_presets_use_latent_dit
    import torch
    cfg = FramerConfig.from_preset("framer-3b")
    # Verify config will build latent diffusion
    assert cfg.image_gen_arch == "latent_dit"

    # Build only the diffusion module to avoid full model overhead
    from model.modules.latent_diffusion import LatentImageGenerator
    diffusion = LatentImageGenerator(cfg)

    # Must have VAE and denoiser, not pixel U-Net
    assert hasattr(diffusion, "vae"), "framer-3b missing VAE"
    assert hasattr(diffusion, "denoiser"), "framer-3b missing denoiser"
    assert not hasattr(diffusion, "unet"), "framer-3b should not have pixel U-Net"

    # Verify VAE encode/decode work
    dummy_image = torch.randn(1, 3, cfg.image_train_resolution, cfg.image_train_resolution)
    latent = diffusion.vae.encode_to_latent(dummy_image)
    assert latent.shape[1] == cfg.vae_latent_channels
    assert latent.shape[2] == cfg.image_train_resolution // cfg.vae_downsample

    # Verify denoiser accepts latents
    dummy_t = torch.zeros(1)
    dummy_context = torch.randn(1, 1, cfg.d_model)
    velocity = diffusion.denoiser(latent, dummy_t, dummy_context)
    assert velocity.shape == latent.shape
