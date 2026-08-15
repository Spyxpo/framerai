"""Preset registry + parameter estimator tests (incl. the 2T flagship)."""

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
]


def test_all_presets_build_configs():
    for name in list_presets():
        cfg = FramerConfig.from_preset(name)
        assert cfg.d_model > 0 and cfg.n_layers > 0
        assert cfg.preset == name


def test_size_aliases():
    assert resolve_preset_name("tiny") == "framer-tiny"
    assert resolve_preset_name("large") == "framer-large"
    assert FramerConfig.from_preset("small").preset == "framer-small"


def test_tiny_preset_image_train_resolution():
    assert FramerConfig.from_preset("framer-tiny").image_train_resolution == 64
    assert FramerConfig.from_preset("framer-tiny-moe").image_train_resolution == 64


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
