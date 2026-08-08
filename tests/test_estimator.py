"""Whole-model parameter estimation: multimodal towers, accuracy, and cost.

The estimator is what makes a trillion-parameter definition checkable on a
laptop, so these tests pin both its accuracy (against really-instantiated tiny
models) and its promise to allocate nothing at scale.
"""

import pytest

from conftest import tiny_config
from model.configs import PRESETS, FramerConfig
from model.framer import FramerModel
from model.utils.helpers import estimate_multimodal_params, estimate_params


def tiny_multimodal_config(**overrides):
    """A full multimodal config small enough to really instantiate on CPU."""
    base = dict(
        text_only=False,
        image_size=32,
        patch_size=8,
        vision_d_model=32,
        vision_n_heads=4,
        vision_n_layers=2,
        audio_d_model=32,
        audio_n_heads=4,
        audio_n_layers=2,
        audio_n_mels=32,
        audio_max_frames=64,
        diffusion_channels=64,
        diffusion_steps=10,
        audio_gen_channels=32,
        audio_gen_frames=32,
        video_frames=2,
        video_resolution=32,
    )
    base.update(overrides)
    return tiny_config(**base)


def test_meta_count_matches_instantiated_multimodal_model():
    cfg = tiny_multimodal_config()
    actual = sum(p.numel() for p in FramerModel(cfg).parameters())
    est = estimate_params(cfg)["model_total"]
    assert abs(actual - est) / actual < 0.02


def test_qk_norm_params_are_counted():
    cfg = tiny_config(use_qk_norm=True)
    actual = sum(p.numel() for p in FramerModel(cfg).parameters())
    est = estimate_params(cfg)["total"]
    assert abs(actual - est) / actual < 0.02


def test_text_only_has_no_multimodal_params():
    est = estimate_params(tiny_config(text_only=True))
    assert est["multimodal_total"] == 0
    assert est["model_total"] == est["total"]
    assert estimate_multimodal_params(FramerConfig(text_only=True))["total"] == 0


def test_multimodal_breakdown_covers_every_tower():
    counts = estimate_multimodal_params(tiny_multimodal_config())
    towers = ("vision_encoder", "vision_projector", "audio_encoder", "audio_projector",
              "image_diffusion", "video_diffusion", "audio_diffusion")
    assert set(counts) == set(towers) | {"total"}
    assert all(counts[name] > 0 for name in towers)
    assert counts["total"] == sum(counts[name] for name in towers)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_is_estimable_without_instantiation(name):
    est = estimate_params(FramerConfig.from_preset(name))
    assert est["model_total"] > 0
    assert est["active"] <= est["total"]
    assert est["bf16_bytes"] == 2 * est["model_total"]
    assert est["adamw_bytes"] == 16 * est["model_total"]
