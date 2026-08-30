"""RoPE context-extension tests.

`rope_scaling_type="yarn"` matched neither the ntk branch nor the linear branch,
so it silently applied no extension at all - a config asking for 4x context got
none. These tests pin each strategy to distinct numerical behaviour so a future
refactor cannot quietly turn one back into a no-op.
"""

import pytest
import torch

from conftest import random_ids, tiny_config
from model.configs.model_config import ROPE_SCALING_TYPES
from model.framer import FramerModel
from model.modules.transformer import RotaryPositionalEmbedding

HEAD_DIM = 64
FACTOR = 4.0
ORIGINAL = 512


def rope(scaling_type, factor=FACTOR):
    return RotaryPositionalEmbedding(
        HEAD_DIM, max_seq_len=2048, scaling_factor=factor,
        scaling_type=scaling_type, original_max_seq_len=ORIGINAL,
    )


@pytest.mark.parametrize("scaling_type", ["linear", "ntk", "yarn"])
def test_scaling_changes_the_embedding(scaling_type):
    cos_none, _ = rope("none")(64)
    cos_scaled, _ = rope(scaling_type)(64)
    assert not torch.allclose(cos_none, cos_scaled), f"{scaling_type} behaved like 'none'"


def test_yarn_is_distinct_from_every_other_strategy():
    """Regression: yarn used to fall through to no scaling whatsoever."""
    cos_yarn, sin_yarn = rope("yarn")(64)
    for other in ("none", "linear", "ntk"):
        cos_other, _ = rope(other)(64)
        assert not torch.allclose(cos_yarn, cos_other), f"yarn matches '{other}'"
    assert torch.isfinite(cos_yarn).all() and torch.isfinite(sin_yarn).all()


def test_yarn_preserves_high_frequencies_and_interpolates_low_ones():
    """NTK-by-parts: local-order dimensions untouched, long-range fully scaled."""
    base = rope("none").inv_freq
    yarn = rope("yarn").inv_freq
    linear = base / FACTOR

    # Highest frequency (shortest wavelength) is left alone.
    assert torch.allclose(yarn[0], base[0])
    # Lowest frequency (never completes a rotation) is interpolated in full.
    assert torch.allclose(yarn[-1], linear[-1])
    # Nothing is scaled harder than plain interpolation or softer than no-op.
    assert (yarn <= base + 1e-8).all() and (yarn >= linear - 1e-8).all()


def test_yarn_applies_the_attention_factor():
    scaled = rope("yarn")
    assert scaled.attention_factor > 1.0
    # cos/sin carry the compensation for attention-entropy drift.
    cos, _ = scaled(8)
    assert cos.abs().max() > 1.0


def test_scaling_factor_one_is_a_no_op():
    for scaling_type in ROPE_SCALING_TYPES:
        cos, _ = rope(scaling_type, factor=1.0)(16)
        cos_none, _ = rope("none", factor=1.0)(16)
        assert torch.allclose(cos, cos_none), f"{scaling_type} scaled at factor 1.0"


def test_unknown_scaling_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown rope scaling_type"):
        RotaryPositionalEmbedding(HEAD_DIM, scaling_type="yaarn")


def test_config_rejects_unknown_scaling_type():
    with pytest.raises(ValueError, match="rope_scaling_type"):
        tiny_config(rope_scaling_type="yaarn").validate()


def test_config_rejects_inverted_frequency_band():
    with pytest.raises(ValueError, match="rope_high_freq_factor"):
        tiny_config(
            rope_scaling_type="yarn", rope_low_freq_factor=8.0, rope_high_freq_factor=2.0
        ).validate()


def test_positions_are_absolute_under_scaling():
    """Cached decode must rotate a token the same way prefill did."""
    scaled = rope("yarn")
    cos_full, sin_full = scaled(16)
    cos_step, sin_step = scaled(1, offset=15)
    assert torch.allclose(cos_full[15], cos_step[0])
    assert torch.allclose(sin_full[15], sin_step[0])


@pytest.mark.parametrize("scaling_type", ["linear", "ntk", "yarn"])
def test_model_runs_beyond_the_trained_length(scaling_type):
    # The window the config claims is the extension; 64 is what it trained at.
    config = tiny_config(
        max_seq_len=256, rope_scaling_factor=4.0, rope_scaling_type=scaling_type,
        rope_original_max_seq_len=64,
    ).validate()
    model = FramerModel(config).eval()
    with torch.no_grad():
        out = model(input_ids=random_ids(config, batch=1, length=96))
    assert out["logits"].shape == (1, 96, config.vocab_size)
    assert torch.isfinite(out["logits"]).all()
