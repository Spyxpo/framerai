"""Long-context tests: RoPE extension validation, KV budget, chunked prefill.

A declared context is easy to fake. `max_seq_len` is just a number in the config,
nothing allocates per-position parameters for it, and before these checks a
preset could claim a million tokens while RoPE covered a few thousand. These
tests pin the three things that make the claim real: validation rejects an
extension no scaling strategy covers, the estimator reports what the window
costs in KV cache, and prefill does not have to run the prompt in one forward.
"""

import pytest
import torch

from conftest import tiny_config
from model.configs.presets import build_preset_config
from model.framer import FramerModel
from model.generate import DEFAULT_PREFILL_CHUNK, FramerGenerator
from model.utils.helpers import estimate_params

ONE_MILLION = 1048576


def long_config(**overrides):
    base = dict(
        max_seq_len=4096,
        rope_original_max_seq_len=512,
        rope_scaling_type="yarn",
        rope_scaling_factor=8.0,
    )
    base.update(overrides)
    return tiny_config(**base)


def test_context_extension_reports_the_ratio():
    assert long_config().context_extension == 8.0
    assert tiny_config().context_extension == 1.0


def test_unscaled_extension_is_rejected():
    """'none' past the original length applies no extension at all."""
    with pytest.raises(ValueError, match="rope_scaling_type is 'none'"):
        long_config(rope_scaling_type="none").validate()


def test_insufficient_scaling_factor_is_rejected():
    """A 4x factor under an 8x window leaves half the context unreachable."""
    with pytest.raises(ValueError, match="below the 8x extension"):
        long_config(rope_scaling_factor=4.0).validate()


def test_original_longer_than_max_is_rejected():
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        long_config(rope_original_max_seq_len=8192).validate()


def test_a_covered_extension_validates():
    assert long_config().validate() is not None


@pytest.mark.parametrize("preset", ["framer-1t-a32b", "framer-2t-a49b"])
def test_flagship_presets_carry_a_million_token_context(preset):
    config = build_preset_config(preset)
    assert config.max_seq_len == ONE_MILLION
    assert config.rope_scaling_type == "yarn"
    # validate() already ran inside build_preset_config, so the factor covers it.
    assert config.rope_scaling_factor >= config.context_extension


def test_context_costs_kv_cache_not_parameters():
    """The window is free in weights and linear in cache. Both matter."""
    short = estimate_params(tiny_config(max_seq_len=1024))
    long = estimate_params(
        tiny_config(
            max_seq_len=8192, rope_original_max_seq_len=1024,
            rope_scaling_type="yarn", rope_scaling_factor=8.0,
        )
    )
    assert long["model_total"] == short["model_total"]
    assert long["kv_cache_bytes"] == 8 * short["kv_cache_bytes"]


def test_kv_cache_matches_the_hand_calculation():
    config = tiny_config(max_seq_len=1024)
    expected = 4 * config.n_layers * config.kv_heads * config.head_dim * config.max_seq_len
    assert estimate_params(config)["kv_cache_bytes"] == expected


def generator(config):
    torch.manual_seed(0)
    return FramerGenerator(FramerModel(config), tokenizer=None, device="cpu")


def test_chunked_prefill_matches_a_single_forward():
    """Chunking is a memory strategy, not a different computation."""
    config = tiny_config(max_seq_len=64)
    gen = generator(config)
    ids = torch.randint(0, config.vocab_size, (1, 21))

    with torch.no_grad():
        whole_past, whole_logits = gen._prefill(ids, chunk_size=64)
        chunked_past, chunked_logits = gen._prefill(ids, chunk_size=4)

    assert torch.allclose(whole_logits, chunked_logits, atol=1e-5)
    for (wk, wv), (ck, cv) in zip(whole_past, chunked_past, strict=True):
        assert wk.shape == ck.shape
        assert torch.allclose(wk, ck, atol=1e-5)
        assert torch.allclose(wv, cv, atol=1e-5)


def test_prefill_cache_covers_every_prompt_position():
    config = tiny_config(max_seq_len=64)
    gen = generator(config)
    ids = torch.randint(0, config.vocab_size, (1, 17))

    with torch.no_grad():
        past, _ = gen._prefill(ids, chunk_size=5)

    assert len(past) == config.n_layers
    assert past[0][0].shape[2] == ids.shape[1]


def test_prefill_chunk_default_is_bounded():
    """A default of None must not degrade into a single-shot prefill."""
    assert 0 < DEFAULT_PREFILL_CHUNK <= 8192
