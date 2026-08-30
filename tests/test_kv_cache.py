"""Block-paged KV cache, and the estimate that has to match it.

The cache was two tensors concatenated on every decoded token, so each step
reallocated the whole history: memory traffic quadratic in the sequence for a
cache that grows by one. It was also stored densely in the activation dtype,
and on the trillion-scale presets the cache, not the weights, is what decides
whether a declared window is servable.

These tests pin that paging changes nothing about the values, that quantisation
costs precision and not correctness, and that `--estimate` reports the cache
the config actually asks for rather than a bf16 one it does not.
"""

import pytest
import torch

from conftest import tiny_config
from model.configs import FramerConfig
from model.configs.presets import build_preset_config
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.modules.kv_cache import KVCache, cache_bytes_per_token
from model.tokenizer import FramerTokenizer
from model.utils.helpers import estimate_params


def _kv(batch=1, heads=2, length=3, dim=8):
    return torch.randn(batch, heads, length, dim), torch.randn(batch, heads, length, dim)


# ── Paging ────────────────────────────────────────────────────────────────

def test_what_goes_in_comes_back_out():
    cache = KVCache(block_size=4)
    keys, values = _kv(length=3)
    out_k, out_v = cache.append(keys, values)

    assert torch.equal(out_k, keys) and torch.equal(out_v, values)
    assert len(cache) == 3


def test_appending_one_token_at_a_time_matches_appending_them_together():
    keys, values = _kv(length=6)

    whole = KVCache(block_size=4)
    whole.append(keys, values)

    stepwise = KVCache(block_size=4)
    for i in range(6):
        stepwise.append(keys[:, :, i:i + 1], values[:, :, i:i + 1])

    assert torch.equal(whole.keys, stepwise.keys)
    assert torch.equal(whole.values, stepwise.values)


def test_the_buffer_grows_in_blocks_rather_than_per_token():
    cache = KVCache(block_size=8)
    keys, values = _kv(length=1)
    cache.append(keys, values)
    assert cache.allocated == 8, "one token should not cost one allocation"

    for _ in range(7):
        cache.append(*_kv(length=1))
    assert cache.allocated == 8, "the block still had room"

    cache.append(*_kv(length=1))
    assert cache.allocated == 16


def test_it_still_reads_as_the_tuple_it_replaced():
    cache = KVCache(block_size=4)
    keys, values = _kv(length=2)
    cache.append(keys, values)

    assert torch.equal(cache[0], keys) and torch.equal(cache[1], values)
    unpacked_k, unpacked_v = cache
    assert torch.equal(unpacked_k, keys)
    with pytest.raises(IndexError):
        cache[2]


def test_a_mismatched_append_is_refused():
    cache = KVCache()
    with pytest.raises(ValueError, match="must match"):
        cache.append(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 4, 8))


def test_a_block_size_of_nothing_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        KVCache(block_size=0)


# ── Quantisation ──────────────────────────────────────────────────────────

def test_quantised_storage_is_close_but_not_exact():
    keys, values = _kv(length=5)
    cache = KVCache(block_size=4, quantized=True)
    cache.append(keys, values)

    assert cache.keys.dtype == keys.dtype, "the compute dtype is unchanged"
    assert not torch.equal(cache.keys, keys), "int8 storage is lossy by construction"
    assert torch.allclose(cache.keys, keys, atol=0.05)


def test_quantisation_holds_up_across_scales():
    # The scale is per token and head, so a row a thousand times larger than
    # its neighbour does not drag the neighbour into its range. Error is judged
    # against each row's own magnitude: absmax quantisation is inherently
    # coarse on the elements of a row that sit near zero, and measuring those
    # per element would report a failure of arithmetic rather than of storage.
    keys = torch.randn(1, 2, 4, 8)
    keys[:, :, 0] *= 1000.0
    cache = KVCache(quantized=True)
    cache.append(keys, keys)

    row_scale = keys.abs().amax(dim=-1, keepdim=True)
    relative = (cache.keys - keys).abs() / row_scale
    assert relative.max() < 0.01

    # The large row is no better and no worse served than the small ones.
    per_row = relative.amax(dim=-1)
    assert per_row.max() / per_row.min().clamp(min=1e-9) < 10.0


def test_quantised_storage_costs_less_than_it_saves_in_scales():
    plain = KVCache(block_size=16)
    quantised = KVCache(block_size=16, quantized=True)
    keys, values = _kv(length=16, dim=64)
    plain.append(keys, values)
    quantised.append(keys, values)

    assert quantised.bytes_used() < plain.bytes_used()


def test_the_per_token_cost_counts_the_scales():
    plain = cache_bytes_per_token(64, 8, 128)
    quantised = cache_bytes_per_token(64, 8, 128, quantized=True)
    assert quantised < plain
    assert quantised > plain / 2, "the scales are not free, and are not ignored"


# ── Through the model ─────────────────────────────────────────────────────

def _generate(config, seed=0):
    tokenizer = FramerTokenizer(vocab_size=config.vocab_size)
    tokenizer.train(["hello world foo bar"], target_vocab_size=config.vocab_size)
    torch.manual_seed(seed)
    model = FramerModel(config).eval()
    gen = FramerGenerator(model, tokenizer, device="cpu")
    torch.manual_seed(seed)
    return gen.generate_text("hello", max_new_tokens=6, temperature=0.0001)


def test_paging_does_not_change_what_the_model_produces():
    paged = tiny_config(vocab_size=300, max_seq_len=128, kv_cache_paged=True)
    unpaged = tiny_config(vocab_size=300, max_seq_len=128, kv_cache_paged=False)
    assert _generate(paged) == _generate(unpaged)


def test_a_quantised_cache_still_produces_an_answer():
    config = tiny_config(vocab_size=300, max_seq_len=128, kv_cache_dtype="int8")
    assert isinstance(_generate(config), str)


def test_the_cache_that_comes_back_is_the_paged_one():
    config = tiny_config(vocab_size=300, max_seq_len=64)
    model = FramerModel(config).eval()
    with torch.no_grad():
        out = model.forward_lm(torch.tensor([[1, 2, 3]]), use_cache=True)
    assert isinstance(out["past_kvs"][0], KVCache)
    assert len(out["past_kvs"][0]) == 3


# ── Configuration and the estimate ────────────────────────────────────────

def test_an_unrecognised_cache_dtype_is_rejected_up_front():
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        FramerConfig(kv_cache_dtype="fp4").validate()
    with pytest.raises(ValueError, match="kv_cache_block_size"):
        FramerConfig(kv_cache_block_size=0).validate()


def test_the_estimate_follows_the_configured_cache():
    config = build_preset_config("framer-1t-a32b")
    bf16 = estimate_params(config)["kv_cache_bytes"]

    config.kv_cache_dtype = "int8"
    quantised = estimate_params(config)["kv_cache_bytes"]

    assert quantised < bf16
    assert quantised > bf16 / 2, "the scales are counted, not wished away"
    assert estimate_params(config)["kv_cache_dtype"] == "int8"


def test_the_million_token_window_is_the_number_that_matters():
    config = build_preset_config("framer-1t-a32b")
    estimate = estimate_params(config)
    # Hundreds of gigabytes for one sequence: more than the weights, which is
    # the whole reason this work exists.
    assert estimate["kv_cache_bytes"] / 2**30 > 100
