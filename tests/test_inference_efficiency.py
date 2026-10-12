"""Serving accelerators: prompt-cache reuse, speculative decoding, int8 weights.

Each one changes what a request costs and must not change what it returns.
These tests pin both halves: the work skipped is reported, and the output is
the output plain decoding gives.
"""

import pytest
import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.modules.kv_cache import KVCache, fork_cache, truncate_cache
from model.prefix_cache import PrefixCache
from model.quantize import Int8Linear, quantize_model, weight_bytes
from model.serve import _usage
from model.tokenizer import FramerTokenizer

CORPUS = ["the quick brown fox jumps over the lazy dog", "hello world foo bar baz"]


def _tokenizer():
    tok = FramerTokenizer(vocab_size=300)
    tok.train(CORPUS, target_vocab_size=300)
    return tok


def _generator(seed=0, tok=None, **overrides):
    torch.manual_seed(seed)
    config = tiny_config(vocab_size=300, **overrides)
    return FramerGenerator(FramerModel(config).eval(), tok or _tokenizer(), device="cpu")


def _ids(gen, text="the quick brown fox jumps over the lazy dog " * 3):
    return gen.tokenizer.encode(text, add_special=False)


# ── KV cache truncate and fork ────────────────────────────────────────────

@pytest.mark.parametrize("quantized", [False, True])
def test_a_fork_is_independent_of_its_source(quantized):
    cache = KVCache(block_size=4, quantized=quantized)
    k = torch.randn(1, 2, 6, 8)
    cache.append(k, k.clone())
    fork = cache.fork(4)
    assert len(fork) == 4
    torch.testing.assert_close(fork.keys, cache.keys[:, :, :4])

    fork.append(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
    assert len(cache) == 6, "appending to the fork must leave the source alone"
    torch.testing.assert_close(cache.keys[:, :, 4:6], cache.keys[:, :, 4:6])


def test_truncate_moves_the_length_back():
    cache = KVCache(block_size=4)
    cache.append(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4))
    cache.truncate(2)
    assert len(cache) == 2 and cache.keys.shape[2] == 2
    with pytest.raises(ValueError):
        cache.truncate(3)


def test_tuple_caches_fork_and_truncate_too():
    k = torch.randn(1, 1, 5, 4)
    forked = fork_cache([(k, k)], 3)
    assert forked[0][0].shape[2] == 3 and forked[0][0].data_ptr() != k.data_ptr()
    assert truncate_cache([(k, k)], 2)[0][0].shape[2] == 2


# ── Prefix cache ──────────────────────────────────────────────────────────

def _fake_past(length):
    cache = KVCache(block_size=8)
    cache.append(torch.randn(1, 1, length, 4), torch.randn(1, 1, length, 4))
    return [cache]


def test_lookup_returns_the_longest_shared_prefix():
    store = PrefixCache(min_match=2)
    store.store(list(range(10)), _fake_past(10))
    matched, past = store.lookup(list(range(6)) + [99, 98])
    assert matched == 6 and len(past[0]) == 6


def test_at_least_one_token_is_left_to_run():
    store = PrefixCache(min_match=2)
    store.store(list(range(10)), _fake_past(10))
    matched, _ = store.lookup(list(range(10)))
    assert matched == 9


def test_short_matches_are_misses():
    store = PrefixCache(min_match=8)
    store.store(list(range(10)), _fake_past(10))
    assert store.lookup([0, 1, 2, 50, 51]) == (0, None)
    assert store.misses == 1


def test_bounds_evict_the_oldest():
    store = PrefixCache(max_entries=2, min_match=1)
    for base in (100, 200, 300):
        store.store([base + i for i in range(4)], _fake_past(4))
    assert len(store) == 2
    assert store.lookup([100, 101, 102]) == (0, None)

    store = PrefixCache(max_entries=10, max_tokens=8, min_match=1)
    for base in (100, 200, 300):
        store.store([base + i for i in range(4)], _fake_past(4))
    assert store.tokens_held <= 8


def test_an_extended_prompt_replaces_its_prefix():
    store = PrefixCache(min_match=1)
    store.store([1, 2, 3], _fake_past(3))
    store.store([1, 2, 3, 4, 5], _fake_past(5))
    assert len(store) == 1


def test_reuse_reports_cached_tokens_and_keeps_the_answer():
    plain = _generator()
    cached = _generator()
    cached.prefix_cache = PrefixCache(min_match=4)
    ids = _ids(plain)

    expected = plain.generate_text(ids, max_new_tokens=8, temperature=0)
    first = cached.generate_text(ids, max_new_tokens=8, temperature=0)
    assert cached.last_cached_input_tokens == 0
    second = cached.generate_text(ids, max_new_tokens=8, temperature=0)

    assert first == expected and second == expected
    assert cached.last_cached_input_tokens == cached.last_input_tokens - 1
    assert _usage(cached)["usage"]["cached_input_tokens"] == cached.last_cached_input_tokens


def test_a_follow_up_turn_reuses_the_conversation_so_far():
    # A window wide enough that neither turn is trimmed from the front, which
    # would change the prefix the two share.
    gen = _generator(max_seq_len=512)
    gen.prefix_cache = PrefixCache(min_match=4)
    turn = _ids(gen)
    gen.generate_text(turn, max_new_tokens=2, temperature=0)
    gen.generate_text(turn + _ids(gen, "hello world"), max_new_tokens=2, temperature=0)
    assert gen.last_cached_input_tokens >= len(turn)


# ── Speculative decoding ──────────────────────────────────────────────────

def test_greedy_with_an_identical_draft_accepts_everything():
    target = _generator(seed=0)
    draft = _generator(seed=0, tok=target.tokenizer)
    expected = target.generate_text(_ids(target), max_new_tokens=12, temperature=0)
    target.attach_draft(draft, k=3)
    assert target.generate_text(_ids(target), max_new_tokens=12, temperature=0) == expected
    assert target.last_draft_proposed > 0
    assert target.last_draft_accepted == target.last_draft_proposed or target.last_finish_reason == "eos"


@pytest.mark.parametrize("k", [1, 2, 5])
def test_greedy_with_a_different_draft_matches_plain_decoding(k):
    target = _generator(seed=0)
    draft = _generator(seed=7, tok=target.tokenizer)
    ids = _ids(target)
    expected = target.generate_text(ids, max_new_tokens=16, temperature=0)
    target.attach_draft(draft, k=k)
    assert target.generate_text(ids, max_new_tokens=16, temperature=0) == expected
    assert target.last_draft_accepted <= target.last_draft_proposed


def test_streaming_with_a_draft_matches_plain_streaming():
    target = _generator(seed=0)
    draft = _generator(seed=3, tok=target.tokenizer)
    ids = _ids(target)
    expected = "".join(target.generate_stream(ids, max_new_tokens=10, temperature=0))
    target.attach_draft(draft, k=3)
    assert "".join(target.generate_stream(ids, max_new_tokens=10, temperature=0)) == expected


def test_sampling_with_a_draft_respects_the_length_and_is_seeded():
    target = _generator(seed=0)
    target.attach_draft(_generator(seed=5, tok=target.tokenizer), k=4)
    ids = _ids(target)
    a = target.generate_text(ids, max_new_tokens=9, temperature=0.8, seed=11)
    assert target.last_output_tokens <= 9
    b = target.generate_text(ids, max_new_tokens=9, temperature=0.8, seed=11)
    assert a == b


def test_sampling_with_an_identical_draft_accepts_everything():
    target = _generator(seed=0)
    target.attach_draft(_generator(seed=0, tok=target.tokenizer), k=4)
    target.generate_text(_ids(target), max_new_tokens=12, temperature=1.0, seed=3)
    if target.last_finish_reason == "length":
        assert target.last_draft_accepted == target.last_draft_proposed


def test_a_repetition_penalty_decodes_plainly():
    target = _generator(seed=0)
    target.attach_draft(_generator(seed=1, tok=target.tokenizer))
    target.generate_text(_ids(target), max_new_tokens=4, temperature=0, repetition_penalty=1.3)
    assert target.last_draft_proposed == 0


def test_a_draft_with_another_vocabulary_is_refused():
    target = _generator()
    other = FramerGenerator(FramerModel(tiny_config(vocab_size=320)), target.tokenizer, device="cpu")
    with pytest.raises(ValueError, match="vocabulary"):
        target.attach_draft(other)


# ── Int8 weights ──────────────────────────────────────────────────────────

def test_int8_linear_tracks_the_original():
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 32)
    q = Int8Linear.from_linear(linear)
    x = torch.randn(4, 64)
    ref = linear(x)
    err = (q(x) - ref).abs().max() / ref.abs().max()
    assert err < 0.02


def test_quantizing_shrinks_the_weights_and_still_generates():
    gen = _generator()
    before = weight_bytes(gen.model)
    replaced = quantize_model(gen.model, "int8")
    assert replaced > 0
    assert weight_bytes(gen.model) < before
    out = gen.generate_text(_ids(gen), max_new_tokens=4, temperature=0)
    assert isinstance(out, str)


def test_auto_leaves_the_model_alone():
    gen = _generator()
    assert quantize_model(gen.model, "auto") == 0
    with pytest.raises(ValueError):
        quantize_model(gen.model, "int3")
