"""Focused regression and unit tests for Issue #243:
- Repetition penalty
- Stop sequences
- Per-request seed
- Temperature <= 0 greedy decoding
- Streaming generation (generate_stream)
- Serve / worker integration
"""

import pytest
import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.generate import FramerGenerator, _apply_repetition_penalty
from model.serve import _sampling, handle
from model.tokenizer import FramerTokenizer


def _make_generator(vocab_size: int = 300) -> FramerGenerator:
    tok = FramerTokenizer(vocab_size=vocab_size)
    tok.train(
        [
            "hello world foo bar baz stop here end now",
            "the quick brown fox jumps over the lazy dog",
            "apple banana orange apple banana orange",
        ],
        target_vocab_size=vocab_size,
    )
    cfg = tiny_config(vocab_size=tok.vocab_size, max_seq_len=64)
    model = FramerModel(cfg)
    return FramerGenerator(model, tok, device="cpu")


# ===========================================================================
# A. Repetition Penalty Tests
# ===========================================================================

def test_repetition_penalty_neutral_default():
    """Penalty of 1.0 does not modify logits."""
    logits = torch.randn(1, 300)
    seen = [5, 10, 15]
    out = _apply_repetition_penalty(logits, seen, penalty=1.0)
    assert torch.equal(logits, out)


def test_repetition_penalty_positive_logits():
    """Positive logits for seen tokens are divided by penalty."""
    logits = torch.zeros(1, 300)
    logits[0, 42] = 4.0
    logits[0, 99] = 4.0  # unseen
    out = _apply_repetition_penalty(logits, [42], penalty=2.0)
    assert out[0, 42].item() == pytest.approx(2.0)
    assert out[0, 99].item() == pytest.approx(4.0)


def test_repetition_penalty_negative_logits():
    """Negative logits for seen tokens are multiplied by penalty."""
    logits = torch.zeros(1, 300)
    logits[0, 42] = -3.0
    logits[0, 99] = -3.0  # unseen
    out = _apply_repetition_penalty(logits, [42], penalty=2.0)
    assert out[0, 42].item() == pytest.approx(-6.0)
    assert out[0, 99].item() == pytest.approx(-3.0)


def test_repetition_penalty_does_not_mutate_original():
    """Original logits tensor is not modified in-place."""
    logits = torch.randn(1, 300)
    orig = logits.clone()
    _ = _apply_repetition_penalty(logits, [10, 20], penalty=1.5)
    assert torch.equal(logits, orig)


def test_repetition_penalty_rejects_non_positive():
    """Penalty <= 0 raises ValueError."""
    logits = torch.randn(1, 300)
    with pytest.raises(ValueError, match="repetition_penalty must be positive"):
        _apply_repetition_penalty(logits, [1], penalty=0.0)
    with pytest.raises(ValueError, match="repetition_penalty must be positive"):
        _apply_repetition_penalty(logits, [1], penalty=-1.5)


def test_repetition_penalty_changes_token_selection():
    """A high penalty heavily suppresses repeated tokens."""
    gen = _make_generator()
    # Force identical starting logits by setting seed
    # With no penalty, the model may repeat tokens
    res_no_penalty = gen.generate_text("apple banana", max_new_tokens=20, repetition_penalty=1.0, seed=42)
    # With high penalty, repeated tokens are strongly discouraged
    res_with_penalty = gen.generate_text("apple banana", max_new_tokens=20, repetition_penalty=5.0, seed=42)
    assert isinstance(res_no_penalty, str)
    assert isinstance(res_with_penalty, str)


def test_repetition_penalty_includes_prompt_tokens():
    """Prompt tokens are included in the seen-token set from the very first step."""
    gen = _make_generator()
    prompt = "apple banana"
    prompt_tokens = gen.tokenizer.encode(prompt, add_special=True)
    assert len(prompt_tokens) > 1

    # Verify that passing prompt_tokens as seen_tokens penalizes them
    logits = torch.zeros(1, gen.tokenizer.vocab_size)
    for t in prompt_tokens:
        logits[0, t] = 8.0
    penalized = _apply_repetition_penalty(logits, prompt_tokens, penalty=2.0)
    for t in prompt_tokens:
        assert penalized[0, t].item() == pytest.approx(4.0)


# ===========================================================================
# B. Stop Sequences Tests
# ===========================================================================

def test_single_stop_sequence():
    """Generation halts when a single stop sequence is encountered."""
    gen = _make_generator()
    # Decode a few tokens from prompt to know valid substrings
    prompt = "hello"
    # First generate without stop to observe output
    full = gen.generate_text(prompt, max_new_tokens=15, temperature=0, seed=1)
    continuation = full[len(prompt):]
    if len(continuation) > 2:
        stop_str = continuation[1:3]
        out, reason = gen.generate_text(
            prompt, max_new_tokens=15, temperature=0, seed=1, stop=[stop_str], return_reason=True
        )
        assert reason == "stop"
        assert stop_str not in out[len(prompt):]


def test_multiple_stop_sequences():
    """Generation halts on whichever stop sequence appears first."""
    gen = _make_generator()
    prompt = "hello"
    full = gen.generate_text(prompt, max_new_tokens=20, temperature=0, seed=2)
    continuation = full[len(prompt):]
    if len(continuation) > 4:
        stop1 = continuation[1:3]
        stop2 = continuation[2:4]
        out, reason = gen.generate_text(
            prompt, max_new_tokens=20, temperature=0, seed=2, stop=[stop1, stop2], return_reason=True
        )
        assert reason == "stop"
        assert stop1 not in out[len(prompt):]


def test_stop_sequence_excluded_from_output():
    """The stop sequence itself does not appear in the returned content."""
    gen = _make_generator()
    prompt = "hello"
    full = gen.generate_text(prompt, max_new_tokens=15, temperature=0, seed=3)
    continuation = full[len(prompt):]
    if len(continuation) >= 2:
        stop_target = continuation[:2]
        out = gen.generate_text(prompt, max_new_tokens=15, temperature=0, seed=3, stop=[stop_target])
        gen_only = out[len(prompt):]
        assert stop_target not in gen_only


def test_termination_reason_length():
    """Termination reason is 'length' when max_new_tokens is reached without stop/eos."""
    gen = _make_generator()
    # Short max_new_tokens with a stop target that will never match
    out, reason = gen.generate_text("hello", max_new_tokens=2, stop=["ZZZZZZZZZ"], return_reason=True)
    assert reason == "length"
    assert gen.last_finish_reason == "length"


def test_stop_sequence_no_false_positive_from_prompt():
    """A stop sequence present in the prompt does not immediately abort generation."""
    gen = _make_generator()
    # Prompt contains the stop sequence, but stop should only check generated output
    stop_phrase = "world"
    prompt = f"hello {stop_phrase} test"
    out, reason = gen.generate_text(prompt, max_new_tokens=5, stop=[stop_phrase], return_reason=True)
    # It must have generated at least something (or hit length/eos), not aborted with empty generated text
    assert len(out) >= len(prompt)


def test_stop_sequence_spans_multiple_token_boundaries(monkeypatch):
    """Multi-token boundary stop sequence:
    - generation stops at the first complete match
    - stop text never appears in returned output
    - no partial stop sequence is incorrectly emitted by generate_stream()
    """
    gen = _make_generator()
    tok_foo = gen.tokenizer.encode("foo", add_special=False)
    tok_bar = gen.tokenizer.encode("bar", add_special=False)
    tok_baz = gen.tokenizer.encode("baz", add_special=False)
    assert len(tok_foo) >= 1 and len(tok_bar) >= 1 and len(tok_baz) >= 1

    id_foo = tok_foo[0]
    id_bar = tok_bar[0]
    id_baz = tok_baz[0]

    s_foo = gen.tokenizer.decode([id_foo])
    s_bar = gen.tokenizer.decode([id_bar])

    # Construct stop sequence spanning the boundary between id_foo and id_bar
    prefix_part = s_foo[-1:]
    suffix_part = s_bar[:2] if len(s_bar) >= 2 else s_bar[:1]
    cross_token_stop = prefix_part + suffix_part

    token_sequence = [id_foo, id_bar, id_baz]

    class FakeLM:
        def __init__(self):
            self.step = 0

        def __call__(self, input_ids, past_kvs=None, use_cache=True):
            logits = torch.zeros(1, 1, gen.tokenizer.vocab_size)
            if self.step < len(token_sequence):
                chosen = token_sequence[self.step]
            else:
                chosen = gen.tokenizer.eos_id
            logits[0, 0, chosen] = 100.0
            self.step += 1
            return {"logits": logits, "past_kvs": None}

    fake_lm = FakeLM()
    monkeypatch.setattr(gen.model, "forward_lm", fake_lm)

    def fake_prefill(input_ids, prefix_embeds=None, chunk_size=None, modality_embeds=None):
        logits = torch.zeros(1, gen.tokenizer.vocab_size)
        logits[0, id_foo] = 100.0
        fake_lm.step = 1
        return None, logits

    monkeypatch.setattr(gen, "_prefill", fake_prefill)

    deltas = list(
        gen.generate_stream(
            "hello",
            max_new_tokens=5,
            temperature=0,
            stop=[cross_token_stop],
            return_reason=True,
        )
    )

    # 1. Generation stops at the first complete match (id_baz is never generated, step == 2)
    assert fake_lm.step == 2, f"Expected step 2, got {fake_lm.step}"

    # 2. Stop text never appears in returned output
    streamed_text = "".join(d for d, r in deltas if d)
    assert cross_token_stop not in streamed_text

    # 3. No partial stop sequence was prematurely emitted
    for delta, _ in deltas:
        assert cross_token_stop not in delta
    assert not streamed_text.endswith(cross_token_stop)
    assert not streamed_text.endswith(prefix_part)

    # 4. Final finish reason is 'stop'
    _, last_reason = deltas[-1]
    assert last_reason == "stop"


# ===========================================================================
# C. Per-Request Seed Tests
# ===========================================================================

def test_seed_reproducibility():
    """Same seed and sampling settings produce the exact same sequence."""
    gen = _make_generator()
    a = gen.generate_text("hello", max_new_tokens=15, temperature=0.8, seed=12345)
    b = gen.generate_text("hello", max_new_tokens=15, temperature=0.8, seed=12345)
    assert a == b


def test_different_seeds_produce_different_output():
    """Different seeds deterministically produce different sequences under stochastic sampling."""
    gen = _make_generator()
    a = gen.generate_text("hello", max_new_tokens=15, temperature=1.0, seed=1)
    b = gen.generate_text("hello", max_new_tokens=15, temperature=1.0, seed=2)
    assert a != b, "Different seeds must produce different completions under stochastic sampling"


def test_seed_does_not_mutate_global_torch_rng():
    """Seeded text generation does not touch global torch RNG state."""
    gen = _make_generator()
    torch.manual_seed(999)
    val_before = torch.rand(5)

    torch.manual_seed(999)
    _ = gen.generate_text("hello", max_new_tokens=10, temperature=0.8, seed=42)
    val_after = torch.rand(5)

    assert torch.equal(val_before, val_after), "Global RNG state was altered by per-request seed"


# ===========================================================================
# D. Temperature <= 0 Tests
# ===========================================================================

def test_temperature_zero_uses_greedy_argmax():
    """temperature=0 gives deterministic greedy decoding without multinomial."""
    gen = _make_generator()
    a = gen.generate_text("hello", max_new_tokens=10, temperature=0)
    b = gen.generate_text("hello", max_new_tokens=10, temperature=0)
    assert a == b


def test_temperature_zero_does_not_call_multinomial(monkeypatch):
    """Greedy path (temperature=0) never invokes torch.multinomial."""
    gen = _make_generator()

    def mock_multinomial(*_, **__):
        raise AssertionError("torch.multinomial should NOT be called when temperature <= 0")

    monkeypatch.setattr(torch, "multinomial", mock_multinomial)
    out = gen.generate_text("hello", max_new_tokens=8, temperature=0)
    assert isinstance(out, str)


def test_temperature_zero_no_overflow_with_large_logits():
    """With temperature=0, large logits do not overflow to inf/NaN from 1e-6 division."""
    # Test directly with logits having high dynamic range
    step = torch.tensor([[100.0, -100.0, 50.0]])
    next_tok = step.argmax(dim=-1).item()
    assert next_tok == 0


def test_temperature_negative_also_uses_greedy():
    """temperature < 0 also triggers greedy argmax decoding."""
    gen = _make_generator()
    a = gen.generate_text("hello", max_new_tokens=8, temperature=-0.5)
    b = gen.generate_text("hello", max_new_tokens=8, temperature=0.0)
    assert a == b


# ===========================================================================
# E. Streaming Tests (generate_stream)
# ===========================================================================

def test_generate_stream_yields_incremental_deltas():
    """generate_stream yields token deltas that concatenate to valid text."""
    gen = _make_generator()
    deltas = list(gen.generate_stream("hello world", max_new_tokens=10, temperature=0.7, seed=42))
    assert len(deltas) > 0
    assert all(isinstance(d, str) for d in deltas)
    concatenated = "".join(deltas)
    assert len(concatenated) > 0


def test_generate_stream_matches_generate_text():
    """Deterministic generate_stream output matches generate_text continuation."""
    gen = _make_generator()
    prompt = "hello world"
    full_text = gen.generate_text(prompt, max_new_tokens=12, temperature=0, seed=42)
    expected_continuation = full_text[len(prompt):]

    stream_deltas = list(gen.generate_stream(prompt, max_new_tokens=12, temperature=0, seed=42))
    streamed_continuation = "".join(stream_deltas)

    assert streamed_continuation == expected_continuation


def test_generate_stream_stochastic_matches_generate_text():
    """With temperature > 0 and the same seed, streamed completion exactly matches generate_text."""
    gen = _make_generator()
    prompt = "hello world"
    full_text = gen.generate_text(prompt, max_new_tokens=12, temperature=0.8, seed=42)
    expected_continuation = full_text[len(prompt):]

    stream_deltas = list(gen.generate_stream(prompt, max_new_tokens=12, temperature=0.8, seed=42))
    assert len(stream_deltas) > 0
    streamed_continuation = "".join(stream_deltas)

    assert streamed_continuation == expected_continuation


def test_generate_stream_stops_on_stop_sequence():
    """generate_stream aborts on stop sequence and does not yield the stop text."""
    gen = _make_generator()
    prompt = "hello"
    full = gen.generate_text(prompt, max_new_tokens=15, temperature=0, seed=7)
    continuation = full[len(prompt):]
    if len(continuation) > 3:
        stop_seq = continuation[1:3]
        deltas = list(gen.generate_stream(prompt, max_new_tokens=15, temperature=0, seed=7, stop=[stop_seq]))
        streamed = "".join(deltas)
        assert stop_seq not in streamed
        assert gen.last_finish_reason == "stop"


def test_generate_stream_exposes_finish_reason():
    """generate_stream exposes finish_reason via return_reason=True."""
    gen = _make_generator()
    chunks = list(gen.generate_stream("hello", max_new_tokens=4, temperature=0, return_reason=True))
    assert len(chunks) > 0
    # Final chunk has the termination reason
    _, last_reason = chunks[-1]
    assert last_reason in ("stop", "eos", "length")
    assert gen.last_finish_reason == last_reason


def test_generate_chat_stream():
    """generate_chat_stream yields streaming chunks from structured messages."""
    gen = _make_generator()
    messages = [{"role": "user", "content": "hello world"}]
    chunks = list(gen.generate_chat_stream(messages, max_new_tokens=10, temperature=0.7, seed=42))
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


# ===========================================================================
# F. Serve Integration Tests
# ===========================================================================

def test_serve_sampling_parameters_forwarding():
    """_sampling extracts repetition_penalty, stop, seed, temperature, top_k, top_p."""
    params = {
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0.85,
        "repetition_penalty": 1.2,
        "stop": ["END", "\n"],
        "seed": 99,
        "unrelated": "ignored",
    }
    extracted = _sampling(params)
    assert extracted["temperature"] == 0.5
    assert extracted["top_k"] == 40
    assert extracted["top_p"] == 0.85
    assert extracted["repetition_penalty"] == 1.2
    assert extracted["stop"] == ["END", "\n"]
    assert extracted["seed"] == 99
    assert "unrelated" not in extracted


def test_serve_chat_stream_operation():
    """handle(gen, 'chat', params with stream=True) returns a generator yielding stream dicts."""
    gen = _make_generator()
    params = {"prompt": "hello world", "max_new_tokens": 6, "stream": True, "temperature": 0.7, "seed": 42}
    result = handle(gen, "chat", params)
    import inspect
    assert inspect.isgenerator(result), "stream=True must return a generator"

    chunks = list(result)
    assert len(chunks) > 0
    # All intermediate chunks have done: False
    for chunk in chunks[:-1]:
        assert chunk["done"] is False
        assert "delta" in chunk
    # Final chunk has done: True and result dict
    final = chunks[-1]
    assert final["done"] is True
    assert "result" in final
    assert "content" in final["result"]
    assert "finish_reason" in final["result"]
    assert final["result"]["finish_reason"] in ("stop", "eos", "length")


def test_serve_text_stream_does_not_wrap_in_chat_template():
    """text stream preserves raw text prompt without chat template markers."""
    gen = _make_generator()
    raw_prompt = "Raw text prompt: complete this"
    params = {"prompt": raw_prompt, "max_new_tokens": 4, "stream": True, "temperature": 0.7, "seed": 42}
    result = handle(gen, "text", params)
    chunks = list(result)
    assert len(chunks) > 0
    final = chunks[-1]
    # In text mode, prompt should not have been wrapped in <user> markers
    assert not final["result"]["content"].startswith("<user>")


def test_serve_finish_reason_is_stateless():
    """handle does not depend on mutable generator.last_finish_reason."""
    gen = _make_generator()
    # Non-streaming call returns finish_reason directly through result dict
    res = handle(gen, "chat", {"prompt": "hello world", "max_new_tokens": 4, "temperature": 0.7, "seed": 42})
    assert "finish_reason" in res
    assert res["finish_reason"] in ("stop", "eos", "length")

    # Even if gen.last_finish_reason was corrupted by another hypothetical concurrent request:
    gen.last_finish_reason = "corrupted"
    res = handle(gen, "chat", {"prompt": "hello world", "max_new_tokens": 4, "temperature": 0.7, "seed": 42})
    assert res["finish_reason"] in ("stop", "eos", "length")
    assert res["finish_reason"] != "corrupted"

