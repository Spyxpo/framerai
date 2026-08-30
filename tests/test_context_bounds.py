"""Bounding a request against the context window.

Nothing used to check a prompt against `max_seq_len`. The tokenizer had no
truncation argument, generation did not look, and RoPE computes positions on
the fly rather than indexing a table, so an over-long request produced angles
the model never trained on and returned degraded output with no diagnostic. On
a preset claiming a million tokens that is the difference between a window that
can be trusted and one that cannot: there was no way to tell a real long-context
answer from an extrapolated one.

These tests pin that the window is now divided up front, that what a request
gave up is reported, and that going past it fails loudly.
"""

import pytest
import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.modules.transformer import RotaryPositionalEmbedding
from model.serve import _window_report
from model.tokenizer import FramerTokenizer


def _tokenizer(vocab_size=300):
    tok = FramerTokenizer(vocab_size=vocab_size)
    tok.train(["the quick brown fox jumps", "hello world foo bar"], target_vocab_size=vocab_size)
    return tok


def _generator(**overrides):
    config = tiny_config(vocab_size=300, **overrides)
    return FramerGenerator(FramerModel(config).eval(), _tokenizer(), device="cpu")


# ── The tokenizer ─────────────────────────────────────────────────────────

def test_encode_bounds_what_it_returns():
    tok = _tokenizer()
    text = "the quick brown fox jumps over the lazy dog again and again"
    assert len(tok.encode(text, max_length=12)) == 12
    assert len(tok.encode(text, max_length=12, add_special=False)) == 12


def test_the_cap_counts_the_special_tokens():
    tok = _tokenizer()
    ids = tok.encode("the quick brown fox jumps over the lazy dog", max_length=8)
    assert ids[0] == tok.sos_id and ids[-1] == tok.eos_id
    assert len(ids) == 8, "the cap is what the model receives, markers included"


def test_which_end_survives_is_the_callers_choice():
    tok = _tokenizer()
    text = "alpha beta gamma delta epsilon zeta eta theta"
    head = tok.decode(tok.encode(text, max_length=8, keep="head", add_special=False))
    tail = tok.decode(tok.encode(text, max_length=8, keep="tail", add_special=False))
    assert text.startswith(head) and text.endswith(tail)
    assert head != tail


def test_a_short_text_is_untouched():
    tok = _tokenizer()
    assert tok.encode("hi", max_length=64) == tok.encode("hi")


def test_an_unusable_cap_says_so():
    tok = _tokenizer()
    with pytest.raises(ValueError, match="no room"):
        tok.encode("hello", max_length=1)
    with pytest.raises(ValueError, match="keep must be"):
        tok.encode("hello", keep="middle")


# ── RoPE ──────────────────────────────────────────────────────────────────

def test_a_position_past_the_window_is_refused():
    rope = RotaryPositionalEmbedding(head_dim=8, max_seq_len=32)
    rope(32)  # the last position that was configured for
    with pytest.raises(ValueError, match="past the configured context window"):
        rope(33)


def test_the_offset_counts_toward_the_window():
    rope = RotaryPositionalEmbedding(head_dim=8, max_seq_len=32)
    rope(2, offset=30)
    with pytest.raises(ValueError, match="past the configured context window"):
        rope(4, offset=30)


def test_a_scaled_window_is_the_extended_one():
    # A model trained at 64 with a 4x extension claims 256, and running at 200
    # is inside that claim rather than past it.
    rope = RotaryPositionalEmbedding(
        head_dim=8, max_seq_len=256, scaling_factor=4.0,
        scaling_type="yarn", original_max_seq_len=64,
    )
    cos, _ = rope(200)
    assert cos.shape[0] == 200


# ── The generator ─────────────────────────────────────────────────────────

def test_an_over_long_prompt_is_trimmed_and_the_loss_reported():
    gen = _generator(max_seq_len=64)
    long_prompt = "the quick brown fox jumps over the lazy dog " * 20

    gen.generate_text(long_prompt, max_new_tokens=8)
    assert gen.last_prompt_tokens_dropped > 0


def test_the_end_of_the_prompt_is_what_survives():
    gen = _generator(max_seq_len=64)
    kept, granted = gen._fit_to_window("alpha " * 200 + "the actual question", 8, [])
    assert kept.endswith("question"), "the question sits at the end of a long prompt"
    assert gen.last_prompt_tokens_dropped > 0
    assert 1 <= granted <= 8


def test_generation_is_clamped_to_what_is_left():
    gen = _generator(max_seq_len=64)
    gen.generate_text("hello", max_new_tokens=4096)
    assert gen.last_max_new_tokens < 4096
    assert gen.last_max_new_tokens >= 1


def test_a_request_that_fits_gives_nothing_up():
    gen = _generator(max_seq_len=512)
    gen.generate_text("hello", max_new_tokens=4)
    assert gen.last_prompt_tokens_dropped == 0
    assert gen.last_max_new_tokens == 4


def test_a_long_prompt_still_leaves_room_to_answer():
    gen = _generator(max_seq_len=64)
    gen.generate_text("the quick brown fox " * 50, max_new_tokens=32)
    assert gen.last_max_new_tokens >= 1, "a prompt must not fill the whole window"


def test_generation_never_runs_past_the_window():
    # The real proof: with the bound in place, a request that would previously
    # have extrapolated now completes instead of producing untrained positions.
    gen = _generator(max_seq_len=48)
    out = gen.generate_text("the quick brown fox jumps " * 30, max_new_tokens=512)
    assert isinstance(out, str)


# ── Reporting ─────────────────────────────────────────────────────────────

def test_the_report_is_silent_when_nothing_was_given_up():
    gen = _generator(max_seq_len=512)
    gen.generate_text("hello", max_new_tokens=4)
    assert _window_report(gen, 4) == {}


def test_the_report_names_what_was_given_up():
    gen = _generator(max_seq_len=64)
    gen.generate_text("the quick brown fox " * 40, max_new_tokens=256)
    report = _window_report(gen, 256)
    assert report["prompt_tokens_dropped"] > 0
    assert report["max_new_tokens_granted"] < 256


def test_modalities_that_fill_the_window_are_refused():
    from tests.test_inference_vision import mm_config

    config = mm_config(max_seq_len=8, vision_tiling=False)
    gen = FramerGenerator(FramerModel(config).eval(), _tokenizer(512), device="cpu")
    with pytest.raises(ValueError, match="leaving no room"):
        gen.generate_text("describe", max_new_tokens=1, image=torch.randn(3, 32, 32))
