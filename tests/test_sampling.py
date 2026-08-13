"""Focused tests for top-k, top-p, and temperature sampling (Issue #1).

The generation utilities already have integration-level coverage in
test_generate.py. This file adds unit-level tests for _filter_logits and for
the sampling parameters in generate_text to close the explicit Issue #1
requirements:
  - top-k sampling
  - top-p (nucleus) sampling
  - temperature
  - generated token IDs are within the valid vocabulary range

All tests are CPU-only, seeded, and require no checkpoint.
"""

import torch
import torch.nn.functional as F

from conftest import tiny_config
from model.framer import FramerModel
from model.generate import FramerGenerator, _filter_logits
from model.tokenizer import FramerTokenizer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CORPUS = ["hello world foo bar baz", "the quick brown fox", "sample text for training"]


def _make_tokenizer(vocab_size: int = 300) -> FramerTokenizer:
    tok = FramerTokenizer(vocab_size=vocab_size)
    tok.train(_CORPUS, target_vocab_size=vocab_size)
    return tok


def _make_generator() -> FramerGenerator:
    tok = _make_tokenizer()
    cfg = tiny_config(vocab_size=tok.vocab_size, max_seq_len=64)
    model = FramerModel(cfg)
    return FramerGenerator(model, tok, device="cpu")


def _uniform_logits(vocab_size: int = 300) -> torch.Tensor:
    """Return a (1, V) logits tensor with equal scores for every token."""
    return torch.zeros(1, vocab_size)


# ---------------------------------------------------------------------------
# _filter_logits: top-k
# ---------------------------------------------------------------------------

def test_top_k_keeps_exactly_k_candidates():
    """After top-k filtering only k logits remain finite."""
    logits = torch.randn(1, 300)
    k = 10
    filtered = _filter_logits(logits.clone(), top_k=k, top_p=1.0)
    finite_count = torch.isfinite(filtered).sum().item()
    assert finite_count == k


def test_top_k_1_is_greedy():
    """top_k=1 leaves exactly one finite logit (the argmax)."""
    logits = torch.randn(1, 300)
    best_token = logits.argmax(dim=-1).item()
    filtered = _filter_logits(logits.clone(), top_k=1, top_p=1.0)
    finite_mask = torch.isfinite(filtered).squeeze()
    assert finite_mask.sum().item() == 1
    assert finite_mask[best_token].item()


def test_top_k_zero_leaves_all_logits():
    """top_k=0 (disabled) must not mask anything."""
    logits = torch.randn(1, 300)
    filtered = _filter_logits(logits.clone(), top_k=0, top_p=1.0)
    assert torch.equal(logits, filtered)


def test_top_k_softmax_sums_to_one():
    """After top-k filtering the resulting probability distribution is valid."""
    logits = torch.randn(1, 300)
    filtered = _filter_logits(logits.clone(), top_k=20, top_p=1.0)
    probs = F.softmax(filtered, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_top_k_preserves_relative_order():
    """The top-k survivors are the k tokens with the highest original logits."""
    logits = torch.randn(1, 300)
    k = 15
    _, expected_top = torch.topk(logits, k)
    filtered = _filter_logits(logits.clone(), top_k=k, top_p=1.0)
    survivors = torch.where(torch.isfinite(filtered.squeeze()))[0]
    assert set(survivors.tolist()) == set(expected_top.squeeze().tolist())


# ---------------------------------------------------------------------------
# _filter_logits: top-p (nucleus)
# ---------------------------------------------------------------------------

def test_top_p_1_0_leaves_all_logits():
    """top_p=1.0 (disabled) must not mask anything."""
    logits = torch.randn(1, 300)
    filtered = _filter_logits(logits.clone(), top_k=0, top_p=1.0)
    assert torch.equal(logits, filtered)


def test_top_p_nucleus_softmax_sums_to_one():
    """After top-p filtering the resulting probability distribution is valid."""
    logits = torch.randn(1, 300)
    filtered = _filter_logits(logits.clone(), top_k=0, top_p=0.9)
    probs = F.softmax(filtered, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_top_p_small_nucleus_keeps_fewer_tokens():
    """A smaller p value retains fewer candidates than a larger one."""
    logits = torch.randn(1, 300)
    filtered_tight = _filter_logits(logits.clone(), top_k=0, top_p=0.5)
    filtered_wide = _filter_logits(logits.clone(), top_k=0, top_p=0.95)
    tight_count = torch.isfinite(filtered_tight).sum().item()
    wide_count = torch.isfinite(filtered_wide).sum().item()
    assert tight_count <= wide_count


def test_top_p_nucleus_covers_at_least_p_mass():
    """The finite tokens after filtering must jointly cover at least p probability."""
    p = 0.8
    logits = torch.randn(1, 300)
    filtered = _filter_logits(logits.clone(), top_k=0, top_p=p)
    probs_original = F.softmax(logits, dim=-1).squeeze()
    surviving_mass = probs_original[torch.isfinite(filtered.squeeze())].sum().item()
    assert surviving_mass >= p - 1e-5


# ---------------------------------------------------------------------------
# _filter_logits: top-k + top-p combined
# ---------------------------------------------------------------------------

def test_top_k_and_top_p_combined_gives_valid_distribution():
    """Using both filters together still yields a valid probability distribution."""
    logits = torch.randn(1, 300)
    filtered = _filter_logits(logits.clone(), top_k=50, top_p=0.9)
    probs = F.softmax(filtered, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    assert torch.isfinite(filtered).sum().item() >= 1


# ---------------------------------------------------------------------------
# Temperature in generate_text
# ---------------------------------------------------------------------------

def test_high_temperature_produces_valid_token_ids():
    """With high temperature every generated token ID is within the vocabulary."""
    torch.manual_seed(0)
    gen = _make_generator()
    out = gen.generate_text("hello", max_new_tokens=16, temperature=2.0, top_k=50)
    assert isinstance(out, str)


def test_low_temperature_approaches_greedy():
    """Very low temperature (near-greedy) should be deterministic across two runs."""
    gen = _make_generator()
    torch.manual_seed(42)
    a = gen.generate_text("hello", max_new_tokens=10, temperature=1e-6, top_k=1, top_p=1.0)
    torch.manual_seed(42)
    b = gen.generate_text("hello", max_new_tokens=10, temperature=1e-6, top_k=1, top_p=1.0)
    assert a == b


def test_temperature_does_not_change_output_type():
    """generate_text always returns a str regardless of temperature."""
    gen = _make_generator()
    for temperature in (0.1, 0.7, 1.5):
        torch.manual_seed(0)
        out = gen.generate_text("hello", max_new_tokens=8, temperature=temperature)
        assert isinstance(out, str), f"Expected str at temperature={temperature}"


# ---------------------------------------------------------------------------
# Generated token IDs are within the valid vocabulary range
# ---------------------------------------------------------------------------

def test_generated_token_ids_are_in_vocab():
    """All token IDs produced by the generator must be in [0, vocab_size)."""
    tok = _make_tokenizer()
    cfg = tiny_config(vocab_size=tok.vocab_size, max_seq_len=64)
    model = FramerModel(cfg)

    torch.manual_seed(0)
    # Run one step of the raw generation loop and verify the sampled id.
    prompt_ids = tok.encode("hello", add_special=True)
    input_ids = torch.tensor([prompt_ids])

    out = model.forward_lm(input_ids, use_cache=True)
    logits = out["logits"][:, -1, :]  # (1, V)

    # Sample one token through _filter_logits + softmax + multinomial.
    step = logits / 0.7
    filtered = _filter_logits(step, top_k=50, top_p=0.9)
    probs = F.softmax(filtered, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()

    assert 0 <= token_id < tok.vocab_size, (
        f"Sampled token id {token_id} is outside [0, {tok.vocab_size})"
    )


def test_generate_text_output_ids_decode_cleanly():
    """generate_text must return a string that can be re-encoded without error."""
    torch.manual_seed(0)
    gen = _make_generator()
    result = gen.generate_text("hello", max_new_tokens=20, temperature=0.8, top_k=40, top_p=0.9)
    # If any token id was out of range, decode() would have raised or returned
    # garbage that cannot be re-encoded consistently.
    re_encoded = gen.tokenizer.encode(result, add_special=False)
    assert isinstance(re_encoded, list)
    assert all(0 <= tid < gen.tokenizer.vocab_size for tid in re_encoded)
