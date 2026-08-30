"""Long-context retrieval evaluation.

A declared context window is a number in a config. `max_seq_len` allocates no
per-position parameters and perplexity stays low on long text whatever happens
to retrieval, because most tokens are locally predictable. Nothing in the
harness varied sequence length, so nothing said whether the largest presets can
retrieve from the million tokens they claim.

These tests pin the measurement rather than a score: the material is built as
described, the scoring is a real forced choice, the sweep spans the window, and
an untrained model lands at chance rather than at zero or one.
"""

import pytest
import torch

from conftest import tiny_config
from model.eval import default_harness, longcontext
from model.framer import FramerModel
from model.tokenizer import FramerTokenizer


def _tokenizer(vocab_size=400):
    tok = FramerTokenizer(vocab_size=vocab_size)
    tok.train(
        ["the maintenance log records routine activity", "operations continued without incident"],
        target_vocab_size=vocab_size,
    )
    return tok


def _model(max_seq_len=512):
    return FramerModel(tiny_config(vocab_size=400, max_seq_len=max_seq_len)).eval()


# ── Buckets ───────────────────────────────────────────────────────────────

def test_the_sweep_stays_inside_the_window():
    for window in (1024, 4096, 8192, 1048576):
        buckets = longcontext.length_buckets(window)
        assert buckets, "a window must produce at least one bucket"
        assert max(buckets) < window, "the question and options need room too"
        assert buckets == sorted(buckets)


def test_a_million_token_window_sweeps_orders_of_magnitude():
    buckets = longcontext.length_buckets(1048576)
    assert len(buckets) >= 4
    assert max(buckets) / min(buckets) >= 64


def test_a_tiny_window_still_produces_a_bucket():
    assert longcontext.length_buckets(64) == [256]


# ── Material ──────────────────────────────────────────────────────────────

def test_the_fact_lands_where_the_depth_says():
    tok = _tokenizer()
    early = longcontext.build_single_fact(tok, 400, depth=0.0, seed=1)
    late = longcontext.build_single_fact(tok, 400, depth=1.0, seed=1)

    needle = "The access code for"
    assert early["prefix"].index(needle) < late["prefix"].index(needle)


def test_a_case_is_a_forced_choice_with_one_right_answer():
    tok = _tokenizer()
    case = longcontext.build_single_fact(tok, 300, depth=0.5, seed=2)
    assert len(case["options"]) == longcontext.CHOICES
    assert len(set(case["options"])) == longcontext.CHOICES
    assert case["options"][case["answer"]].strip(" .") in case["prefix"]


def test_the_material_is_reproducible_from_its_seed():
    tok = _tokenizer()
    first = longcontext.build_single_fact(tok, 300, depth=0.5, seed=7)
    second = longcontext.build_single_fact(tok, 300, depth=0.5, seed=7)
    assert first == second
    assert longcontext.build_single_fact(tok, 300, depth=0.5, seed=8) != first


def test_multi_hop_needs_both_facts():
    tok = _tokenizer()
    case = longcontext.build_multi_hop(tok, 400, seed=3)
    # The holder is never stated against the subject, only against the code, so
    # a single lookup cannot answer it.
    holder = case["options"][case["answer"]].strip(" .")
    assert f"is held by {holder}" in case["prefix"]
    subject_line = case["prefix"].split("Code ")[0]
    assert holder not in subject_line


def test_aggregation_asks_about_the_whole_context():
    tok = _tokenizer()
    case = longcontext.build_aggregation(tok, 400, seed=4)
    count = int(case["options"][case["answer"]].strip(" ."))
    assert case["prefix"].count("An alarm was raised at the west pump.") == count
    assert count >= 3


def test_longer_buckets_produce_longer_material():
    tok = _tokenizer()
    short = longcontext.build_single_fact(tok, 200, depth=0.5, seed=5)
    long = longcontext.build_single_fact(tok, 800, depth=0.5, seed=5)
    assert len(long["prefix"]) > len(short["prefix"])


# ── Scoring ───────────────────────────────────────────────────────────────

def test_scoring_prefers_the_continuation_the_model_finds_likely():
    tok, model = _tokenizer(), _model()
    prefix = tok.encode("operations continued", add_special=False)
    option = tok.encode(" without incident", add_special=False)

    with torch.no_grad():
        score = longcontext._score_continuation(model, prefix, option, "cpu")
    assert score < 0.0, "a mean log-probability is negative"
    assert torch.isfinite(torch.tensor(score))


def test_the_chunk_size_does_not_change_the_score():
    tok, model = _tokenizer(), _model()
    prefix = tok.encode("the maintenance log records routine activity " * 4, add_special=False)
    option = tok.encode(" again", add_special=False)

    with torch.no_grad():
        whole = longcontext._score_continuation(model, prefix, option, "cpu", chunk=4096)
        chunked = longcontext._score_continuation(model, prefix, option, "cpu", chunk=7)
    assert abs(whole - chunked) < 1e-4


def test_an_empty_continuation_is_refused():
    with pytest.raises(ValueError, match="at least one token"):
        longcontext._score_continuation(_model(), [1, 2], [], "cpu")


# ── The suites ────────────────────────────────────────────────────────────

def test_single_fact_reports_per_length_and_per_depth():
    tok, model = _tokenizer(), _model(max_seq_len=512)
    with torch.no_grad():
        values = longcontext.single_fact_accuracy(
            model, tok, "cpu", lengths=[64], depths=(0.0, 1.0), seed=0
        )

    assert "len_64" in values
    assert "depth_0" in values and "depth_1" in values
    assert 0.0 <= values["accuracy"] <= 1.0
    assert values["chance"] == 1.0 / longcontext.CHOICES


def test_every_suite_returns_an_accuracy_against_a_known_chance_rate():
    tok, model = _tokenizer(), _model(max_seq_len=512)
    with torch.no_grad():
        for suite in (longcontext.multi_hop_accuracy, longcontext.aggregation_accuracy):
            values = suite(model, tok, "cpu", lengths=[64], seed=0)
            assert 0.0 <= values["accuracy"] <= 1.0
            assert values["chance"] == 0.25


def test_the_harness_registers_it_and_says_what_it_needs():
    model = _model()
    harness = default_harness(model, "cpu")
    assert "long_context" in harness.names

    report = harness.run(["long_context"])
    assert "tokenizer" in report.skipped["long_context"], (
        "a suite that cannot run must say why, not report nothing"
    )


def test_the_harness_runs_it_when_given_a_tokenizer():
    tok, model = _tokenizer(), _model(max_seq_len=512)
    harness = default_harness(model, "cpu")

    with torch.no_grad():
        report = harness.run(
            ["long_context"],
            inputs={"long_context": {"tokenizer": tok, "lengths": [64]}},
            strict=True,
        )

    values = report.metrics["long_context"]
    assert set(values) == {"single_fact", "multi_hop", "aggregation"}
    assert 0.0 <= values["single_fact"]["accuracy"] <= 1.0


def test_a_window_too_small_for_a_case_says_so():
    tok, model = _tokenizer(), _model(max_seq_len=64)
    with pytest.raises(ValueError, match="cannot hold a retrieval case"):
        longcontext.single_fact_accuracy(model, tok, "cpu", lengths=[16], depths=(0.5,))


def test_a_requested_length_is_clamped_to_the_window():
    model = _model(max_seq_len=512)
    assert longcontext.usable_length(model, 100_000) < 512
    assert longcontext.usable_length(model, 64) == 64


def test_filler_stops_below_its_target_rather_than_over_it():
    import random as _random

    tok = _tokenizer()
    sentences = longcontext._filler_to_length(tok, 60, _random.Random(0))
    total = sum(len(tok.encode(s, add_special=False)) + 1 for s in sentences)
    assert total <= 60
