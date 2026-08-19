"""Evaluation metrics and harness.

Parameter count is a ceiling, not a measurement. These tests pin the metrics to
properties that must hold regardless of the model being measured: a distribution
scores zero against itself, an identical signal has infinite SNR, a perfect
transcript has zero error rate, and a harness that cannot run a suite says so
instead of reporting a number it did not compute.
"""

import math

import pytest
import torch

from conftest import tiny_config
from model.eval import EvalHarness, EvalReport, default_harness
from model.eval.audio import (
    character_error_rate,
    mel_distance,
    si_sdr,
    speaker_similarity,
    word_error_rate,
)
from model.eval.image import alignment_score, fid
from model.eval.metrics import (
    cosine_alignment,
    frechet_from_features,
    gaussian_statistics,
    levenshtein,
)
from model.eval.text import bits_per_byte, perplexity, token_accuracy
from model.eval.video import fvd, temporal_consistency
from model.framer import FramerModel


def eval_config(**overrides):
    base = dict(
        text_only=False,
        image_size=32,
        patch_size=16,
        vision_d_model=32,
        vision_n_heads=4,
        vision_n_layers=1,
        audio_n_fft=64,
        audio_hop_length=16,
        audio_n_mels=16,
        audio_max_frames=32,
        audio_d_model=32,
        audio_n_heads=4,
        audio_n_layers=1,
        diffusion_steps=10,
        diffusion_channels=64,
        video_frames=2,
        video_resolution=16,
        audio_gen_frames=16,
        audio_gen_channels=32,
    )
    base.update(overrides)
    return tiny_config(**base)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_frechet_distance_of_a_distribution_against_itself_is_zero():
    torch.manual_seed(0)
    features = torch.randn(64, 8)
    assert float(frechet_from_features(features, features)) == pytest.approx(0.0, abs=1e-4)


def test_frechet_distance_grows_with_separation():
    torch.manual_seed(0)
    base = torch.randn(128, 8)
    near = base + 0.1
    far = base + 5.0
    assert frechet_from_features(base, near) < frechet_from_features(base, far)


def test_frechet_distance_is_symmetric():
    torch.manual_seed(0)
    a, b = torch.randn(64, 6), torch.randn(64, 6) * 2 + 1
    assert float(frechet_from_features(a, b)) == pytest.approx(
        float(frechet_from_features(b, a)), rel=1e-4
    )


def test_gaussian_statistics_shapes_and_guards():
    mean, covariance = gaussian_statistics(torch.randn(32, 5))
    assert mean.shape == (5,) and covariance.shape == (5, 5)
    # Covariance must be symmetric, which the square root relies on.
    assert torch.allclose(covariance, covariance.t(), atol=1e-6)

    with pytest.raises(ValueError, match="at least two samples"):
        gaussian_statistics(torch.randn(1, 5))
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        gaussian_statistics(torch.randn(2, 3, 4))


def test_cosine_alignment_endpoints():
    a = torch.randn(8, 16)
    assert float(cosine_alignment(a, a)) == pytest.approx(1.0, abs=1e-5)
    assert float(cosine_alignment(a, -a)) == pytest.approx(-1.0, abs=1e-5)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("kitten", "sitting", 3),
        ("", "abc", 3),
        ("abc", "abc", 0),
        ("flaw", "lawn", 2),
    ],
)
def test_levenshtein(a, b, expected):
    assert levenshtein(list(a), list(b)) == expected


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def test_perplexity_is_finite_and_above_one():
    config = tiny_config()
    model = FramerModel(config)
    ids = torch.randint(0, config.vocab_size, (2, 8))
    value = perplexity(model, [(ids, ids)])
    assert math.isfinite(value) and value > 1.0


def test_perplexity_of_an_empty_evaluation_is_infinite():
    assert perplexity(FramerModel(tiny_config()), []) == float("inf")


def test_ignored_labels_are_excluded():
    """Padding must not be scored, or perplexity depends on batch shape."""
    config = tiny_config()
    model = FramerModel(config)
    ids = torch.randint(0, config.vocab_size, (1, 8))
    labels = ids.clone()
    labels[:, 4:] = -100
    assert math.isfinite(perplexity(model, [(ids, labels)]))
    assert perplexity(model, [(ids, torch.full_like(ids, -100))]) == float("inf")


def test_token_accuracy_is_a_fraction():
    config = tiny_config()
    model = FramerModel(config)
    ids = torch.randint(0, config.vocab_size, (2, 8))
    assert 0.0 <= token_accuracy(model, [(ids, ids)]) <= 1.0


def test_bits_per_byte_is_tokenizer_independent():
    """The same corpus at two token counts must give the same bits per byte."""
    # 100 tokens at 0.5 nats each over 200 bytes, versus 50 at 1.0 nats.
    assert bits_per_byte(0.5, 100, 200) == pytest.approx(bits_per_byte(1.0, 50, 200))
    assert bits_per_byte(1.0, 10, 0) == float("inf")


# --------------------------------------------------------------------------
# Image and video
# --------------------------------------------------------------------------


def test_image_fid_is_finite_and_near_zero_against_itself():
    model = FramerModel(eval_config()).eval()
    images = torch.randn(8, 3, 32, 32)
    assert fid(model, images, images) == pytest.approx(0.0, abs=1e-2)


def test_image_fid_separates_distributions():
    model = FramerModel(eval_config()).eval()
    real = torch.randn(8, 3, 32, 32)
    assert fid(model, real, real * 5 + 2) > fid(model, real, real)


def test_alignment_score_is_a_cosine():
    config = eval_config()
    model = FramerModel(config).eval()
    value = alignment_score(
        model, torch.randn(4, 3, 32, 32), torch.randint(0, config.vocab_size, (4, 6))
    )
    assert -1.0 <= value <= 1.0


def test_video_fvd_runs_without_a_video_vae():
    """The fallback path matters: not every preset has a video VAE."""
    model = FramerModel(eval_config()).eval()
    clips = torch.randn(4, 3, 2, 16, 16)
    assert math.isfinite(fvd(model, clips, clips))


def test_video_fvd_runs_with_a_video_vae():
    config = eval_config(
        video_gen_arch="spacetime_dit", video_vae_base_channels=8,
        video_vae_latent_channels=4, video_vae_temporal_downsample=2,
        video_vae_spatial_downsample=4, video_dit_d_model=48,
        video_dit_n_layers=1, video_dit_n_heads=4,
    )
    model = FramerModel(config).eval()
    clips = torch.randn(4, 3, 2, 16, 16)
    assert math.isfinite(fvd(model, clips, clips))


def test_temporal_consistency_is_zero_for_a_still_clip():
    still = torch.randn(1, 3, 1, 8, 8).expand(1, 3, 4, 8, 8)
    assert temporal_consistency(still) == pytest.approx(0.0, abs=1e-6)
    assert temporal_consistency(torch.randn(1, 3, 4, 8, 8)) > 0
    # A single frame has no transitions to measure.
    assert temporal_consistency(torch.randn(1, 3, 1, 8, 8)) == 0.0


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def test_si_sdr_is_large_for_an_identical_signal():
    signal = torch.randn(1, 512)
    assert si_sdr(signal, signal) > 60


def test_si_sdr_ignores_gain():
    """A vocoder with the right waveform and the wrong gain is not the error here.

    A perfect reconstruction has no residual, so the ratio is bounded only by
    floating-point error; comparing two such values exactly is meaningless.
    What matters is that scaling does not push the score down into the range a
    real distortion would produce.
    """
    signal = torch.randn(1, 512)
    for scale in (0.1, 3.0, 100.0):
        assert si_sdr(signal * scale, signal) > 60

    noisy = signal + 0.5 * torch.randn(1, 512)
    assert si_sdr(noisy * 3.0, signal) == pytest.approx(si_sdr(noisy, signal), rel=1e-3)


def test_si_sdr_falls_with_noise():
    signal = torch.randn(1, 512)
    assert si_sdr(signal + torch.randn(1, 512), signal) < si_sdr(signal, signal)


def test_mel_distance_is_zero_for_identical_spectrograms():
    mel = torch.randn(1, 16, 20)
    assert mel_distance(mel, mel) == pytest.approx(0.0)
    assert mel_distance(mel, mel + 1.0) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "reference,hypothesis,expected",
    [
        ("the quick brown fox", "the quick brown fox", 0.0),
        ("the quick brown fox", "the quick brown dog", 0.25),
        ("a b c d", "", 1.0),
        ("", "", 0.0),
    ],
)
def test_word_error_rate(reference, hypothesis, expected):
    assert word_error_rate(reference, hypothesis) == pytest.approx(expected)


def test_character_error_rate():
    assert character_error_rate("hello", "hello") == pytest.approx(0.0)
    assert character_error_rate("hello", "hallo") == pytest.approx(0.2)


def test_speaker_similarity_endpoints():
    embedding = torch.randn(4, 32)
    assert speaker_similarity(embedding, embedding) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def test_the_harness_runs_registered_suites():
    harness = EvalHarness(model=None, device="cpu")
    harness.register("dummy", lambda model, device, **kw: {"value": 1.0})
    report = harness.run()
    assert report.metrics["dummy"]["value"] == 1.0
    assert report.skipped == {}


def test_a_duplicate_suite_is_rejected():
    harness = EvalHarness()
    harness.register("dummy", lambda model, device, **kw: {})
    with pytest.raises(ValueError, match="already registered"):
        harness.register("dummy", lambda model, device, **kw: {})


def test_a_failing_suite_is_skipped_with_a_reason_not_silently():
    """A suite that could not run must not look like one that scored well."""
    harness = EvalHarness()

    def broken(model, device, **kw):
        raise ValueError("no inputs given")

    harness.register("broken", broken)
    harness.register("fine", lambda model, device, **kw: {"value": 2.0})

    report = harness.run()
    assert report.metrics["fine"]["value"] == 2.0
    assert "no inputs given" in report.skipped["broken"]


def test_strict_mode_reraises():
    harness = EvalHarness()
    harness.register("broken", lambda model, device, **kw: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        harness.run(strict=True)


def test_an_unregistered_suite_is_reported():
    assert "not registered" in EvalHarness().run(["missing"]).skipped["missing"]


def test_the_report_serialises():
    report = EvalReport()
    report.add("text", {"perplexity": 12.5})
    report.skip("image", "no inputs")

    payload = report.to_dict()
    assert payload["metrics"]["text"]["perplexity"] == 12.5
    assert payload["skipped"]["image"] == "no inputs"
    assert "perplexity" in report.to_json()
    assert "skipped" in report.summary()


def test_the_default_harness_covers_every_modality():
    model = FramerModel(eval_config()).eval()
    harness = default_harness(model)
    assert harness.names == ["audio", "image", "text", "video"]


def test_the_default_harness_runs_end_to_end():
    config = eval_config()
    model = FramerModel(config).eval()
    harness = default_harness(model)

    ids = torch.randint(0, config.vocab_size, (4, 8))
    images = torch.randn(4, 3, 32, 32)
    clips = torch.randn(4, 3, 2, 16, 16)
    waveform = torch.randn(1, 512)

    report = harness.run(inputs={
        "text": {"batches": [(ids, ids)]},
        "image": {"real": images, "fake": images, "captions": ids},
        "video": {"real": clips, "fake": clips},
        "audio": {
            "reference": waveform, "estimate": waveform,
            "transcript": "hello world", "hypothesis": "hello world",
        },
    })

    assert report.skipped == {}, report.skipped
    assert math.isfinite(report.metrics["text"]["perplexity"])
    assert math.isfinite(report.metrics["image"]["fid"])
    assert math.isfinite(report.metrics["video"]["fvd"])
    assert report.metrics["audio"]["wer"] == 0.0


def test_the_default_harness_reports_what_it_could_not_run():
    model = FramerModel(eval_config()).eval()
    report = default_harness(model).run()
    assert set(report.skipped) == {"text", "image", "audio", "video"}
    assert all("needs" in reason for reason in report.skipped.values())


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------


def test_text_benchmark_returns_perplexity_and_accuracy():
    """Text benchmark must compute both metrics on the provided corpus."""
    import os
    import tempfile

    from model.eval.benchmarks import evaluate_text_benchmark
    from model.tokenizer import FramerTokenizer

    config = eval_config()
    model = FramerModel(config).eval()
    tokenizer = FramerTokenizer(config.vocab_size)
    tokenizer.train(["hello world", "test corpus"], target_vocab_size=100)

    corpus = "hello world\ntest corpus"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        tmp.write(corpus)
        tmp_path = tmp.name

    try:
        result = evaluate_text_benchmark(model, tokenizer, tmp_path, seq_len=8, batch_size=1)
        assert result.benchmark == "wikitext-2"
        assert "perplexity" in result.metrics
        assert "token_accuracy" in result.metrics
        assert math.isfinite(result.metrics["perplexity"])
        assert 0.0 <= result.metrics["token_accuracy"] <= 1.0
        assert result.samples > 0
    finally:
        os.remove(tmp_path)


def test_text_benchmark_raises_on_missing_file():
    """Missing benchmark files must raise FileNotFoundError, not produce fake results."""
    from model.eval.benchmarks import evaluate_text_benchmark
    from model.tokenizer import FramerTokenizer

    config = eval_config()
    model = FramerModel(config).eval()
    tokenizer = FramerTokenizer(config.vocab_size)

    with pytest.raises(FileNotFoundError, match="not found"):
        evaluate_text_benchmark(model, tokenizer, "no_such_file.txt")


def test_text_benchmark_raises_on_empty_file():
    """Empty benchmark files must raise ValueError."""
    import os
    import tempfile

    from model.eval.benchmarks import evaluate_text_benchmark
    from model.tokenizer import FramerTokenizer

    config = eval_config()
    model = FramerModel(config).eval()
    tokenizer = FramerTokenizer(config.vocab_size)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="empty"):
            evaluate_text_benchmark(model, tokenizer, tmp_path)
    finally:
        os.remove(tmp_path)


def test_code_benchmark_returns_pass_at_1():
    """Code benchmark must compute pass@1 by running the supplied tests."""
    import json
    import os
    import tempfile

    from model.eval.benchmarks import evaluate_code_benchmark
    from model.generate import FramerGenerator
    from model.tokenizer import FramerTokenizer

    config = eval_config()
    model = FramerModel(config).eval()
    tokenizer = FramerTokenizer(config.vocab_size)
    tokenizer.train(["def test(): pass"], target_vocab_size=100)
    generator = FramerGenerator(model, tokenizer, "cpu")

    # Trivial passing case
    cases = [
        {
            "task_id": "test/0",
            "prompt": "def passes():\n    return True\n",
            "test": "assert passes() == True",
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
        for case in cases:
            tmp.write(json.dumps(case) + "\n")
        tmp_path = tmp.name

    try:
        result = evaluate_code_benchmark(generator, tmp_path, seed=0, limit=1)
        assert result.benchmark == "humaneval"
        assert "pass@1" in result.metrics
        assert 0.0 <= result.metrics["pass@1"] <= 1.0
        assert result.samples == 1
    finally:
        os.remove(tmp_path)


def test_code_benchmark_raises_on_missing_file():
    """Missing code benchmark files must raise FileNotFoundError."""
    from model.eval.benchmarks import evaluate_code_benchmark
    from model.generate import FramerGenerator
    from model.tokenizer import FramerTokenizer

    config = eval_config()
    model = FramerModel(config).eval()
    tokenizer = FramerTokenizer(config.vocab_size)
    generator = FramerGenerator(model, tokenizer, "cpu")

    with pytest.raises(FileNotFoundError, match="not found"):
        evaluate_code_benchmark(generator, "no_such_file.jsonl")


def test_code_benchmark_raises_on_malformed_json():
    """Invalid JSON in code benchmark must raise ValueError."""
    import os
    import tempfile

    from model.eval.benchmarks import evaluate_code_benchmark
    from model.generate import FramerGenerator
    from model.tokenizer import FramerTokenizer

    config = eval_config()
    model = FramerModel(config).eval()
    tokenizer = FramerTokenizer(config.vocab_size)
    generator = FramerGenerator(model, tokenizer, "cpu")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
        tmp.write("{not valid json\n")
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="invalid JSON"):
            evaluate_code_benchmark(generator, tmp_path)
    finally:
        os.remove(tmp_path)


def test_code_benchmark_raises_on_missing_required_keys():
    """Code benchmark cases without required keys must raise ValueError."""
    import json
    import os
    import tempfile

    from model.eval.benchmarks import evaluate_code_benchmark
    from model.generate import FramerGenerator
    from model.tokenizer import FramerTokenizer

    config = eval_config()
    model = FramerModel(config).eval()
    tokenizer = FramerTokenizer(config.vocab_size)
    generator = FramerGenerator(model, tokenizer, "cpu")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
        tmp.write(json.dumps({"task_id": "test/0"}) + "\n")
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="missing"):
            evaluate_code_benchmark(generator, tmp_path)
    finally:
        os.remove(tmp_path)


def test_code_benchmark_timeout_counts_as_failure():
    """Code that times out must be scored as failing."""
    from model.eval.benchmarks import _run_humaneval_test

    # Infinite loop
    code = "def loop():\n    while True: pass\n"
    test = "loop()"
    assert _run_humaneval_test(code, test, timeout=1) is False


def test_code_benchmark_exception_counts_as_failure():
    """Code that raises must be scored as failing."""
    from model.eval.benchmarks import _run_humaneval_test

    code = "def raises():\n    raise ValueError('fail')\n"
    test = "raises()"
    assert _run_humaneval_test(code, test) is False


def test_code_benchmark_passing_test_returns_true():
    """Code that passes its test must be scored as passing."""
    from model.eval.benchmarks import _run_humaneval_test

    code = "def add(a, b):\n    return a + b\n"
    test = "assert add(2, 3) == 5"
    assert _run_humaneval_test(code, test) is True


def test_benchmark_suite_registration_works_end_to_end():
    """Benchmark suites registered with the harness must run and report results."""
    import os
    import tempfile

    from model.eval import EvalHarness
    from model.tokenizer import FramerTokenizer

    config = eval_config()
    model = FramerModel(config).eval()
    tokenizer = FramerTokenizer(config.vocab_size)
    tokenizer.train(["test"], target_vocab_size=50)

    harness = EvalHarness(model, "cpu")

    # Register a dummy benchmark suite
    @harness.suite("dummy_text")
    def _dummy(model, device, **_):
        corpus = "test corpus"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write(corpus)
            tmp_path = tmp.name
        try:
            from model.eval.benchmarks import evaluate_text_benchmark
            result = evaluate_text_benchmark(model, tokenizer, tmp_path, seq_len=4, batch_size=1)
            return result.metrics
        finally:
            os.remove(tmp_path)

    report = harness.run(["dummy_text"])
    assert "dummy_text" in report.metrics
    assert "perplexity" in report.metrics["dummy_text"]


def test_benchmark_suite_missing_data_is_skipped():
    """A benchmark suite that cannot find its data must be reported as skipped."""
    from model.eval import EvalHarness

    config = eval_config()
    model = FramerModel(config).eval()
    harness = EvalHarness(model, "cpu")

    @harness.suite("missing_data")
    def _missing(model, device, **_):
        raise FileNotFoundError("benchmark data not found: /no/such/path")

    report = harness.run(["missing_data"])
    assert "missing_data" in report.skipped
    assert "not found" in report.skipped["missing_data"]
    assert "missing_data" not in report.metrics
