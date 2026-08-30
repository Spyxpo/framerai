"""Dense text recognition: the image side's missing error rate.

The audio suites have had a character error rate since the harness was written.
The image suites measure whether a picture looks right and whether it matches a
caption, and say nothing about whether the words inside it were read, so there
was no number to move and no way to tell a description of a page from an answer
about what is on it.

These tests pin the measurement, not a score. An untrained model reads nothing,
which is the floor the metric is supposed to report rather than a failure of
the suite.
"""

import math

import pytest
import torch

from conftest import tiny_config
from model.eval import default_harness
from model.eval.dense_text import (
    DEFAULT_FONT_SIZES,
    SAMPLE_LINES,
    build_samples,
    dense_text_accuracy,
    image_to_tensor,
    render_text_image,
)
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.tokenizer import FramerTokenizer


def mm_config(**overrides):
    base = dict(
        text_only=False, image_size=32, patch_size=16,
        vision_d_model=32, vision_n_heads=4, vision_n_layers=1,
        audio_n_fft=64, audio_hop_length=16, audio_n_mels=16, audio_max_frames=32,
        audio_d_model=32, audio_n_heads=4, audio_n_layers=1,
        diffusion_steps=10, diffusion_channels=64,
        video_frames=2, video_resolution=16,
        audio_gen_frames=16, audio_gen_channels=32,
        vocab_size=512, max_seq_len=512,
    )
    base.update(overrides)
    return tiny_config(**base)


def _generator(config=None):
    config = config or mm_config()
    tokenizer = FramerTokenizer(vocab_size=config.vocab_size)
    tokenizer.train(["the quarterly total was 4820"], target_vocab_size=config.vocab_size)
    return FramerGenerator(FramerModel(config).eval(), tokenizer, device="cpu")


# ── Rendering ─────────────────────────────────────────────────────────────

def _grey(image):
    import numpy as np

    return np.asarray(image.convert("L"))


def test_text_is_rendered_as_dark_marks_on_a_light_page():
    pixels = _grey(render_text_image("the quarterly total was 4820"))
    assert pixels.min() < 128, "there must be ink on the page"
    assert pixels.max() > 200, "and page around it"


def test_rendering_is_reproducible():
    first = render_text_image("invoice 10592", font_size=16)
    second = render_text_image("invoice 10592", font_size=16)
    assert (_grey(first) == _grey(second)).all()


def test_a_larger_font_leaves_more_ink():
    def ink(size):
        return int((_grey(render_text_image("section 3.2 page 47", font_size=size)) < 128).sum())

    assert ink(24) > ink(11)


def test_the_page_becomes_the_tensor_the_tower_expects():
    config = mm_config(vision_tiling=False)
    tensor = image_to_tensor(render_text_image("page 47"), config)
    assert tensor.shape == (3, config.image_size, config.image_size)
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_tiling_keeps_the_page_its_own_shape():
    config = mm_config(vision_tiling=True)
    tensor = image_to_tensor(render_text_image("page 47"), config)
    assert tensor.shape[2] > tensor.shape[1], "a line of text is wider than it is tall"


# ── The sweep ─────────────────────────────────────────────────────────────

def test_the_samples_sweep_font_size():
    samples = build_samples(config=mm_config(vision_tiling=False))
    assert len(samples) == len(SAMPLE_LINES)
    assert {s["font_size"] for s in samples} == set(DEFAULT_FONT_SIZES)
    assert all(isinstance(s["image"], torch.Tensor) for s in samples)


def test_every_sample_carries_the_text_it_was_rendered_from():
    for sample in build_samples(config=mm_config(vision_tiling=False)):
        assert sample["text"] in SAMPLE_LINES


# ── The metric ────────────────────────────────────────────────────────────

def test_an_untrained_model_reads_nothing_and_the_metric_says_so():
    generator = _generator()
    samples = build_samples(
        lines=("page 47",), font_sizes=(16,), config=generator.model.config
    )
    values = dense_text_accuracy(generator, samples=samples, max_new_tokens=4)

    # Edit distance is normalised by the reference, so a hypothesis longer than
    # the reference can exceed 1.0. What must hold is that it is a real,
    # non-negative number: an untrained model reading nothing is the floor this
    # metric exists to report, not a failure of the suite.
    assert values["cer"] >= 0.0 and math.isfinite(values["cer"])
    assert values["wer"] >= 0.0 and math.isfinite(values["wer"])
    assert values["samples"] == 1
    assert "cer_16px" in values


def test_the_rate_is_reported_per_font_size():
    generator = _generator()
    samples = build_samples(
        lines=("page 47", "invoice 10592"), font_sizes=(24, 11),
        config=generator.model.config,
    )
    values = dense_text_accuracy(generator, samples=samples, max_new_tokens=4)
    assert "cer_24px" in values and "cer_11px" in values


def test_a_perfect_reading_scores_zero():
    # The metric itself, without a model in the way: a hypothesis equal to the
    # reference must cost nothing, or every later number is meaningless.
    from model.eval.audio import character_error_rate

    assert character_error_rate("page 47", "page 47") == 0.0
    assert character_error_rate("page 47", "") == 1.0


def test_a_suite_with_nothing_to_read_says_so():
    with pytest.raises(ValueError, match="at least one rendered sample"):
        dense_text_accuracy(_generator(), samples=[])


# ── The harness ───────────────────────────────────────────────────────────

def test_the_harness_registers_it_and_says_what_it_needs():
    harness = default_harness(FramerModel(mm_config()).eval(), "cpu")
    assert "dense_text" in harness.names

    report = harness.run(["dense_text"])
    assert "generator" in report.skipped["dense_text"]


def test_the_harness_runs_it_when_given_a_generator():
    generator = _generator()
    harness = default_harness(generator.model, "cpu")
    samples = build_samples(
        lines=("page 47",), font_sizes=(16,), config=generator.model.config
    )

    report = harness.run(
        ["dense_text"],
        inputs={"dense_text": {"generator": generator, "samples": samples}},
        strict=True,
    )
    assert "cer" in report.metrics["dense_text"]
    assert "wer" in report.metrics["dense_text"]
