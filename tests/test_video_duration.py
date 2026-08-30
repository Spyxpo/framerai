"""Video: the frame rate that was asked for, and a shot longer than one window.

Three separate caps, none of them the model's. The writer hardcoded 100 ms a
frame, so every clip came back at 10 fps whatever the request said and whatever
the spacetime decoder was conditioned on. The route exposed only a prompt and a
frame count, though the worker had accepted size, frame rate and seed all
along. And duration was bounded by a single denoising window, so a long shot
could only be made as separate clips, which do not join.
"""

import os

import pytest
import torch
from PIL import Image

from model.configs import FramerConfig
from model.modules.latent_video import build_video_generator
from model.serve import DEFAULT_VIDEO_FPS, _save_video


def video_config(**overrides):
    base = dict(
        text_only=False,
        video_gen_arch="spacetime_dit",
        video_frames=8,
        video_resolution=32,
        d_model=64,
        sampler_steps=2,
    )
    base.update(overrides)
    return FramerConfig(**base)


def _frames(count=6, size=32):
    return [Image.new("RGB", (size, size), (i * 8 % 255, 0, 0)) for i in range(count)]


# ── The writer ────────────────────────────────────────────────────────────

def test_a_clip_is_written_at_the_rate_it_was_asked_for(tmp_path):
    name = _save_video(_frames(), str(tmp_path), fps=25)
    written = tmp_path / name
    assert written.exists() and written.stat().st_size > 0

    if name.endswith(".gif"):
        with Image.open(written) as clip:
            # 25 fps is 40 ms a frame. The old writer said 100 ms whatever was
            # requested, which is 10 fps and nothing else.
            assert clip.info["duration"] == 40


def test_a_different_rate_produces_a_different_clip(tmp_path):
    slow = _save_video(_frames(), str(tmp_path), fps=5)
    fast = _save_video(_frames(), str(tmp_path), fps=50)

    if slow.endswith(".gif") and fast.endswith(".gif"):
        with Image.open(tmp_path / slow) as a, Image.open(tmp_path / fast) as b:
            assert a.info["duration"] > b.info["duration"]


def test_an_absent_rate_falls_back_to_the_configured_one(tmp_path):
    name = _save_video(_frames(), str(tmp_path))
    if name.endswith(".gif"):
        with Image.open(tmp_path / name) as clip:
            # GIF stores frame times in centiseconds, so 24 fps is representable
            # only to the nearest 10 ms. That rounding is a property of the
            # format, and one of the reasons the container path exists.
            assert abs(clip.info["duration"] - round(1000 / DEFAULT_VIDEO_FPS)) <= 10


def test_a_rate_of_nothing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="fps must be positive"):
        _save_video(_frames(), str(tmp_path), fps=0)


def test_the_writer_leaves_one_file_behind(tmp_path):
    _save_video(_frames(), str(tmp_path), fps=24)
    assert len(os.listdir(tmp_path)) == 1, "a failed encode must not leave a stub"


# ── Duration ──────────────────────────────────────────────────────────────

def test_a_clip_longer_than_the_window_is_produced_whole():
    generator = build_video_generator(video_config()).eval()
    context = torch.randn(1, 4, 64)

    with torch.no_grad():
        clip = generator.sample_long(
            1, context, "cpu", frames=24, height=32, width=32,
            steps=2, window_frames=8, overlap_frames=2,
        )
    assert clip.shape[2] == 24, "duration is no longer capped at one window"


def test_a_request_inside_one_window_still_produces_that_window():
    generator = build_video_generator(video_config()).eval()
    context = torch.randn(1, 4, 64)

    with torch.no_grad():
        clip = generator.sample_long(
            1, context, "cpu", frames=8, height=32, width=32,
            steps=2, window_frames=8, overlap_frames=2,
        )
    assert clip.shape[2] == 8


def test_windows_are_held_to_what_came_before_them():
    # The join is what separates a long shot from concatenated clips, so the
    # carry-over is asserted directly: the solver is told the opening latent
    # frames are already decided.
    generator = build_video_generator(video_config()).eval()
    seen = {}

    original = generator.sampler.sample

    def recording_sample(*args, **kwargs):
        seen.setdefault("clamps", []).append(kwargs.get("clamp_fn"))
        return original(*args, **kwargs)

    generator.sampler.sample = recording_sample
    with torch.no_grad():
        generator.sample_long(
            1, torch.randn(1, 4, 64), "cpu", frames=24, height=32, width=32,
            steps=2, window_frames=8, overlap_frames=2,
        )

    clamps = seen["clamps"]
    assert len(clamps) > 1, "a long clip takes more than one window"
    assert clamps[0] is None, "the first window has nothing to carry from"
    assert all(c is not None for c in clamps[1:]), "every later window carries over"


def test_the_clamp_actually_holds_the_opening_frames():
    generator = build_video_generator(video_config()).eval()
    held = {}

    original = generator.sampler.sample

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        clamp_fn = kwargs.get("clamp_fn")
        if clamp_fn is not None:
            probe = torch.zeros_like(result)
            held["after"] = clamp_fn(probe, 1.0)
        return result

    generator.sampler.sample = capture
    with torch.no_grad():
        generator.sample_long(
            1, torch.randn(1, 4, 64), "cpu", frames=24, height=32, width=32,
            steps=2, window_frames=8, overlap_frames=4,
        )

    after = held["after"]
    assert after[:, :, 0].abs().sum() > 0, "the opening frames were overwritten with carried content"
    assert after[:, :, -1].abs().sum() == 0, "the rest is left to the solver"


def test_an_overlap_that_swallows_the_window_is_refused():
    generator = build_video_generator(video_config()).eval()
    with pytest.raises(ValueError, match="overlap_frames"):
        generator.sample_long(1, None, "cpu", frames=16, window_frames=8, overlap_frames=8)
    with pytest.raises(ValueError, match="window_frames must be positive"):
        generator.sample_long(1, None, "cpu", frames=16, window_frames=0)


def test_no_overlap_is_allowed_even_though_it_is_a_worse_join():
    generator = build_video_generator(video_config()).eval()
    with torch.no_grad():
        clip = generator.sample_long(
            1, torch.randn(1, 4, 64), "cpu", frames=16, height=32, width=32,
            steps=2, window_frames=8, overlap_frames=0,
        )
    assert clip.shape[2] == 16
