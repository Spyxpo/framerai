"""Editing images, animating them, sampler controls, and reading video."""

import numpy as np
import pytest
import torch
from PIL import Image
from test_latent_diffusion import latent_config
from test_spacetime_video import video_config

from model.framer import FramerModel
from model.generate import FramerGenerator
from model.modules.flow import ODESampler
from model.tokenizer import FramerTokenizer


def _gen(config):
    torch.manual_seed(0)
    tok = FramerTokenizer(vocab_size=config.vocab_size)
    tok.train(["a red square on blue"], target_vocab_size=config.vocab_size)
    return FramerGenerator(FramerModel(config).eval(), tok, "cpu")


def _picture(size=32, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


# ── Sampler ───────────────────────────────────────────────────────────────

def test_the_uniform_timeline_is_unchanged():
    torch.testing.assert_close(ODESampler.timeline(4), torch.linspace(0, 1, 5))


def test_shift_spends_more_steps_at_high_noise():
    plain, shifted = ODESampler.timeline(10), ODESampler.timeline(10, shift=3.0)
    assert shifted[0] == 0 and shifted[-1] == 1
    assert torch.all(torch.diff(shifted) > 0)
    # The early steps are smaller, so more of them sit near pure noise.
    assert shifted[1] < plain[1]


def test_a_late_start_begins_at_that_time():
    timeline = ODESampler.timeline(4, start=0.5)
    assert timeline[0] == 0.5 and timeline[-1] == 1.0


def test_guidance_outside_the_interval_skips_the_unconditional_pass():
    calls = []

    def velocity(x, t, ctx):
        calls.append(x.shape[0])
        return torch.zeros_like(x)

    sampler = ODESampler(steps=10)
    ctx, null = torch.randn(1, 2, 4), torch.zeros(1, 1, 4)
    sampler.sample(velocity, (1, 3), context=ctx, null_context=null, cfg_scale=4.0)
    assert calls.count(2) == 10

    calls.clear()
    sampler.sample(velocity, (1, 3), context=ctx, null_context=null, cfg_scale=4.0, cfg_interval=(0.2, 0.6))
    assert 0 < calls.count(2) < 10 and calls.count(1) > 0


def test_starting_from_data_at_time_one_returns_it():
    sampler = ODESampler(steps=4)
    init = torch.randn(1, 3)
    out = sampler.sample(lambda x, t, c: torch.ones_like(x), (1, 3), init=init, start=1.0)
    torch.testing.assert_close(out, init)


# ── Images ────────────────────────────────────────────────────────────────

def test_strength_zero_returns_the_encoded_original():
    gen = _gen(latent_config())
    source = _picture()
    a = np.asarray(gen.generate_image("x", init_image=source, strength=0.0, seed=1)[0], dtype=np.float32)
    b = np.asarray(gen.generate_image("y", init_image=source, strength=0.0, seed=2)[0], dtype=np.float32)
    assert np.allclose(a, b), "nothing is redrawn at strength 0, whatever the prompt"


def test_strength_controls_how_far_the_result_moves():
    gen = _gen(latent_config())
    source = _picture()
    base = np.asarray(gen.generate_image("x", init_image=source, strength=0.0, seed=1)[0], dtype=np.float32)

    def distance(strength):
        out = gen.generate_image("x", init_image=source, strength=strength, seed=1)[0]
        return np.abs(np.asarray(out, dtype=np.float32) - base).mean()

    assert distance(0.2) <= distance(1.0)


def test_the_output_takes_the_init_image_size():
    gen = _gen(latent_config())
    images, request = gen.generate_image("x", init_image=_picture(64), strength=0.5, return_request=True)
    assert images[0].size == (64, 64) and (request.width, request.height) == (64, 64)


def test_inpainting_holds_everything_outside_the_mask(monkeypatch):
    gen = _gen(latent_config())
    diffusion = gen.model.diffusion
    seen = {}
    original = diffusion.sampler.sample

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        seen["out"], seen["clamp"] = out, kwargs["clamp_fn"]
        return out

    monkeypatch.setattr(diffusion.sampler, "sample", spy)
    mask = Image.new("L", (32, 32), 0)
    mask.paste(255, (0, 0, 16, 32))  # repaint the left half
    gen.generate_image("x", init_image=_picture(), mask=mask, seed=3)

    init = diffusion.vae.encode_to_latent(
        torch.from_numpy(np.asarray(_picture(), dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None],
        sample=False,
    )
    # The right half of the latent grid is the original at t = 1.
    right = seen["out"][..., :, init.shape[-1] // 2:]
    torch.testing.assert_close(right, init[..., :, init.shape[-1] // 2:], atol=1e-5, rtol=0)


def test_editing_needs_the_latent_generator():
    gen = _gen(latent_config(image_gen_arch="unet"))
    with pytest.raises(ValueError, match="latent_dit"):
        gen.generate_image("x", init_image=_picture(), strength=0.5)


def test_out_of_range_strength_is_refused():
    gen = _gen(latent_config())
    with pytest.raises(ValueError, match="strength"):
        gen.generate_image("x", init_image=_picture(), strength=1.5)


def test_shift_and_guidance_interval_reach_the_sampler(monkeypatch):
    gen = _gen(latent_config())
    seen = {}
    original = gen.model.diffusion.sampler.sample

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(gen.model.diffusion.sampler, "sample", spy)
    gen.generate_image("x", shift=3.0, guidance_interval=[0.1, 0.8], seed=0)
    assert seen["shift"] == 3.0 and seen["cfg_interval"] == (0.1, 0.8)


# ── Video ─────────────────────────────────────────────────────────────────

def test_animating_an_image_opens_on_it(monkeypatch):
    config = video_config()
    gen = _gen(config)
    video_gen = gen.model.video_gen
    seen = {}
    original = video_gen.sampler.sample

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        seen["out"] = out
        return out

    monkeypatch.setattr(video_gen.sampler, "sample", spy)
    size = config.video_resolution
    frame = _picture(size)
    frames = gen.generate_video("x", first_frame=frame, seed=0)
    assert len(frames) == config.video_frames

    held = video_gen.vae.encode_to_latent(
        (torch.from_numpy(np.asarray(frame, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None])[:, :, None],
        sample=False,
    )[:, :, :1]
    torch.testing.assert_close(seen["out"][:, :, :1], held, atol=1e-5, rtol=0)


def test_animating_needs_the_spacetime_generator():
    gen = _gen(latent_config())
    if gen.model.config.video_gen_arch == "spacetime_dit":
        pytest.skip("this config already uses the spacetime generator")
    with pytest.raises(ValueError, match="spacetime_dit"):
        gen.generate_video("x", first_frame=_picture(gen.model.config.video_resolution))


def test_describe_video_reads_every_frame_in_order(monkeypatch):
    gen = _gen(latent_config())
    encoded = []
    original = gen.model.forward_vision

    def spy(x):
        encoded.append(x.shape[0])
        return original(x)

    monkeypatch.setattr(gen.model, "forward_vision", spy)
    size = gen.model.config.image_size
    frames = torch.rand(4, 3, size, size) * 2 - 1
    gen.describe_video(frames, prompt="what happens", max_new_tokens=2, temperature=0)
    assert encoded == [1, 1, 1, 1]
    per_frame = original(frames[:1]).shape[1]
    assert gen.last_input_tokens >= 4 * per_frame
