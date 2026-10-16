"""Voice: cached acoustic-token sampling, cloning, watermarking, long transcription."""

import numpy as np
import pytest
import torch
from test_audio_codec import audio_config

from model import audio_watermark
from model.audio_segment import split_on_silence
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.serve import handle
from model.tokenizer import FramerTokenizer


def _gen(**overrides):
    torch.manual_seed(0)
    tok = FramerTokenizer(vocab_size=300)
    tok.train(["hello world speak now"], target_vocab_size=300)
    config = audio_config(vocab_size=300, **overrides)
    return FramerGenerator(FramerModel(config).eval(), tok, "cpu")


def _codes_of(audio_gen, monkeypatch):
    """Make ``decode`` hand back the codes it was given."""
    monkeypatch.setattr(audio_gen, "decode", lambda codes: codes)


def _uncached_sample(audio_gen, context, n_frames, temperature, speaker, generator, prompt_codes=None):
    """The original algorithm: rerun the whole stack for every frame."""
    codes = torch.zeros(1, audio_gen.n_quantizers, 1, dtype=torch.long)
    if prompt_codes is not None:
        codes = torch.cat([codes, prompt_codes], dim=-1)
    start = codes.shape[-1]
    for _ in range(n_frames):
        hidden = audio_gen._run_lm(codes, context, speaker)
        logits = audio_gen.head(hidden)[:, :, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        picked = torch.multinomial(probs.reshape(-1, audio_gen.codebook_size), 1, generator=generator)
        codes = torch.cat([codes, picked.view(1, audio_gen.n_quantizers, 1)], dim=-1)
    return codes[:, :, start:]


# ── Cached sampling ───────────────────────────────────────────────────────

@pytest.mark.parametrize("speaker_on", [False, True])
def test_cached_sampling_matches_the_full_rerun(monkeypatch, speaker_on):
    gen = _gen(use_speaker_conditioning=speaker_on)
    audio_gen = gen.model.audio_gen.eval()
    context = gen._text_context("hello world")
    speaker = None
    if speaker_on:
        speaker = audio_gen.speaker_embedding(torch.randn(1, 2048))
    expected = _uncached_sample(audio_gen, context, 6, 0.9, speaker, torch.Generator().manual_seed(5))
    _codes_of(audio_gen, monkeypatch)
    got = audio_gen.sample(context, "cpu", n_frames=6, temperature=0.9, speaker=speaker,
                           generator=torch.Generator().manual_seed(5))
    assert torch.equal(got, expected)


def test_a_voice_prompt_is_continued_not_repeated(monkeypatch):
    gen = _gen(use_speaker_conditioning=True)
    audio_gen = gen.model.audio_gen.eval()
    context = gen._text_context("hello")
    prompt_codes = torch.randint(0, audio_gen.codebook_size, (1, audio_gen.n_quantizers, 3))
    expected = _uncached_sample(audio_gen, context, 4, 0.9, None, torch.Generator().manual_seed(1), prompt_codes)
    _codes_of(audio_gen, monkeypatch)
    got = audio_gen.sample(context, "cpu", n_frames=4, temperature=0.9,
                           generator=torch.Generator().manual_seed(1), prompt_codes=prompt_codes)
    assert got.shape[-1] == 4
    assert torch.equal(got, expected)


def test_frames_are_clamped_to_the_position_limit(monkeypatch):
    gen = _gen()
    audio_gen = gen.model.audio_gen.eval()
    _codes_of(audio_gen, monkeypatch)
    got = audio_gen.sample(None, "cpu", n_frames=10_000)
    assert got.shape[-1] == audio_gen.max_positions - 1


def test_top_k_restricts_the_choice(monkeypatch):
    gen = _gen()
    audio_gen = gen.model.audio_gen.eval()
    _codes_of(audio_gen, monkeypatch)
    torch.manual_seed(0)
    got = audio_gen.sample(None, "cpu", n_frames=5, top_k=1, temperature=1.0)
    again = audio_gen.sample(None, "cpu", n_frames=5, top_k=1, temperature=1.0)
    assert torch.equal(got, again), "top-1 is greedy, so it must repeat"


# ── Cloning ───────────────────────────────────────────────────────────────

def test_cloning_returns_audio_and_conditions_on_the_reference(monkeypatch):
    gen = _gen(use_speaker_conditioning=True)
    audio_gen = gen.model.audio_gen
    wave, sr = gen.generate_audio("hello", speaker_audio=torch.randn(4096), seed=3)
    assert sr == gen.model.config.codec_sample_rate
    assert wave.ndim == 1 and wave.size > 0

    # Two voices must reach the acoustic model as different conditioning.
    codes = torch.zeros(1, audio_gen.n_quantizers, 4, dtype=torch.long)
    a = audio_gen._run_lm(codes, None, audio_gen.speaker_embedding(torch.randn(1, 4096)))
    b = audio_gen._run_lm(codes, None, audio_gen.speaker_embedding(torch.randn(1, 4096) * 0.1 + 0.5))
    assert not torch.allclose(a, b)

    seen = {}
    original = audio_gen.sample

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(audio_gen, "sample", spy)
    gen.generate_audio("hello", speaker_audio=torch.randn(4096))
    assert seen["speaker"].shape == (1, gen.model.config.speaker_embed_dim)


def test_a_reference_transcript_adds_an_acoustic_prompt():
    gen = _gen(use_speaker_conditioning=True)
    wave, _ = gen.generate_audio("now", speaker_audio=torch.randn(4096), reference_text="hello world", seed=1)
    assert wave.size > 0


def test_a_model_without_speaker_conditioning_refuses_to_clone():
    gen = _gen(use_speaker_conditioning=False)
    with pytest.raises(ValueError, match="speaker_conditioning"):
        gen.generate_audio("hello", speaker_audio=torch.randn(4096))


def test_the_worker_marks_every_clone(tmp_path):
    import soundfile as sf

    gen = _gen(use_speaker_conditioning=True)
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.random.default_rng(0).standard_normal(48000).astype("float32") * 0.1, 24000)
    res = handle(gen, "audio", {"prompt": "hi", "speaker_path": str(ref), "watermark": False,
                                "out_dir": str(tmp_path)})
    assert res["cloned"] and res["watermarked"], "a clone is marked whatever the request says"


def test_the_worker_refuses_a_reference_that_is_too_short(tmp_path):
    import soundfile as sf

    gen = _gen(use_speaker_conditioning=True)
    ref = tmp_path / "short.wav"
    sf.write(ref, np.zeros(2400, dtype="float32"), 24000)
    with pytest.raises(ValueError, match="at least"):
        handle(gen, "audio", {"prompt": "hi", "speaker_path": str(ref), "out_dir": str(tmp_path)})


# ── Watermark ─────────────────────────────────────────────────────────────

def _speech(seconds=3.0, sr=24000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    voiced = 0.3 * np.sin(2 * np.pi * 140 * t) + 0.15 * np.sin(2 * np.pi * 280 * t)
    return ((voiced + 0.05 * rng.standard_normal(t.size)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype("float32")


def test_marked_audio_is_detected_and_clean_audio_is_not():
    clean = _speech()
    assert not audio_watermark.detect(clean)["watermarked"]
    assert audio_watermark.detect(audio_watermark.embed(clean))["watermarked"]


def test_the_mark_survives_volume_and_light_noise():
    marked = audio_watermark.embed(_speech())
    assert audio_watermark.detect(marked * 0.2)["watermarked"]
    noisy = marked + 0.01 * np.random.default_rng(1).standard_normal(marked.size)
    assert audio_watermark.detect(noisy)["watermarked"]


def test_the_mark_is_quiet():
    clean = _speech()
    marked = audio_watermark.embed(clean)
    snr = 10 * np.log10(np.mean(clean**2) / np.mean((marked - clean) ** 2))
    assert snr > 30


def test_another_key_does_not_detect_it():
    marked = audio_watermark.embed(_speech(), key="a")
    assert not audio_watermark.detect(marked, key="b")["watermarked"]


def test_the_worker_verifies_a_file(tmp_path):
    import soundfile as sf

    gen = _gen()
    path = tmp_path / "m.wav"
    sf.write(path, audio_watermark.embed(_speech()), 24000)
    assert handle(gen, "verify_audio", {"audio_path": str(path), "out_dir": str(tmp_path)})["watermarked"]


# ── Verifying speech at the rate it was made (Issue #509) ─────────────────
#
# The mark is laid sample by sample at the rate the generator produced, and
# resampling scrambles it. mel_diffusion, the default generator, produces
# audio_sample_rate (16 kHz); rvq_lm produces codec_sample_rate (24 kHz).
# verify_audio used to read every clip at the codec rate, so speech from a
# 16 kHz preset was resampled first and always came back unmarked.


def _generated(gen, tmp_path, monkeypatch, rate, seconds=4.0, **params):
    """The audio op's own output, with only the model's raw waveform stood in."""
    monkeypatch.setattr(gen, "generate_audio", lambda *a, **k: (_speech(seconds, sr=rate), rate))
    res = handle(gen, "audio", {"prompt": "read this aloud", "out_dir": str(tmp_path), **params})
    return res, tmp_path / res["file"]


def _verify(gen, path, tmp_path):
    return handle(gen, "verify_audio", {"audio_path": str(path), "out_dir": str(tmp_path)})


@pytest.mark.parametrize("arch", ["mel_diffusion", "rvq_lm"])
def test_the_worker_verifies_the_speech_it_generates(tmp_path, monkeypatch, arch):
    import soundfile as sf

    gen = _gen(audio_gen_arch=arch)
    rate = gen.model.audio_gen.sample_rate
    res, path = _generated(gen, tmp_path, monkeypatch, rate)
    assert res["watermarked"]
    assert sf.info(str(path)).samplerate == rate, "saved at the generator's own rate"

    verdict = _verify(gen, path, tmp_path)
    assert verdict["watermarked"], f"{arch} speech saved at {rate} Hz verified as {verdict}"


def _generator_rate(config):
    """The rate build_audio_generator gives the configured generator."""
    return config.codec_sample_rate if config.audio_gen_arch == "rvq_lm" else config.audio_sample_rate


@pytest.mark.parametrize("preset", ["framer-tiny", "framer-small", "framer-8b", "framer-160b-a16b"])
def test_every_preset_verifies_its_own_speech(tmp_path, monkeypatch, preset):
    """16 kHz and 24 kHz presets alike; the config is the shipped preset's."""
    from types import SimpleNamespace

    from model.configs.presets import build_preset_config

    config = build_preset_config(preset)
    rate = _generator_rate(config)
    gen = SimpleNamespace(model=SimpleNamespace(config=config), generate_audio=None)
    _, path = _generated(gen, tmp_path, monkeypatch, rate)
    verdict = _verify(gen, path, tmp_path)
    assert verdict["watermarked"], f"{preset} ({config.audio_gen_arch}, {rate} Hz) verified as {verdict}"


@pytest.mark.parametrize("arch", ["mel_diffusion", "rvq_lm"])
def test_speech_generated_unmarked_is_not_reported_marked(tmp_path, monkeypatch, arch):
    gen = _gen(audio_gen_arch=arch)
    res, path = _generated(gen, tmp_path, monkeypatch, gen.model.audio_gen.sample_rate, watermark=False)
    assert not res["watermarked"]
    assert not _verify(gen, path, tmp_path)["watermarked"]


def test_a_marked_clip_resampled_afterwards_is_not_claimed(tmp_path):
    """The documented limit stands: the check follows the file's rate, it does not undo a resample."""
    import soundfile as sf

    from model.data import _resample

    gen = _gen()
    marked = audio_watermark.embed(_speech(4.0, sr=16000))
    path = tmp_path / "resampled.wav"
    sf.write(path, _resample(torch.from_numpy(marked), 16000, 24000).numpy(), 24000)
    assert not _verify(gen, path, tmp_path)["watermarked"]


def test_an_empty_clip_is_unmarked(tmp_path):
    import soundfile as sf

    gen = _gen()
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype="float32"), 16000)
    verdict = _verify(gen, path, tmp_path)
    assert not verdict["watermarked"]
    assert verdict["score"] == 0.0


def test_a_file_that_is_not_audio_is_refused(tmp_path):
    gen = _gen()
    path = tmp_path / "not-audio.wav"
    path.write_bytes(b"this is not a wav file")
    # The worker loop answers a raised error with ok: false, never with a verdict.
    with pytest.raises(RuntimeError, match="Error opening"):
        _verify(gen, path, tmp_path)
    with pytest.raises(RuntimeError, match="Error opening"):
        _verify(gen, tmp_path / "missing.wav", tmp_path)


# ── Long transcription ────────────────────────────────────────────────────

def test_short_audio_is_one_piece():
    wav = torch.randn(1000)
    assert len(split_on_silence(wav, 2000)) == 1


def test_long_audio_is_cut_at_pauses_and_loses_nothing():
    sr = 16000
    tone = torch.sin(torch.arange(sr) / sr * 2 * torch.pi * 200)
    silence = torch.zeros(sr // 2)
    wav = torch.cat([tone, silence, tone, silence, tone])
    pieces = split_on_silence(wav, max_samples=int(1.8 * sr))
    assert all(p.numel() <= int(1.8 * sr) for p in pieces)
    assert sum(p.numel() for p in pieces) == wav.numel()
    # Every cut lands in a pause: the sample before each cut is silent.
    position = 0
    for piece in pieces[:-1]:
        position += piece.numel()
        assert wav[position - 1].abs() < 1e-6


def test_transcribe_covers_every_window(monkeypatch):
    gen = _gen()
    seen = []

    def fake_generate_text(prompt, audio=None, **kwargs):
        seen.append(audio.numel())
        gen.last_completion = f" part{len(seen)}"
        return prompt + gen.last_completion

    monkeypatch.setattr(gen, "generate_text", fake_generate_text)
    limit = gen.model.config.audio_max_frames * gen.model.config.audio_hop_length
    out = gen.transcribe(torch.randn(limit * 3 + 10), prompt="P:")
    assert len(seen) >= 4 and sum(seen) == limit * 3 + 10
    assert out == "P:" + " ".join(f"part{i}" for i in range(1, len(seen) + 1))
    assert gen.last_completion == " ".join(f"part{i}" for i in range(1, len(seen) + 1))
