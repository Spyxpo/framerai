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
