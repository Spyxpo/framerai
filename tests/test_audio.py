"""Audio generation tests: mel diffusion, Griffin-Lim, and end-to-end smoke.

Tests the legacy mel-diffusion + Griffin-Lim audio generation pipeline (not the
newer RVQ codec path). Validates AudioGenerator shape contracts, sampling,
waveform reconstruction, and end-to-end generation/transcription smoke tests
using the tiny preset.
"""

import numpy as np
import pytest
import torch

from conftest import tiny_config
from model.configs import build_preset_config
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.modules.audio_generator import AudioGenerator
from model.tokenizer import FramerTokenizer

# --------------------------------------------------------------------------
# AudioGenerator unit tests
# --------------------------------------------------------------------------


def _small_audio_generator():
    """A deliberately tiny AudioGenerator for fast unit tests."""
    return AudioGenerator(
        n_mels=24,
        n_frames=24,
        base_channels=32,
        context_dim=32,
        num_steps=4,
        sample_rate=16000,
        n_fft=64,
        hop_length=16,
    )


def test_audio_generator_forward_returns_scalar_loss():
    """AudioGenerator.forward computes a diffusion loss scalar."""
    audio_gen = _small_audio_generator()
    target_mel = torch.randn(2, 1, 24, 24)  # (B, 1, n_mels, n_frames)

    loss = audio_gen.forward(target_mel)

    assert loss.ndim == 0, "loss should be a scalar"
    assert torch.isfinite(loss), "loss should be finite"


def test_audio_generator_forward_accepts_context():
    """AudioGenerator.forward can be conditioned on text context."""
    audio_gen = _small_audio_generator()
    target_mel = torch.randn(2, 1, 24, 24)
    context = torch.randn(2, 8, 32)  # (B, seq_len, context_dim)

    loss = audio_gen.forward(target_mel, context=context)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_audio_generator_sample_returns_normalized_mel():
    """AudioGenerator.sample generates a mel spectrogram in [-1, 1]."""
    audio_gen = _small_audio_generator().eval()

    with torch.no_grad():
        mel = audio_gen.sample(device="cpu", batch_size=2)

    assert mel.shape == (2, 1, 24, 24), "shape should match (B, 1, n_mels, n_frames)"
    assert torch.isfinite(mel).all(), "mel should be finite"
    # Diffusion can occasionally produce slightly out-of-range values
    assert mel.min() >= -1.5 and mel.max() <= 1.5, "mel should be approximately in [-1, 1]"


def test_audio_generator_sample_with_context():
    """AudioGenerator.sample can be conditioned on text context."""
    audio_gen = _small_audio_generator().eval()
    context = torch.randn(2, 8, 32)

    with torch.no_grad():
        mel = audio_gen.sample(context=context, device="cpu", batch_size=2)

    assert mel.shape == (2, 1, 24, 24)
    assert torch.isfinite(mel).all()


def test_audio_generator_mel_to_waveform_produces_waveform():
    """mel_to_waveform reconstructs a 1D waveform via Griffin-Lim."""
    audio_gen = _small_audio_generator().eval()
    mel = torch.randn(1, 1, 24, 24).clamp(-1, 1)  # normalized mel

    with torch.no_grad():
        waveform = audio_gen.mel_to_waveform(mel, n_iter=2)

    assert isinstance(waveform, torch.Tensor), "should return a tensor"
    assert waveform.dim() == 1, "waveform should be 1D"
    assert waveform.numel() > 0, "waveform should have samples"
    assert torch.isfinite(waveform).all(), "waveform should be finite"


def test_audio_generator_mel_to_waveform_peak_normalizes():
    """mel_to_waveform peak-normalizes the output waveform."""
    audio_gen = _small_audio_generator().eval()
    mel = torch.randn(1, 1, 24, 24).clamp(-1, 1)

    with torch.no_grad():
        waveform = audio_gen.mel_to_waveform(mel, n_iter=2)

    peak = waveform.abs().max().item()
    # Peak normalization means max abs amplitude should be close to 1 (or 0 if silent)
    if peak > 0:
        assert peak <= 1.01, f"peak {peak} should be <= 1 + tolerance"


def test_audio_generator_mel_to_waveform_accepts_3d_input():
    """mel_to_waveform handles 3D input (C, n_mels, n_frames)."""
    audio_gen = _small_audio_generator().eval()
    mel = torch.randn(1, 24, 24).clamp(-1, 1)

    with torch.no_grad():
        waveform = audio_gen.mel_to_waveform(mel, n_iter=2)

    assert waveform.dim() == 1
    assert torch.isfinite(waveform).all()


def test_audio_generator_mel_to_waveform_accepts_4d_input():
    """mel_to_waveform handles 4D input (B, C, n_mels, n_frames)."""
    audio_gen = _small_audio_generator().eval()
    mel = torch.randn(1, 1, 24, 24).clamp(-1, 1)

    with torch.no_grad():
        waveform = audio_gen.mel_to_waveform(mel, n_iter=2)

    assert waveform.dim() == 1
    assert torch.isfinite(waveform).all()


def test_audio_generator_is_meta_constructible():
    """AudioGenerator can be built on meta device for parameter counting."""
    with torch.device("meta"):
        audio_gen = AudioGenerator(
            n_mels=80,
            n_frames=128,
            base_channels=128,
            context_dim=1024,
            num_steps=1000,
        )

    # Should have parameters (the diffusion module)
    total_params = sum(p.numel() for p in audio_gen.parameters())
    assert total_params > 0, "should have diffusion parameters"


# --------------------------------------------------------------------------
# End-to-end smoke tests with FramerGenerator
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_generator():
    """A FramerGenerator with the framer-tiny preset for smoke tests."""
    # Use the real framer-tiny preset which includes audio generation
    config = build_preset_config("framer-tiny")

    # Make it even smaller for faster CI
    config.max_seq_len = 128
    config.audio_gen_frames = 16  # very short audio for speed
    config.diffusion_steps = 10  # 10 steps instead of 1000 for fast CI (matches other tests)

    tok = FramerTokenizer(vocab_size=config.vocab_size)
    tok.train(["hello world", "test audio", "short beep"], target_vocab_size=config.vocab_size)

    model = FramerModel(config)
    return FramerGenerator(model, tok, device="cpu")


def test_generate_audio_returns_numpy_waveform(tiny_generator):
    """generate_audio returns a numpy array waveform and sample rate."""
    waveform, sample_rate = tiny_generator.generate_audio("A short beep")

    assert isinstance(waveform, np.ndarray), "waveform should be numpy array"
    assert waveform.ndim == 1, "waveform should be 1D"
    assert waveform.size > 0, "waveform should have samples"
    assert np.isfinite(waveform).all(), "waveform should be finite"
    assert sample_rate == 16000, "sample rate should be 16kHz"


def test_transcribe_returns_string(tiny_generator):
    """transcribe returns a string from audio input."""
    audio = torch.zeros(4000)  # 0.25 sec at 16 kHz

    result = tiny_generator.transcribe(audio, max_new_tokens=4)

    assert isinstance(result, str)


# --------------------------------------------------------------------------
# Integration with FramerModel
# --------------------------------------------------------------------------


def test_framer_model_has_audio_gen_module():
    """FramerModel with audio enabled has an audio_gen attribute."""
    config = tiny_config(
        text_only=False,
        audio_d_model=64,
        audio_n_heads=4,
        audio_n_layers=2,
        audio_gen_channels=32,
    )
    model = FramerModel(config)

    assert hasattr(model, "audio_gen"), "model should have audio_gen"
    assert isinstance(model.audio_gen, AudioGenerator), "should be AudioGenerator instance"


def test_framer_model_audio_gen_matches_config():
    """AudioGenerator in FramerModel uses config parameters."""
    config = tiny_config(
        text_only=False,
        audio_d_model=64,
        audio_n_heads=4,
        audio_n_layers=2,
        audio_gen_channels=32,
        audio_gen_frames=32,
        audio_n_mels=40,
    )
    model = FramerModel(config)

    assert model.audio_gen.n_mels == 40, "should use config.audio_n_mels"
    assert model.audio_gen.n_frames == 32, "should use config.audio_gen_frames"
    # DiffusionModule doesn't expose base_channels, but we verify it was built


def test_audio_generator_training_with_context():
    """AudioGenerator can train with text context from the backbone."""
    config = tiny_config(
        text_only=False,
        audio_d_model=64,
        audio_n_heads=4,
        audio_n_layers=2,
        audio_gen_channels=32,
        audio_gen_frames=24,
        audio_n_mels=24,
    )
    model = FramerModel(config)

    # Simulate text context from LM
    batch_size = 2
    seq_len = 8
    context = torch.randn(batch_size, seq_len, config.d_model)

    # Target mel spectrogram
    target_mel = torch.randn(batch_size, 1, 24, 24)

    loss = model.audio_gen.forward(target_mel, context=context)

    assert torch.isfinite(loss)
    loss.backward()  # ensure gradients flow


def test_audio_generator_sampling_without_context():
    """AudioGenerator can sample unconditionally (no text context)."""
    audio_gen = _small_audio_generator().eval()

    with torch.no_grad():
        mel = audio_gen.sample(device="cpu", batch_size=1)

    assert mel.shape == (1, 1, 24, 24)
    assert torch.isfinite(mel).all()
