"""Forward-pass smoke tests for every multimodal tower.

The rest of the suite exercises the text backbone. Nothing executed a vision,
audio, image-diffusion, or video forward pass, which is how a `RuntimeError` on
every single video forward shipped unnoticed. These tests are deliberately tiny
and shape-only: they assert the towers run and produce finite losses, not that
they produce good output.
"""

import pytest
import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.modules.audio_encoder import AudioEncoder
from model.modules.audio_generator import AudioGenerator
from model.modules.diffusion import DiffusionModule
from model.modules.video_generator import VideoGenerator
from model.modules.vision_encoder import VisionEncoder


def multimodal_config(**overrides):
    """Smallest config that still builds every tower."""
    base = dict(
        text_only=False,
        image_size=32,
        patch_size=16,
        vision_d_model=32,
        vision_n_heads=4,
        vision_n_layers=1,
        audio_sample_rate=16000,
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


def test_vision_encoder_forward():
    encoder = VisionEncoder(32, 16, 32, 4, 1, 0.0)
    out = encoder(torch.randn(2, 3, 32, 32))
    # 2x2 patches plus the CLS token.
    assert out.shape == (2, 5, 32)
    assert torch.isfinite(out).all()


def test_audio_encoder_forward():
    encoder = AudioEncoder(
        sample_rate=16000, n_fft=64, hop_length=16, n_mels=16,
        d_model=32, n_heads=4, n_layers=1, max_frames=32, dropout=0.0,
    )
    out = encoder(torch.randn(2, 4000))
    assert out.shape[0] == 2 and out.shape[2] == 32
    assert torch.isfinite(out).all()


def test_image_diffusion_forward_and_sample():
    module = DiffusionModule(in_channels=3, base_channels=64, context_dim=64, num_steps=10)
    loss = module(torch.randn(1, 3, 16, 16), context=torch.randn(1, 4, 64))
    assert loss.ndim == 0 and torch.isfinite(loss)

    sample = module.sample((1, 3, 16, 16), context=torch.randn(1, 4, 64), device="cpu")
    assert sample.shape == (1, 3, 16, 16)
    assert torch.isfinite(sample).all()


def test_video_generator_forward_and_sample():
    """Regression: the 3D U-Net sized its timestep embedding from the input
    channel count, so every forward raised a shape error before reaching a frame."""
    module = VideoGenerator(frames=2, resolution=16, base_channels=32, context_dim=64, num_steps=10)
    loss = module(torch.randn(1, 3, 2, 16, 16), context=torch.randn(1, 4, 64))
    assert loss.ndim == 0 and torch.isfinite(loss)

    video = module.sample(1, context=torch.randn(1, 4, 64), device="cpu")
    assert video.shape == (1, 3, 2, 16, 16)
    assert torch.isfinite(video).all()


def test_audio_generator_forward_and_sample():
    module = AudioGenerator(
        n_mels=16, n_frames=16, base_channels=32, context_dim=64,
        num_steps=10, sample_rate=16000, n_fft=64, hop_length=16,
    )
    loss = module(torch.randn(1, 16, 16), context=torch.randn(1, 4, 64))
    assert loss.ndim == 0 and torch.isfinite(loss)

    mel = module.sample(context=torch.randn(1, 4, 64), device="cpu")
    assert mel.shape == (1, 1, 16, 16)

    waveform = module.mel_to_waveform(mel, n_iter=2)
    assert waveform.numel() > 0 and torch.isfinite(waveform).all()


@pytest.mark.parametrize("modality", ["images", "audio"])
def test_model_encodes_input_modalities(modality):
    config = multimodal_config()
    model = FramerModel(config).eval()
    inputs = {
        "input_ids": torch.randint(0, config.vocab_size, (1, 8)),
        modality: torch.randn(1, 3, 32, 32) if modality == "images" else torch.randn(1, 4000),
    }
    with torch.no_grad():
        out = model(**inputs)
    # The prefix is sliced back off, so logits stay aligned with the input tokens.
    assert out["logits"].shape == (1, 8, config.vocab_size)


def test_model_forward_with_every_target():
    """Every generation head runs in one unified pass and contributes a loss."""
    config = multimodal_config()
    model = FramerModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    out = model(
        input_ids=input_ids,
        images=torch.randn(1, 3, 32, 32),
        audio=torch.randn(1, 4000),
        target_images=torch.randn(1, 3, 16, 16),
        target_video=torch.randn(1, 3, 2, 16, 16),
        target_audio=torch.randn(1, 16, 16),
        labels=input_ids,
    )

    for key in ("text_loss", "image_loss", "video_loss", "audio_loss"):
        assert key in out, f"missing {key}"
        assert torch.isfinite(out[key]), f"{key} is not finite"
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())
