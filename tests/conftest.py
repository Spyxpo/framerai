"""Shared pytest fixtures/helpers for FramerAI tests (CPU, tiny, fast)."""

import torch

from model.configs import FramerConfig


def tiny_config(**overrides):
    """A minimal text-only config that builds and runs quickly on CPU."""
    base = dict(
        vocab_size=256,
        max_seq_len=64,
        d_model=64,
        n_heads=8,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        text_only=True,
    )
    base.update(overrides)
    return FramerConfig(**base)


def tiny_moe_config(**overrides):
    base = dict(
        use_moe=True,
        n_experts=4,
        n_experts_per_tok=2,
        n_shared_experts=1,
        expert_d_ff=128,
        moe_layer_freq=1,
        first_dense_layers=0,
    )
    base.update(overrides)
    return tiny_config(**base)


def tiny_multimodal_config(**overrides):
    """The smallest config that still builds every tower, for CPU tests.

    Text-only builds skip the vision and audio encoders entirely, so anything
    exercising real image or audio understanding needs this instead.
    """
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


def random_ids(config, batch=2, length=None):
    length = length or (config.max_seq_len - 1)
    return torch.randint(0, config.vocab_size, (batch, length))
