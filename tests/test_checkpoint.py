"""Deferred initialization and checkpoint save/load.

`framer-2t-a49b` is 3.6 TiB in bf16, so it cannot be constructed on one host
even to be sharded. These tests cover the two mechanisms that make a model that
size expressible: building shapes on the meta device and materializing them
afterwards, and writing checkpoints through a path that does not assume the
whole model fits in one file.
"""

import pytest
import torch

from conftest import tiny_config
from model.configs import FramerConfig
from model.framer import FramerModel
from model.training.checkpoint import (
    gather_full_state_dict,
    load_sharded,
    save_full,
    save_sharded,
)
from model.training.optim import build_optimizer


def multimodal_config(**overrides):
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
# Deferred initialization
# --------------------------------------------------------------------------


def test_meta_build_allocates_nothing():
    model = FramerModel.from_config_meta(multimodal_config())
    assert all(p.is_meta for p in model.parameters())
    assert all(b.is_meta for b in model.buffers())
    assert sum(p.numel() for p in model.parameters()) > 0


def test_meta_build_shapes_match_a_real_build():
    config = multimodal_config()
    meta = FramerModel.from_config_meta(config)
    real = FramerModel(config)
    assert {n: p.shape for n, p in meta.named_parameters()} == {
        n: p.shape for n, p in real.named_parameters()
    }


def test_materialized_model_is_fully_initialized():
    """to_empty() allocates without contents; init_weights_ has to fill all of it."""
    model = FramerModel.from_config_meta(multimodal_config())
    model.to_empty(device="cpu")
    model.init_weights_(buffer_device="cpu")

    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"{name} holds uninitialized memory"
    for name, buffer in model.named_buffers():
        assert torch.isfinite(buffer).all(), f"{name} holds uninitialized memory"


def test_buffers_are_recomputed_exactly():
    """Buffers are derived, not learned, so they must match a normal build."""
    config = multimodal_config()
    model = FramerModel.from_config_meta(config)
    model.to_empty(device="cpu")
    model.init_weights_(buffer_device="cpu")

    reference = dict(FramerModel(config).named_buffers())
    recomputed = dict(model.named_buffers())
    assert set(reference) == set(recomputed)
    for name, expected in reference.items():
        assert torch.allclose(recomputed[name], expected), f"{name} was not recomputed"


def test_materialized_model_runs_a_forward_pass():
    config = multimodal_config()
    model = FramerModel.from_config_meta(config)
    model.to_empty(device="cpu")
    model.init_weights_(buffer_device="cpu").eval()

    with torch.no_grad():
        out = model(
            input_ids=torch.randint(0, config.vocab_size, (1, 8)),
            images=torch.randn(1, 3, 32, 32),
            audio=torch.randn(1, 4000),
        )
    assert out["logits"].shape == (1, 8, config.vocab_size)
    assert torch.isfinite(out["logits"]).all()


def test_rope_buffer_survives_reset_for_every_scaling_type():
    """The NTK base adjustment must not be applied twice on a rebuild."""
    for scaling_type in ("none", "linear", "ntk", "yarn"):
        config = tiny_config(
            rope_scaling_factor=4.0, rope_scaling_type=scaling_type,
            rope_original_max_seq_len=64,
        )
        reference = FramerModel(config)
        rebuilt = FramerModel(config)
        rebuilt.reset_buffers()
        for name, expected in reference.named_buffers():
            actual = dict(rebuilt.named_buffers())[name]
            assert torch.allclose(actual, expected), f"{scaling_type}: {name} drifted on reset"


def test_init_weights_is_idempotent_in_shape():
    model = FramerModel(tiny_config())
    before = {n: p.shape for n, p in model.named_parameters()}
    model.init_weights_()
    assert {n: p.shape for n, p in model.named_parameters()} == before


def test_the_flagship_preset_is_meta_constructible():
    """The whole point: a 2T definition builds without allocating 3.6 TiB."""
    config = FramerConfig.from_preset("framer-2t-a49b")
    model = FramerModel.from_config_meta(config)
    total = sum(p.numel() for p in model.parameters())
    assert 1.9e12 < total < 2.1e12
    assert all(p.is_meta for p in model.parameters())


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path):
    config = tiny_config()
    model = FramerModel(config)
    optimizer = build_optimizer(model, config)

    path = save_sharded(model, optimizer, str(tmp_path / "ckpt"), step=7, config=config)

    restored = FramerModel(config)
    restored_optimizer = build_optimizer(restored, config)
    step = load_sharded(restored, restored_optimizer, path)

    assert step == 7
    for (name, a), (_, b) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), f"{name} did not survive the round trip"


def test_save_writes_readable_sidecars(tmp_path):
    import json

    config = tiny_config()
    model = FramerModel(config)
    path = save_sharded(model, None, str(tmp_path / "ckpt"), step=3, config=config)

    with open(f"{path}/config.json") as f:
        assert json.load(f)["d_model"] == config.d_model
    with open(f"{path}/checkpoint_meta.json") as f:
        assert json.load(f)["step"] == 3


def test_save_without_an_optimizer(tmp_path):
    config = tiny_config()
    model = FramerModel(config)
    path = save_sharded(model, None, str(tmp_path / "ckpt"), step=1, config=config)
    assert load_sharded(FramerModel(config), None, path) == 1


def test_gather_matches_state_dict_at_world_size_one():
    model = FramerModel(tiny_config())
    gathered = gather_full_state_dict(model)
    reference = model.state_dict()
    assert set(gathered) == set(reference)
    assert all(torch.equal(gathered[k], reference[k]) for k in reference)


def test_save_full_writes_one_portable_file(tmp_path):
    config = tiny_config()
    model = FramerModel(config)
    path = save_full(model, str(tmp_path / "model.pt"), step=11, config=config)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == 11
    assert payload["config"]["d_model"] == config.d_model

    restored = FramerModel(config)
    restored.load_state_dict(payload["model_state_dict"])


def test_loading_a_missing_checkpoint_says_why(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="sharded"):
        load_sharded(FramerModel(tiny_config()), None, str(empty))
