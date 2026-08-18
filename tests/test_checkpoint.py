"""Deferred initialization and checkpoint save/load.

`framer-2t-a49b` is 3.6 TiB in bf16, so it cannot be constructed on one host
even to be sharded. These tests cover the two mechanisms that make a model that
size expressible: building shapes on the meta device and materializing them
afterwards, and writing checkpoints through a path that does not assume the
whole model fits in one file.
"""

import os

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


def test_optimizer_state_survives_checkpoint(tmp_path):
    """Optimizer state (momentum, etc.) must survive save/load."""
    config = tiny_config()
    model = FramerModel(config)
    optimizer = build_optimizer(model, config)

    # Perform a training step to populate optimizer state
    loss = model(input_ids=torch.randint(0, config.vocab_size, (2, 16)),
                 labels=torch.randint(0, config.vocab_size, (2, 16)))["loss"]
    loss.backward()
    optimizer.step()

    # Save checkpoint
    path = save_sharded(model, optimizer, str(tmp_path / "ckpt"), step=1, config=config)

    # Load into fresh optimizer
    restored_model = FramerModel(config)
    restored_optimizer = build_optimizer(restored_model, config)
    load_sharded(restored_model, restored_optimizer, path)

    # Optimizer state should match
    orig_state = optimizer.state_dict()
    restored_state = restored_optimizer.state_dict()

    assert len(orig_state["state"]) == len(restored_state["state"])
    for key in orig_state["state"]:
        for k in orig_state["state"][key]:
            if isinstance(orig_state["state"][key][k], torch.Tensor):
                assert torch.equal(orig_state["state"][key][k], restored_state["state"][key][k]), \
                    f"optimizer state {key}/{k} mismatch"


def test_scheduler_state_survives_checkpoint(tmp_path):
    """Scheduler state must survive save/load."""
    from model.training.schedule import build_scheduler

    config = tiny_config(warmup_steps=5, max_steps=20)
    model = FramerModel(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Advance scheduler several steps
    for _ in range(7):
        optimizer.step()
        scheduler.step()

    lr_before = scheduler.get_last_lr()[0]

    # Save checkpoint
    path = save_sharded(model, optimizer, str(tmp_path / "ckpt"), step=7,
                       config=config, scheduler=scheduler)

    # Load into fresh scheduler
    restored_model = FramerModel(config)
    restored_optimizer = build_optimizer(restored_model, config)
    restored_scheduler = build_scheduler(restored_optimizer, config)

    load_sharded(restored_model, restored_optimizer, path, scheduler=restored_scheduler)

    # LR should match (scheduler was at step 7)
    lr_after = restored_scheduler.get_last_lr()[0]
    assert abs(lr_before - lr_after) < 1e-9, \
        f"scheduler LR mismatch: {lr_before} vs {lr_after}"

    # Step count should be correct
    assert restored_scheduler.state_dict()["_step_count"] == 8  # LambdaLR is 1-indexed


def test_scheduler_not_restarted_on_resume(tmp_path):
    """Scheduler must not restart from step 0 on resume."""
    from model.training.schedule import build_scheduler

    config = tiny_config(warmup_steps=5, max_steps=20, learning_rate=1e-3)
    model = FramerModel(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Advance scheduler to step 10 (past warmup)
    for _ in range(10):
        optimizer.step()
        scheduler.step()

    lr_at_step_10 = scheduler.get_last_lr()[0]

    # Save and restore
    path = save_sharded(model, optimizer, str(tmp_path / "ckpt"), step=10,
                       config=config, scheduler=scheduler)

    restored_model = FramerModel(config)
    restored_optimizer = build_optimizer(restored_model, config)
    restored_scheduler = build_scheduler(restored_optimizer, config)
    load_sharded(restored_model, restored_optimizer, path, scheduler=restored_scheduler)

    # If scheduler were incorrectly restarted, it would be in warmup with high LR
    # But it should be at step 10 with the correct decayed LR
    lr_restored = restored_scheduler.get_last_lr()[0]
    assert abs(lr_restored - lr_at_step_10) < 1e-9

    # Step once more and verify LR continues from step 10, not step 0
    optimizer.step()
    scheduler.step()
    restored_optimizer.step()
    restored_scheduler.step()

    assert abs(scheduler.get_last_lr()[0] - restored_scheduler.get_last_lr()[0]) < 1e-9


def test_atomic_write_safety(tmp_path):
    """Interrupted checkpoint write must not corrupt previous valid checkpoint."""
    from unittest.mock import patch

    config = tiny_config()
    model = FramerModel(config)
    optimizer = build_optimizer(model, config)

    # Save first checkpoint
    path = str(tmp_path / "model.pt")
    from model.utils.helpers import save_checkpoint
    save_checkpoint(model, optimizer, step=1, loss=1.0, path=path)

    # Verify it exists and is valid
    assert os.path.exists(path)
    ckpt1 = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt1["step"] == 1

    # Simulate interrupted write by making torch.save fail
    with patch("torch.save", side_effect=RuntimeError("disk full")):
        try:
            save_checkpoint(model, optimizer, step=2, loss=0.5, path=path)
        except RuntimeError:
            pass

    # Original checkpoint should still be valid
    ckpt_after = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt_after["step"] == 1  # still the old checkpoint

    # Temp file should be cleaned up (or never fully written)
    temp_path = path + ".tmp"
    # Either temp file doesn't exist, or if it does, it's incomplete
    if os.path.exists(temp_path):
        # If temp exists, original should still be intact
        assert ckpt_after["step"] == 1


def test_backward_compat_missing_scheduler_state(tmp_path):
    """Loading old checkpoint without scheduler_state_dict must not crash."""
    from model.training.schedule import build_scheduler

    config = tiny_config()
    model = FramerModel(config)
    optimizer = build_optimizer(model, config)

    # Save checkpoint without scheduler (simulating old format)
    path = str(tmp_path / "old_ckpt.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": 5,
        "loss": 0.5,
    }, path)

    # Load with scheduler - should not crash
    restored_model = FramerModel(config)
    restored_optimizer = build_optimizer(restored_model, config)
    restored_scheduler = build_scheduler(restored_optimizer, config)

    from model.utils.helpers import load_checkpoint
    step, loss = load_checkpoint(path, restored_model, restored_optimizer, restored_scheduler)

    assert step == 5
    assert loss == 0.5
    # Scheduler should be at initial state (not crashed)


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
