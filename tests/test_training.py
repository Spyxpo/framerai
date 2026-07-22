"""Training-infra tests: LR schedule, precision resolution, optimizer groups."""

import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.training.optim import build_optimizer
from model.training.precision import resolve_precision
from model.training.schedule import lr_at_step


def test_warmup_then_cosine_decay():
    warmup, max_steps, base, floor = 10, 100, 1e-3, 1e-5
    # Warmup ramps up linearly from ~0 to base.
    assert lr_at_step(0, warmup, max_steps, base, floor) < base
    assert lr_at_step(warmup, warmup, max_steps, base, floor) == base
    # Then decays monotonically toward the floor.
    mid = lr_at_step(55, warmup, max_steps, base, floor)
    late = lr_at_step(90, warmup, max_steps, base, floor)
    assert base > mid > late >= floor
    assert lr_at_step(max_steps, warmup, max_steps, base, floor) == floor


def test_precision_resolution_cpu_is_fp32_no_scaler():
    # CPU autocast is emulated and slow, so CPU always runs fp32 (no scaler).
    dtype, scaler = resolve_precision(torch.device("cpu"), "bf16")
    assert dtype is None and scaler is False
    dtype, scaler = resolve_precision(torch.device("cpu"), "fp32")
    assert dtype is None and scaler is False


def test_optimizer_splits_decay_groups():
    cfg = tiny_config()
    model = FramerModel(cfg)
    opt = build_optimizer(model, cfg)
    decay_wd = opt.param_groups[0]["weight_decay"]
    nodecay_wd = opt.param_groups[1]["weight_decay"]
    assert decay_wd == cfg.weight_decay and nodecay_wd == 0.0
    assert len(opt.param_groups[0]["params"]) > 0
    assert len(opt.param_groups[1]["params"]) > 0  # RMSNorm gains
