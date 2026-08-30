"""Single-entry language-model training loop.

Handles precision (bf16/fp16/fp32 autocast + optional GradScaler), warmup-cosine
LR, gradient accumulation, gradient clipping, activation checkpointing (via the
model's config flag), MoE auxiliary loss (already folded into the model's total
loss), checkpointing, and multi-rank logging. Works single-device by default and
under torchrun when the model has been FSDP-wrapped by the caller.
"""

import os
import time

import torch
import torch.nn as nn

from .distributed import get_world_size, is_main_process
from .optim import build_optimizer
from .precision import autocast_context, resolve_precision
from .schedule import build_scheduler


def _infinite(loader):
    """Yield batches forever, restarting map-style loaders each epoch.

    Raises if an entire epoch produces no batches — otherwise an empty loader
    (e.g. a packed shard corpus smaller than the sequence length) would spin
    forever instead of failing clearly.
    """
    epoch = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError(
                "Training data loader produced no batches. If using --shard-dir, the "
                "corpus likely has fewer tokens than the sequence length — add more "
                "data, lower max_seq_len, or use a larger --shard-tokens corpus."
            )
        epoch += 1


def train_language_model(
    config,
    model,
    dataloader,
    device,
    output_dir,
    start_step: int = 0,
    log_interval: int = 10,
    save_interval: int = 500,
    logger=None,
    optimizer=None,
    scheduler=None,
):
    """Train the LM head of ``model`` to ``config.max_steps``. Returns final step.

    Accepts optional pre-created optimizer and scheduler for resume scenarios.
    """
    log = (logger.info if logger else print) if is_main_process() else (lambda *a, **k: None)

    if optimizer is None:
        optimizer = build_optimizer(model, config)
    if scheduler is None:
        scheduler = build_scheduler(optimizer, config)

    autocast_dtype, use_scaler = resolve_precision(device, config.precision, config.mixed_precision)
    scaler = torch.amp.GradScaler(device.type) if use_scaler else None
    log(f"Precision: autocast={autocast_dtype}, grad_scaler={use_scaler}, "
        f"grad_ckpt={config.use_gradient_checkpointing}, world_size={get_world_size()}")

    model.train()
    accum = max(1, config.gradient_accumulation_steps)
    grad_clip = config.grad_clip
    step = start_step
    running = 0.0
    t0 = time.time()
    batches = _infinite(dataloader)

    optimizer.zero_grad(set_to_none=True)
    while step < config.max_steps:
        for _micro in range(accum):
            batch = next(batches)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast_context(device, autocast_dtype):
                results = model(input_ids=input_ids, labels=labels)
                loss = results["loss"] / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_interval == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            aux = results.get("aux_loss")
            aux_str = f" | aux {float(aux):.4f}" if aux is not None else ""
            ctc = results.get("ctc_loss")
            ctc_str = f" | ctc {float(ctc):.4f}" if ctc is not None else ""
            log(f"Step {step}/{config.max_steps} | loss {running / log_interval:.4f}{aux_str}{ctc_str} "
                f"| lr {lr:.2e} | {step / max(elapsed, 1e-9):.1f} it/s")
            running = 0.0

        if step % save_interval == 0 and is_main_process():
            _save(model, optimizer, scheduler, config, step, output_dir, f"checkpoint_{step}.pt")
            log(f"Checkpoint saved at step {step}")

    if is_main_process():
        _save(model, optimizer, scheduler, config, step, output_dir, "model_final.pt")
        log(f"Training complete. Final model saved ({step} steps).")
    return step


def _save(model, optimizer, scheduler, config, step, output_dir, filename):
    """Save a checkpoint with full training state.

    Under FSDP2 each rank holds only a slice of every parameter, so writing
    ``state_dict()`` straight to one file used to persist a single rank's shard
    under a name that claimed to be the whole model. Distributed runs now gather
    the full state dict on rank 0 instead. For a model too large to gather, call
    ``save_sharded`` directly.
    """
    from dataclasses import asdict

    from .checkpoint import gather_full_state_dict

    os.makedirs(output_dir, exist_ok=True)
    state = gather_full_state_dict(model, cpu_offload=True)
    if not state:  # non-zero rank under distributed; rank 0 does the writing
        return

    checkpoint = {
        "model_state_dict": state,
        "config": asdict(config),
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Atomic write
    path = os.path.join(output_dir, filename)
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)
