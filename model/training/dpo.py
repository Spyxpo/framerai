"""Direct Preference Optimization (DPO) training loop."""

import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_parameter_counts
from .distributed import get_world_size, is_main_process
from .optim import build_optimizer
from .precision import autocast_context, resolve_precision
from .schedule import build_scheduler


def _infinite(loader):
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
            raise RuntimeError("DPO data loader produced no batches.")
        epoch += 1


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute per-sequence log probabilities over target completion tokens.

    logits: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len) - pre-shifted next-token targets
    """
    if logits.shape[1] != labels.shape[1]:
        raise ValueError(f"Logits seq len ({logits.shape[1]}) != labels seq len ({labels.shape[1]})")

    log_probs = F.log_softmax(logits, dim=-1)
    mask = (labels != ignore_index)
    clamped_labels = labels.masked_fill(~mask, 0)

    per_token_logps = torch.gather(
        log_probs, dim=2, index=clamped_labels.unsqueeze(-1)
    ).squeeze(-1)

    return (per_token_logps * mask).sum(dim=-1)


def compute_dpo_loss(
    pi_chosen_logps: torch.Tensor,
    pi_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss and reward implicit scale metrics."""
    pi_logratios = pi_chosen_logps - pi_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    chosen_rewards = beta * (pi_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (pi_rejected_logps - ref_rejected_logps).detach()

    return loss, chosen_rewards, rejected_rewards


def train_dpo(
    config: Any,
    policy_model: Any,
    ref_model: Any,
    dataloader: Any,
    device: Any,
    output_dir: str,
    beta: float = 0.1,
    start_step: int = 0,
    log_interval: int = 10,
    save_interval: int = 500,
    logger: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
) -> int:
    """Train policy_model using DPO against ref_model."""
    log = (logger.info if logger else print) if is_main_process() else (lambda *a, **k: None)

    # Freeze reference model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    if optimizer is None:
        optimizer = build_optimizer(policy_model, config)
    if scheduler is None:
        scheduler = build_scheduler(optimizer, config)

    autocast_dtype, use_scaler = resolve_precision(device, config.precision, config.mixed_precision)
    scaler = torch.amp.GradScaler(device.type) if use_scaler else None

    counts = get_parameter_counts(policy_model)
    log(f"Model parameters: total={counts['total']:,} | trainable={counts['trainable']:,}")
    log(f"Starting DPO Training: beta={beta}, autocast={autocast_dtype}, world_size={get_world_size()}")

    policy_model.train()
    accum = max(1, config.gradient_accumulation_steps)
    grad_clip = config.grad_clip
    step = start_step
    running_loss = 0.0
    t0 = time.time()
    batches = _infinite(dataloader)

    optimizer.zero_grad(set_to_none=True)
    while step < config.max_steps:
        for _micro in range(accum):
            batch = next(batches)

            chosen_ids = batch["chosen_input_ids"].to(device, non_blocking=True)
            chosen_labels = batch["chosen_labels"].to(device, non_blocking=True)
            rejected_ids = batch["rejected_input_ids"].to(device, non_blocking=True)
            rejected_labels = batch["rejected_labels"].to(device, non_blocking=True)

            with autocast_context(device, autocast_dtype):
                # Policy forward passes
                pi_chosen_logits = policy_model(input_ids=chosen_ids)["logits"]
                pi_rejected_logits = policy_model(input_ids=rejected_ids)["logits"]

                pi_chosen_logps = get_batch_logps(pi_chosen_logits, chosen_labels)
                pi_rejected_logps = get_batch_logps(pi_rejected_logits, rejected_labels)

                # Reference forward passes
                with torch.no_grad():
                    ref_chosen_logits = ref_model(input_ids=chosen_ids)["logits"]
                    ref_rejected_logits = ref_model(input_ids=rejected_ids)["logits"]

                    ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels)
                    ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels)

                loss, _, _ = compute_dpo_loss(
                    pi_chosen_logps, pi_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                    beta=beta,
                )
                loss = loss / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_interval == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            log(f"DPO Step {step}/{config.max_steps} | loss {running_loss / log_interval:.4f} "
                f"| lr {lr:.2e} | {step / max(elapsed, 1e-9):.1f} it/s")
            running_loss = 0.0

        if step % save_interval == 0 and is_main_process():
            from .trainer import _save
            _save(policy_model, optimizer, scheduler, config, step, output_dir, f"checkpoint_dpo_{step}.pt")
            log(f"DPO Checkpoint saved at step {step}")

    if is_main_process():
        from .trainer import _save
        _save(policy_model, optimizer, scheduler, config, step, output_dir, "model_dpo_final.pt")
        log(f"DPO Training complete. Final model saved ({step} steps).")

    return step
