"""Distillation training: few-step latent diffusion student from many-step teacher.

Trains a student latent image generator that produces the teacher's guided
field in single-digit steps without needing classifier-free guidance at
inference time. The teacher walks trajectory segments with CFG on; the student
learns to cover the same segments in one unguided step.

References the FlowDistiller objective from model.modules.flow, which is
complete and tested. This training loop wires it into the existing checkpoint,
optimizer, scheduler, and precision infrastructure.
"""

import os
import random
import time

import torch
import torch.nn as nn

from .distributed import get_world_size, is_main_process
from .precision import autocast_context, resolve_precision
from .schedule import build_scheduler


def _infinite(loader):
    """Yield batches forever, restarting each epoch."""
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
            raise RuntimeError("Distillation data loader produced no batches.")
        epoch += 1


def train_distill(
    config,
    teacher_model,
    student_model,
    dataloader,
    device,
    output_dir,
    distiller,
    start_step: int = 0,
    log_interval: int = 10,
    save_interval: int = 500,
    logger=None,
    optimizer=None,
    scheduler=None,
    resume_checkpoint: str = None,
):
    """Train student latent image generator via distillation from teacher.

    The teacher walks trajectory segments with CFG guidance under no_grad. The
    student learns to match those endpoints in a single unguided step. Only the
    student's diffusion denoiser and null_context are trained; the VAE, text
    backbone, and all teacher parameters remain frozen.

    Args:
        config: FramerConfig with flow_distilled=True for student
        teacher_model: Frozen FramerModel with trained latent diffusion
        student_model: Trainable FramerModel initialized from teacher
        dataloader: Yields {"target_images": Tensor, "input_ids": Tensor}
        device: torch device
        output_dir: Where to save checkpoints
        distiller: FlowDistiller instance with teacher_substeps configured
        start_step: Resume from this step
        log_interval: Log every N steps
        save_interval: Save checkpoint every N steps
        logger: Logger object (optional)
        optimizer: Pre-created optimizer (optional, for resume)
        scheduler: Pre-created scheduler (optional, for resume)
        resume_checkpoint: Path to checkpoint for resuming optimizer/scheduler state

    Returns:
        Final training step number
    """
    log = (logger.info if logger else print) if is_main_process() else (lambda *a, **k: None)

    if optimizer is None:
        # Only optimize student's diffusion denoiser and null_context
        trainable_params = (
            list(student_model.diffusion.denoiser.parameters())
            + [student_model.diffusion.null_context]
        )
        # Build optimizer manually since we're not optimizing all model parameters
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
    if scheduler is None:
        scheduler = build_scheduler(optimizer, config)

    # Load optimizer/scheduler state from checkpoint when resuming
    if resume_checkpoint and start_step > 0:
        from model.utils import load_checkpoint as load_ckpt
        load_ckpt(resume_checkpoint, model=None, optimizer=optimizer, scheduler=scheduler)
        log(f"Restored optimizer and scheduler state from {resume_checkpoint} at step {start_step}")

    autocast_dtype, use_scaler = resolve_precision(device, config.precision, config.mixed_precision)
    scaler = torch.amp.GradScaler(device.type) if use_scaler else None

    student_steps = config.flow_distilled_steps
    segments = distiller.segments(student_steps, device=device)

    log(f"Distillation Training: teacher_substeps={distiller.teacher_substeps}, "
        f"student_steps={student_steps}, autocast={autocast_dtype}, "
        f"cfg_scale={config.cfg_scale}, world_size={get_world_size()}")

    # Teacher stays in eval, student trains
    teacher_model.eval()
    student_model.train()

    # Freeze student's VAE and text backbone - only diffusion denoiser trains
    for param in student_model.diffusion.vae.parameters():
        param.requires_grad = False
    # Freeze the text model (token embeddings, transformer layers, lm_head)
    for param in student_model.token_embed.parameters():
        param.requires_grad = False
    for param in student_model.layers.parameters():
        param.requires_grad = False
    for param in student_model.norm.parameters():
        param.requires_grad = False
    for param in student_model.lm_head.parameters():
        param.requires_grad = False

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
            images = batch["target_images"].to(device, non_blocking=True)
            captions = batch["input_ids"].to(device, non_blocking=True)

            with autocast_context(device, autocast_dtype):
                # Encode images to latent space (frozen VAE)
                with torch.no_grad():
                    latents = student_model.diffusion.vae.encode_to_latent(images)

                    # Encode captions to text context (frozen backbone)
                    # Use the full forward_lm rather than just backbone to handle attention masks
                    text_output = student_model.forward_lm(captions)
                    text_context = text_output["hidden"].mean(1, keepdim=True)

                # Sample noise and random trajectory segment
                noise = torch.randn_like(latents)
                seg_idx = random.randint(0, len(segments) - 2)
                t_start = segments[seg_idx]
                t_end = segments[seg_idx + 1]

                # Interpolate to segment start
                from model.modules.flow import RectifiedFlow
                t_start_scalar = float(t_start)
                t_start_batch = torch.full((latents.shape[0],), t_start_scalar, device=device)
                x_start = RectifiedFlow.interpolate(noise, latents, t_start_batch)

                # Teacher walks segment with CFG guidance (frozen, no_grad)
                with torch.no_grad():
                    def teacher_velocity_fn(x, t, ctx):
                        return teacher_model.diffusion.denoiser(x, t, ctx)

                    teacher_endpoint = distiller.teacher_endpoint(
                        teacher_velocity_fn,
                        x_start,
                        t_start,
                        t_end,
                        context=text_context,
                        null_context=teacher_model.diffusion.null_context,
                        cfg_scale=config.cfg_scale,
                    )

                # Student predicts velocity at segment start (trainable, no CFG)
                student_velocity = student_model.diffusion.denoiser(
                    x_start, t_start_batch, text_context
                )

                # Distillation loss: student's one-step endpoint vs teacher's guided endpoint
                loss = distiller.loss(student_velocity, x_start, t_start, t_end, teacher_endpoint)
                loss = loss / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(student_model.diffusion.denoiser.parameters()) + [student_model.diffusion.null_context],
                grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(
                list(student_model.diffusion.denoiser.parameters()) + [student_model.diffusion.null_context],
                grad_clip
            )
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_interval == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            log(f"Distill Step {step}/{config.max_steps} | loss {running_loss / log_interval:.4f} "
                f"| lr {lr:.2e} | {step / max(elapsed, 1e-9):.1f} it/s")
            running_loss = 0.0

        if step % save_interval == 0 and is_main_process():
            _save(student_model, optimizer, scheduler, config, step, output_dir, f"checkpoint_distill_{step}.pt")
            log(f"Distillation checkpoint saved at step {step}")

    if is_main_process():
        _save(student_model, optimizer, scheduler, config, step, output_dir, "model_distill_final.pt")
        log(f"Distillation complete. Final student model saved ({step} steps).")

    return step


def _save(model, optimizer, scheduler, config, step, output_dir, filename):
    """Save a checkpoint with full training state.

    Reuses the checkpoint gathering logic from trainer.py to handle FSDP.
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
