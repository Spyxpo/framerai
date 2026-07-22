"""Learning-rate schedule: linear warmup then cosine decay to a floor."""

import math

import torch


def lr_at_step(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    """Warmup-cosine learning rate multiplier resolved to an absolute LR."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def build_scheduler(optimizer, config):
    """LambdaLR implementing warmup + cosine, consuming config.warmup_steps."""
    base_lr = config.learning_rate
    min_lr = config.min_learning_rate

    def lr_lambda(step):
        # LambdaLR multiplies the optimizer base LR, so return a ratio.
        return lr_at_step(step, config.warmup_steps, config.max_steps, base_lr, min_lr) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
