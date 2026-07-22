"""Training infrastructure for FramerAI: schedules, precision, distribution."""

from .distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    maybe_wrap_fsdp,
)
from .optim import build_optimizer
from .precision import autocast_context, resolve_precision
from .schedule import build_scheduler, lr_at_step
from .trainer import train_language_model

__all__ = [
    "build_scheduler",
    "lr_at_step",
    "resolve_precision",
    "autocast_context",
    "build_optimizer",
    "init_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "maybe_wrap_fsdp",
    "train_language_model",
]
