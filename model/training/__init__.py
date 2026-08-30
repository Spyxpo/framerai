"""Training infrastructure for FramerAI: schedules, precision, distribution."""

from .checkpoint import gather_full_state_dict, load_sharded, save_full, save_sharded
from .distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    maybe_wrap_fsdp,
)
from .dpo import compute_dpo_loss, get_batch_logps, train_dpo
from .expert_parallel import (
    ExpertParallelPlan,
    build_device_mesh,
    plan_from_environment,
    shard_experts,
    shard_model_experts,
)
from .optim import build_optimizer
from .precision import autocast_context, resolve_precision
from .schedule import build_scheduler, lr_at_step
from .sft import train_sft
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
    "train_sft",
    "train_dpo",
    "get_batch_logps",
    "compute_dpo_loss",
    "save_sharded",
    "load_sharded",
    "save_full",
    "gather_full_state_dict",
    "ExpertParallelPlan",
    "plan_from_environment",
    "shard_experts",
    "shard_model_experts",
    "build_device_mesh",
]
