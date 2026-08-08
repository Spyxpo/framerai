"""Distributed training helpers (torch-native FSDP2), guarded for single-device.

Single-GPU / laptop is the default and needs none of this. When launched under
``torchrun`` with WORLD_SIZE > 1 on CUDA, ``init_distributed`` sets up the
process group and ``maybe_wrap_fsdp`` sharded-wraps the transformer blocks with
FSDP2 (``fully_shard``) so parameters, gradients, and optimizer state are split
across ranks. Tensor / expert / pipeline parallelism (needed to actually train
the trillion-parameter preset) are cluster-scale and left as extension points.
"""

import os

import torch
import torch.distributed as dist


def _env_int(name: str, default: int = 1) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _env_int("WORLD_SIZE", 1)


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env_int("RANK", 0)


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(device: torch.device):
    """Initialize the process group when running multi-rank on CUDA.

    Returns True if distributed training is active, else False (single device).
    """
    if get_world_size() <= 1:
        return False
    if device.type != "cuda":
        # NCCL/FSDP require CUDA; MPS/CPU stay single-process.
        return False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        local_rank = _env_int("LOCAL_RANK", 0)
        torch.cuda.set_device(local_rank)
    return True


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_wrap_fsdp(model, config, device: torch.device, mesh=None):
    """FSDP2-shard the transformer blocks when distributed; else return as-is.

    ``mesh`` is an optional 2D ``(dp, ep)`` device mesh. Expert weights are
    already split across the ``ep`` dimension by expert parallelism, so FSDP
    must shard over ``dp`` only - sharding them again would divide them twice
    and leave each rank holding a fragment of a fragment.
    """
    if get_world_size() <= 1 or device.type != "cuda":
        return model

    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError:
        # Older PyTorch without FSDP2: fall back to the model unchanged rather
        # than crash. (Legacy FSDP1 wrapping could be added here if needed.)
        if is_main_process():
            print("[dist] FSDP2 unavailable; running without parameter sharding.")
        return model

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(config.precision, torch.float32)
    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)

    # Shard each transformer block, then the root module. With a 2D mesh the
    # data-parallel dimension is the one to shard over.
    shard_kwargs = {"mp_policy": mp_policy}
    if mesh is not None:
        shard_kwargs["mesh"] = mesh["dp"]

    for block in model.layers:
        fully_shard(block, **shard_kwargs)
    fully_shard(model, **shard_kwargs)
    return model
