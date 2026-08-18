"""Sharded checkpoint save/load and full state-dict gather.

The single-file ``torch.save(model.state_dict())`` path cannot express a model
larger than one host: ``framer-2t-a49b`` is 3.6 TiB in bf16 and roughly 29 TiB
with optimizer state. Worse, under FSDP2 the naive path silently wrote *each
rank's shard* to the same filename, producing a checkpoint that looked fine and
could not be reloaded.

This module routes both through ``torch.distributed.checkpoint``, which writes
one file per rank into a directory and reassembles them on load, and provides a
rank-0 gather for when a single portable file really is wanted (export, upload,
sharing) and the model is small enough to fit.

Everything degrades cleanly to the plain single-file path when distributed is
unavailable or the world size is one, so single-device runs and the test suite
are unaffected.
"""

import json
import os
from dataclasses import asdict

import torch
import torch.distributed as dist

CONFIG_FILENAME = "config.json"
METADATA_FILENAME = "checkpoint_meta.json"


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _unwrap(model):
    """The underlying module, past DDP's `.module` wrapper."""
    return model.module if hasattr(model, "module") else model


def _write_sidecars(path: str, config, step: int, extra: dict = None):
    """Config and step live outside the sharded payload so they stay readable."""
    if config is not None:
        config_dict = config if isinstance(config, dict) else asdict(config)
        with open(os.path.join(path, CONFIG_FILENAME), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    meta = {"step": step, "sharded": True, **(extra or {})}
    with open(os.path.join(path, METADATA_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)


def save_sharded(model, optimizer, path: str, step: int = 0, config=None, scheduler=None) -> str:
    """Save model (and optimizer + scheduler) state as a sharded checkpoint directory.

    Every rank participates and writes its own shard. Falls back to a single
    ``.pt`` file named ``<path>/model.pt`` when not running distributed, so the
    same call site works in both cases.

    Returns the directory written.
    """
    os.makedirs(path, exist_ok=True)

    if not _is_distributed():
        core = _unwrap(model)
        payload = {"model_state_dict": core.state_dict(), "step": step}
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()

        # Atomic write
        single_file = os.path.join(path, "model.pt")
        temp_file = single_file + ".tmp"
        torch.save(payload, temp_file)
        os.replace(temp_file, single_file)

        _write_sidecars(path, config, step, {"sharded": False})
        return path

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optim_state = get_state_dict(
        _unwrap(model), [] if optimizer is None else [optimizer]
    )
    state = {"model": model_state}
    if optimizer is not None:
        state["optim"] = optim_state
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    dcp.save(state, checkpoint_id=path)
    if dist.get_rank() == 0:
        _write_sidecars(path, config, step)
    dist.barrier()
    return path


def load_sharded(model, optimizer, path: str, scheduler=None) -> int:
    """Load a checkpoint written by :func:`save_sharded`, in place.

    Resharding is handled by the checkpoint layer, so a run saved on N ranks can
    be resumed on M. Returns the step recorded at save time.

    Backward compatible: older checkpoints without scheduler_state_dict are
    supported.
    """
    meta_path = os.path.join(path, METADATA_FILENAME)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    single_file = os.path.join(path, "model.pt")
    if not _is_distributed() or not meta.get("sharded", True):
        if not os.path.exists(single_file):
            raise FileNotFoundError(
                f"{path} holds a sharded checkpoint; loading it needs a "
                "distributed process group. Use gather_full_state_dict on the "
                "training run to produce a single-file export instead."
            )
        payload = torch.load(single_file, map_location="cpu", weights_only=False)
        _unwrap(model).load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        return payload.get("step", meta.get("step", 0))

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    core = _unwrap(model)
    optimizers = [] if optimizer is None else [optimizer]
    model_state, optim_state = get_state_dict(core, optimizers)
    state = {"model": model_state}
    if optimizer is not None:
        state["optim"] = optim_state
    if scheduler is not None:
        state["scheduler"] = {}

    dcp.load(state, checkpoint_id=path)
    set_state_dict(
        core,
        optimizers,
        model_state_dict=state["model"],
        optim_state_dict=state.get("optim"),
    )
    if scheduler is not None and "scheduler" in state and state["scheduler"]:
        scheduler.load_state_dict(state["scheduler"])

    return meta.get("step", 0)


def gather_full_state_dict(model, cpu_offload: bool = True) -> dict:
    """Collect a complete, unsharded state dict on rank 0.

    Under FSDP2 each rank holds only its slice of every parameter, so
    ``state_dict()`` returns a shard. This gathers the full tensors, offloaded to
    CPU by default because the point is usually to write them to disk. Returns an
    empty dict on non-zero ranks.

    Only usable when the whole model fits in one host's memory. It does not for
    the largest presets, which is what :func:`save_sharded` is for.
    """
    core = _unwrap(model)
    if not _is_distributed():
        return core.state_dict()

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload)
    return get_model_state_dict(core, options=options)


def save_full(model, path: str, step: int = 0, config=None) -> str:
    """Gather and write a single portable checkpoint file from rank 0.

    The counterpart to :func:`save_sharded` for models that fit on one host.
    """
    state = gather_full_state_dict(model, cpu_offload=True)
    if _is_distributed() and dist.get_rank() != 0:
        dist.barrier()
        return path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_state_dict": state, "step": step}
    if config is not None:
        payload["config"] = config if isinstance(config, dict) else asdict(config)
    torch.save(payload, path)
    if _is_distributed():
        dist.barrier()
    return path
