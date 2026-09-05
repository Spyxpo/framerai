"""Mixed-precision helpers: pick bf16/fp16/fp32 per device, autocast context."""

import contextlib

import torch


def resolve_precision(device: torch.device, precision: str, mixed_precision: bool = True):
    """Resolve the autocast dtype and whether a GradScaler is needed.

    Returns ``(autocast_dtype_or_None, use_grad_scaler)``.

    - bf16 is preferred on CUDA Ampere+ (no scaler needed, more stable at scale).
    - fp16 falls back on older CUDA and needs a GradScaler.
    - bf16 autocast is used on MPS/CPU where supported; otherwise fp32.
    """
    # Validate precision up front to avoid silent fallback to fp32
    from ..configs.model_config import PRECISION_TYPES
    if precision not in PRECISION_TYPES:
        raise ValueError(f"Invalid precision '{precision}'. Must be one of {', '.join(PRECISION_TYPES)}")

    if not mixed_precision or precision == "fp32":
        return None, False

    if device.type == "cuda":
        if precision == "bf16":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, False
            return torch.float16, True  # older GPU: fall back to fp16
        if precision == "fp16":
            return torch.float16, True
    elif device.type == "mps":
        # bf16 autocast works on Apple Silicon; fp16 off-CUDA is not useful.
        return torch.bfloat16, False
    elif device.type == "cpu":
        # CPU bf16/fp16 autocast is emulated and dramatically slower for
        # training; run fp32 on CPU regardless of the requested precision.
        return None, False

    return None, False


def autocast_context(device: torch.device, dtype):
    """An autocast context manager, or a no-op when dtype is None."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)
