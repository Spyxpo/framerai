"""Centralized RNG seeding for reproducible training.

Call ``apply_seed`` once before model initialization and before the training
loop begins. Seeding here covers Python's built-in ``random``, NumPy, and
PyTorch (CPU and, when available, CUDA). It does *not* enable
``torch.use_deterministic_algorithms``: that flag rejects a number of CUDA
kernels used in normal training (e.g. certain atomics in backward passes) and
would degrade throughput without meaningfully improving reproducibility for
most runs.

Remaining sources of non-determinism that seeding cannot eliminate:
- CUDA atomic operations in backward passes (cuBLAS, cuDNN).
- Multi-GPU all-reduce ordering under NCCL.
- Non-deterministic DataLoader worker interleaving (use ``num_workers=0`` or
  a fixed worker seed if full bit-exact reproducibility is required).
"""

import random

import torch


def apply_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs.

    Args:
        seed: Integer seed value. The default in FramerConfig is 42.
    """
    random.seed(seed)

    try:
        import numpy as np  # noqa: PLC0415 — optional at import time for clarity

        np.random.seed(seed)
    except ImportError:  # pragma: no cover — numpy is in requirements.txt
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
