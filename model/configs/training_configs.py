"""Ready-to-use, reproducible training configurations for the four main presets.

Each entry is a dict of :class:`FramerConfig` field overrides covering *training
hyperparameters only*. The model architecture fields (``d_model``, ``n_layers``,
``n_heads``, etc.) are already defined in ``presets.py`` and are not repeated here.

Use :func:`get_training_config` to retrieve the config for a preset by name or
legacy alias, then merge it into a :class:`FramerConfig`:

    from model.configs import FramerConfig, get_training_config

    tc = get_training_config("framer-small")
    config = FramerConfig.from_preset("framer-small", **tc)

To use a ready-to-use configuration from the CLI, pass the overrides explicitly.
``build.py`` exposes a flag for every training field in the table below, so the
command is self-contained and fully reproducible:

    python build.py --mode train --preset framer-small --seed 42 \\
        --precision bf16 --max-steps 20000 --warmup-steps 1000 --grad-accum 4

See the "Reproducible training" section in GUIDE.md for the full per-preset
commands.

Value rationale
---------------
All values are grounded in existing repository sources:

* ``batch_size`` — FramerConfig default is 8 (model_config.py). Kept at 8 for
  all sizes; effective batch size is scaled through ``gradient_accumulation_steps``
  rather than a larger per-step batch, which keeps memory requirements stable.

* ``gradient_accumulation_steps`` — FramerConfig default is 4 (effective batch 32).
  Tiny reduces to 1 (no accumulation needed for a smoke-test). Small keeps 4.
  Medium doubles to 8 (effective batch 64). Large uses 16 (effective batch 128)
  paired with gradient checkpointing to trade compute for memory.

* ``learning_rate`` — FramerConfig default is 3e-4. Kept across all sizes; the
  warmup-cosine schedule in trainer.py handles the ramp regardless of model size.

* ``warmup_steps`` — FramerConfig default is 2000. Scaled proportionally to
  max_steps: ~5% of the run. Tiny (10k steps) → 500; small (20k) → 1000;
  medium/large (100k) → 2000.

* ``max_steps`` — GUIDE.md examples: ``--max-steps 10000`` for tiny smoke-test,
  ``--max-steps 20000`` for small. FramerConfig default (100000) for medium/large.

* ``precision`` — FramerConfig default is "bf16". Tiny runs on CPU where the
  precision resolver already forces fp32 (trainer.py resolve_precision); "fp32"
  is documented here to match that behaviour. Small/medium/large use "bf16".

* ``use_gradient_checkpointing`` — FramerConfig default is False. Enabled only for
  framer-large because it is the only preset the build.py comment documents as
  benefiting from ``--grad-checkpointing`` to save memory.

* ``grad_clip`` — FramerConfig default is 1.0. Unchanged across all sizes.

* ``seed`` — 42 across all sizes (FramerConfig default). Override with --seed.

Hardware notes are qualitative recommendations derived from presets.py comments:
  "framer-tiny is CPU smoke tests"
  "small/mid presets can train on a single consumer GPU or CPU"
No exact VRAM figures are stated; those are not established in the repository.
"""

from .presets import _ALIASES  # noqa: PLC2701 — internal use within configs package

# ---------------------------------------------------------------------------
# Four ready-to-use training configurations
# ---------------------------------------------------------------------------

# Each dict contains only training-hyperparameter fields that exist on
# FramerConfig. Architecture fields are intentionally absent.
_TRAINING_CONFIGS: dict[str, dict] = {
    "framer-tiny": dict(
        # CPU smoke-test scale. Runs on CPU; no GPU required.
        # Effective batch = batch_size × gradient_accumulation_steps = 8 × 1 = 8.
        seed=42,
        batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        min_learning_rate=1e-6,
        warmup_steps=500,
        max_steps=10_000,
        precision="fp32",  # CPU runner; resolve_precision forces fp32 on CPU
        use_gradient_checkpointing=False,
        grad_clip=1.0,
        weight_decay=0.01,
        # Hardware: CPU or any GPU. No GPU required.
    ),
    "framer-small": dict(
        # Small scale. Single consumer GPU or CPU.
        # Effective batch = 8 × 4 = 32.
        seed=42,
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        min_learning_rate=1e-6,
        warmup_steps=1_000,
        max_steps=20_000,
        precision="bf16",
        use_gradient_checkpointing=False,
        grad_clip=1.0,
        weight_decay=0.01,
        # Hardware: single consumer GPU recommended; CPU is supported but slow.
    ),
    "framer-medium": dict(
        # Medium scale. Single GPU with sufficient VRAM.
        # Effective batch = 8 × 8 = 64.
        seed=42,
        batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        min_learning_rate=1e-6,
        warmup_steps=2_000,
        max_steps=100_000,
        precision="bf16",
        use_gradient_checkpointing=False,
        grad_clip=1.0,
        weight_decay=0.01,
        # Hardware: single GPU with sufficient VRAM recommended.
    ),
    "framer-large": dict(
        # Large scale. High-memory GPU or multi-GPU setup recommended.
        # Effective batch = 8 × 16 = 128. Gradient checkpointing enabled to
        # reduce activation memory at the cost of extra recomputation.
        seed=42,
        batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=3e-4,
        min_learning_rate=1e-6,
        warmup_steps=2_000,
        max_steps=100_000,
        precision="bf16",
        use_gradient_checkpointing=True,
        grad_clip=1.0,
        weight_decay=0.01,
        # Hardware: high-memory GPU or multi-GPU recommended.
    ),
}

# Public name — exported from model/configs/__init__.py.
TRAINING_CONFIGS: dict[str, dict] = _TRAINING_CONFIGS

# Keys that every training config must contain. Used by tests and by
# get_training_config to validate the returned dict.
REQUIRED_TRAINING_KEYS: tuple[str, ...] = (
    "seed",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "warmup_steps",
    "max_steps",
    "precision",
    "use_gradient_checkpointing",
    "grad_clip",
)


def get_training_config(name: str) -> dict:
    """Return the training configuration dict for a preset name or legacy alias.

    Args:
        name: A canonical preset name (e.g. ``"framer-small"``) or a legacy
              size alias (``"tiny"``, ``"small"``, ``"medium"``, ``"large"``).

    Returns:
        A dict of :class:`FramerConfig` training-field overrides.

    Raises:
        KeyError: If the name does not resolve to a known training configuration.

    Example::

        from model.configs import FramerConfig, get_training_config

        tc = get_training_config("small")           # alias works
        config = FramerConfig.from_preset("framer-small", **tc)
    """
    resolved = _ALIASES.get(name, name)
    if resolved not in _TRAINING_CONFIGS:
        available = ", ".join(sorted(_TRAINING_CONFIGS))
        aliases = ", ".join(sorted(_ALIASES))
        raise KeyError(
            f"No training config for '{name}' (resolved: '{resolved}'). "
            f"Available: {available} (aliases: {aliases})"
        )
    return dict(_TRAINING_CONFIGS[resolved])  # return a copy so callers can mutate safely
