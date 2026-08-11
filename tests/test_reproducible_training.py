"""Tests for Issue #5: reproducible training configurations.

All tests run on CPU with tiny synthetic models and complete in seconds.
No large model is instantiated; framer-small/medium/large are only exercised
through config construction and dict lookup (no forward pass).
"""

import torch

from build import _build_config_from_args, _make_parser
from conftest import tiny_config
from model.configs import (
    REQUIRED_TRAINING_KEYS,
    TRAINING_CONFIGS,
    FramerConfig,
    get_training_config,
)
from model.framer import FramerModel
from model.utils import apply_seed

# ---------------------------------------------------------------------------
# 1. Default seed is present on FramerConfig
# ---------------------------------------------------------------------------

def test_default_seed_is_42():
    """FramerConfig carries a seed field defaulting to 42."""
    config = FramerConfig()
    assert hasattr(config, "seed"), "FramerConfig must have a 'seed' field"
    assert config.seed == 42


def test_preset_configs_inherit_default_seed():
    """Preset configs retain the default seed unless explicitly overridden."""
    for preset_name in ("framer-tiny", "framer-small", "framer-medium", "framer-large"):
        config = FramerConfig.from_preset(preset_name)
        assert config.seed == 42, f"{preset_name} should default to seed=42"


# ---------------------------------------------------------------------------
# 2. All four training configs resolve correctly
# ---------------------------------------------------------------------------

def test_all_four_training_configs_present():
    """TRAINING_CONFIGS contains exactly the four main presets."""
    assert set(TRAINING_CONFIGS.keys()) == {
        "framer-tiny",
        "framer-small",
        "framer-medium",
        "framer-large",
    }


def test_training_configs_have_required_keys():
    """Every training config contains all REQUIRED_TRAINING_KEYS."""
    for preset_name, tc in TRAINING_CONFIGS.items():
        for key in REQUIRED_TRAINING_KEYS:
            assert key in tc, (
                f"Training config '{preset_name}' is missing required key '{key}'"
            )


def test_get_training_config_resolves_canonical_names():
    """get_training_config works with canonical preset names."""
    for preset_name in ("framer-tiny", "framer-small", "framer-medium", "framer-large"):
        tc = get_training_config(preset_name)
        assert isinstance(tc, dict)
        assert tc["seed"] == 42
        assert tc["batch_size"] > 0
        assert tc["max_steps"] > 0


def test_get_training_config_resolves_legacy_aliases():
    """get_training_config accepts the legacy size aliases tiny/small/medium/large."""
    for alias, canonical in (
        ("tiny", "framer-tiny"),
        ("small", "framer-small"),
        ("medium", "framer-medium"),
        ("large", "framer-large"),
    ):
        tc_alias = get_training_config(alias)
        tc_canonical = get_training_config(canonical)
        assert tc_alias == tc_canonical, (
            f"Alias '{alias}' should resolve to the same config as '{canonical}'"
        )


def test_get_training_config_returns_copy():
    """get_training_config returns an independent copy; mutations don't affect the registry."""
    tc = get_training_config("framer-tiny")
    tc["seed"] = 9999
    tc2 = get_training_config("framer-tiny")
    assert tc2["seed"] == 42, "Mutating a returned config must not affect TRAINING_CONFIGS"


def test_get_training_config_raises_on_unknown_name():
    """get_training_config raises KeyError for unrecognised names."""
    import pytest

    with pytest.raises(KeyError, match="No training config"):
        get_training_config("framer-nonexistent")


def test_training_configs_are_valid_framer_config_fields():
    """Every key in a training config is a valid FramerConfig field."""
    valid_fields = {f.name for f in __import__("dataclasses").fields(FramerConfig)}
    for preset_name, tc in TRAINING_CONFIGS.items():
        for key in tc:
            assert key in valid_fields, (
                f"Training config '{preset_name}' contains unknown FramerConfig field '{key}'"
            )


def test_training_config_merges_into_framer_config():
    """A training config can be merged into FramerConfig.from_preset without error."""
    for preset_name in ("framer-tiny", "framer-small", "framer-medium", "framer-large"):
        tc = get_training_config(preset_name)
        config = FramerConfig.from_preset(preset_name, **tc)
        config.validate()
        assert config.seed == 42


# ---------------------------------------------------------------------------
# 3. CLI --seed override
# ---------------------------------------------------------------------------

def test_cli_seed_overrides_default():
    """Simulates the build.py CLI path: assigning config.seed overrides the default."""
    config = FramerConfig.from_preset("framer-tiny")
    assert config.seed == 42  # default

    # Simulate: if args.seed is not None: config.seed = args.seed
    config.seed = 7
    assert config.seed == 7


def test_seed_field_is_overridable_via_from_preset():
    """from_preset accepts seed as a keyword override."""
    config = FramerConfig.from_preset("framer-tiny", seed=123)
    assert config.seed == 123


# ---------------------------------------------------------------------------
# 3b. CLI parser tests — verify argparse accepts the flags and maps them
#     to the correct FramerConfig fields. No training or model init runs.
# ---------------------------------------------------------------------------

def _parse(argv: list[str]) -> FramerConfig:
    """Parse argv through the real build.py parser and return the built config."""
    args = _make_parser().parse_args(argv)
    return _build_config_from_args(args)


def test_cli_parser_accepts_seed_flag():
    """--seed is accepted by the parser and maps to config.seed."""
    config = _parse(["--preset", "framer-tiny", "--seed", "7"])
    assert config.seed == 7


def test_cli_seed_overrides_preset_default():
    """--seed overrides the preset default of 42."""
    config_default = _parse(["--preset", "framer-tiny"])
    assert config_default.seed == 42  # unchanged when flag absent

    config_override = _parse(["--preset", "framer-tiny", "--seed", "99"])
    assert config_override.seed == 99


def test_cli_parser_accepts_warmup_steps_flag():
    """--warmup-steps is accepted by the parser and maps to config.warmup_steps."""
    config = _parse(["--preset", "framer-tiny", "--warmup-steps", "500"])
    assert config.warmup_steps == 500


def test_cli_warmup_steps_overrides_preset_default():
    """--warmup-steps overrides the preset/FramerConfig default."""
    config_default = _parse(["--preset", "framer-tiny"])
    default_warmup = config_default.warmup_steps  # preset default (2000)

    config_override = _parse(["--preset", "framer-tiny", "--warmup-steps", "123"])
    assert config_override.warmup_steps == 123
    assert config_override.warmup_steps != default_warmup


def test_cli_parser_accepts_grad_accum_flag():
    """--grad-accum is accepted by the parser and maps to config.gradient_accumulation_steps."""
    config = _parse(["--preset", "framer-tiny", "--grad-accum", "1"])
    assert config.gradient_accumulation_steps == 1


def test_cli_grad_accum_overrides_preset_default():
    """--grad-accum overrides the preset/FramerConfig default."""
    config_default = _parse(["--preset", "framer-tiny"])
    default_accum = config_default.gradient_accumulation_steps  # FramerConfig default (4)

    config_override = _parse(["--preset", "framer-tiny", "--grad-accum", "16"])
    assert config_override.gradient_accumulation_steps == 16
    assert config_override.gradient_accumulation_steps != default_accum


def test_cli_all_three_flags_together():
    """--seed, --warmup-steps, and --grad-accum can all be passed at once."""
    config = _parse([
        "--preset", "framer-small",
        "--seed", "42",
        "--warmup-steps", "1000",
        "--grad-accum", "4",
        "--precision", "bf16",
        "--max-steps", "20000",
    ])
    assert config.seed == 42
    assert config.warmup_steps == 1000
    assert config.gradient_accumulation_steps == 4
    assert config.precision == "bf16"
    assert config.max_steps == 20000


def test_cli_framer_small_documented_command():
    """The exact flags from the GUIDE.md framer-small command produce the documented config."""
    config = _parse([
        "--preset", "framer-small",
        "--seed", "42",
        "--precision", "bf16",
        "--max-steps", "20000",
        "--warmup-steps", "1000",
        "--grad-accum", "4",
    ])
    tc = get_training_config("framer-small")
    assert config.seed == tc["seed"]
    assert config.warmup_steps == tc["warmup_steps"]
    assert config.gradient_accumulation_steps == tc["gradient_accumulation_steps"]
    assert config.precision == tc["precision"]
    assert config.max_steps == tc["max_steps"]


def test_cli_framer_large_documented_command():
    """The exact flags from the GUIDE.md framer-large command produce the documented config."""
    config = _parse([
        "--preset", "framer-large",
        "--seed", "42",
        "--precision", "bf16",
        "--max-steps", "100000",
        "--warmup-steps", "2000",
        "--grad-accum", "16",
        "--grad-checkpointing",
    ])
    tc = get_training_config("framer-large")
    assert config.seed == tc["seed"]
    assert config.warmup_steps == tc["warmup_steps"]
    assert config.gradient_accumulation_steps == tc["gradient_accumulation_steps"]
    assert config.precision == tc["precision"]
    assert config.max_steps == tc["max_steps"]
    assert config.use_gradient_checkpointing == tc["use_gradient_checkpointing"]


# ---------------------------------------------------------------------------
# 4. apply_seed makes model initialization reproducible
# ---------------------------------------------------------------------------

def test_apply_seed_produces_identical_weight_init():
    """Two tiny models initialized under the same seed have identical parameters.

    Uses a tiny text-only config so the test is fast on CPU.
    No CUDA is required; torch.manual_seed is sufficient for CPU tensors.
    """
    cfg = tiny_config(
        vocab_size=256, d_model=64, n_layers=2, n_heads=8, n_kv_heads=2, d_ff=128
    )

    apply_seed(42)
    model_a = FramerModel(cfg)

    apply_seed(42)
    model_b = FramerModel(cfg)

    # Compare every parameter tensor.
    params_a = list(model_a.parameters())
    params_b = list(model_b.parameters())
    assert len(params_a) == len(params_b), "Models have different parameter counts"
    for i, (pa, pb) in enumerate(zip(params_a, params_b, strict=True)):
        assert torch.equal(pa, pb), (
            f"Parameter {i} differs between two models seeded with apply_seed(42)"
        )


def test_different_seeds_produce_different_weights():
    """Two tiny models initialized with different seeds differ in at least one parameter."""
    cfg = tiny_config(
        vocab_size=256, d_model=64, n_layers=2, n_heads=8, n_kv_heads=2, d_ff=128
    )

    apply_seed(42)
    model_a = FramerModel(cfg)

    apply_seed(99)
    model_b = FramerModel(cfg)

    any_different = any(
        not torch.equal(pa, pb)
        for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True)
    )
    assert any_different, "Different seeds should produce at least one differing parameter"
