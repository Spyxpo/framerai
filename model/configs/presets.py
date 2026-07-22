"""Named model-size presets for FramerAI, from laptop-scale to 1T-MoE.

Each preset is a set of :class:`FramerConfig` field overrides. Dense presets
scale the classic decoder; MoE presets add sparse experts so *total* parameters
grow into the hundreds-of-billions / trillion range while *active* (per-token)
parameters stay tractable.

Reality note: the largest MoE presets are correct, instantiable *definitions*
whose parameter counts can be estimated on a laptop (see
``model.utils.helpers.estimate_params``), but training them requires a multi-GPU
cluster. Small/mid presets are trainable on a single consumer GPU or CPU.
"""

# Dense decoder ladder (n_heads * head_dim == d_model; head_dim == 128 or 64).
_DENSE = {
    "framer-tiny": dict(  # ~15M, CPU smoke tests
        d_model=256, n_layers=6, n_heads=8, n_kv_heads=4, d_ff=1024, max_seq_len=1024,
        vision_d_model=256, vision_n_heads=4, vision_n_layers=4,
        audio_d_model=256, audio_n_heads=4, audio_n_layers=4,
        diffusion_channels=64, audio_gen_channels=32,
    ),
    "framer-small": dict(  # ~120M
        d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, d_ff=3072, max_seq_len=2048,
        vision_d_model=512, vision_n_heads=8, vision_n_layers=6,
        audio_d_model=512, audio_n_heads=8, audio_n_layers=6,
        diffusion_channels=128, audio_gen_channels=64,
    ),
    "framer-medium": dict(  # ~450M
        d_model=1024, n_layers=24, n_heads=16, n_kv_heads=8, d_ff=4096, max_seq_len=2048,
    ),
    "framer-large": dict(  # ~1.4B
        d_model=2048, n_layers=24, n_heads=16, n_kv_heads=8, d_ff=5632, max_seq_len=4096,
        vision_d_model=2048, vision_n_heads=32, vision_n_layers=16,
        audio_d_model=2048, audio_n_heads=32, audio_n_layers=16,
        diffusion_channels=512, audio_gen_channels=256,
    ),
    "framer-3b": dict(
        d_model=2560, n_layers=32, n_heads=20, n_kv_heads=4, d_ff=6912, max_seq_len=4096,
    ),
    "framer-8b": dict(  # Llama-3-8B-shaped
        d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, d_ff=14336, max_seq_len=8192,
    ),
}

# Sparse Mixture-of-Experts ladder. Labels are total / active (approximate).
_MOE = {
    "framer-tiny-moe": dict(  # laptop-validatable MoE
        d_model=256, n_layers=6, n_heads=8, n_kv_heads=4, d_ff=1024, max_seq_len=1024,
        use_moe=True, n_experts=8, n_experts_per_tok=2, n_shared_experts=1,
        expert_d_ff=512, moe_layer_freq=1, first_dense_layers=1,
    ),
    "framer-30b-a3b": dict(  # ~30B total / ~3B active
        d_model=2048, n_layers=28, n_heads=16, n_kv_heads=4, d_ff=8192, max_seq_len=4096,
        use_moe=True, n_experts=128, n_experts_per_tok=8, n_shared_experts=2,
        expert_d_ff=1536, moe_layer_freq=1, first_dense_layers=1,
    ),
    "framer-160b-a16b": dict(  # ~160B total / ~16B active
        d_model=4096, n_layers=48, n_heads=32, n_kv_heads=8, d_ff=14336, max_seq_len=8192,
        use_moe=True, n_experts=128, n_experts_per_tok=10, n_shared_experts=1,
        expert_d_ff=2048, moe_layer_freq=1, first_dense_layers=2,
    ),
    "framer-1t-a32b": dict(  # ~1T total / ~32B active (cluster-only)
        d_model=8192, n_layers=64, n_heads=64, n_kv_heads=8, d_ff=28672, max_seq_len=8192,
        use_moe=True, n_experts=256, n_experts_per_tok=4, n_shared_experts=1,
        expert_d_ff=2560, moe_layer_freq=1, first_dense_layers=3,
    ),
}

# Legacy --size aliases (build.py historically exposed tiny/small/medium/large).
_ALIASES = {
    "tiny": "framer-tiny",
    "small": "framer-small",
    "medium": "framer-medium",
    "large": "framer-large",
}

PRESETS = {**_DENSE, **_MOE}


def list_presets() -> list:
    return sorted(PRESETS.keys())


def resolve_preset_name(name: str) -> str:
    return _ALIASES.get(name, name)


def build_preset_config(name: str, **overrides):
    """Return a FramerConfig for a named preset (accepts legacy size aliases)."""
    from .model_config import FramerConfig

    resolved = resolve_preset_name(name)
    if resolved not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {', '.join(list_presets())} "
            f"(aliases: {', '.join(_ALIASES)})"
        )
    fields = {"preset": resolved, **PRESETS[resolved], **overrides}
    return FramerConfig(**fields)
