from dataclasses import dataclass, fields


@dataclass
class FramerConfig:
    """Configuration for the FramerAI multimodal model."""

    # Transformer / Text config
    vocab_size: int = 50304
    max_seq_len: int = 2048
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = None  # Grouped-query attention. None -> == n_heads (plain MHA).
    n_layers: int = 24
    d_ff: int = 4096
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    use_qk_norm: bool = False  # RMSNorm on q/k per head (stability at scale)

    # Mixture-of-Experts (sparse FFN). use_moe=False keeps the dense SwiGLU FFN.
    use_moe: bool = False
    n_experts: int = 0  # number of routed experts per MoE layer
    n_experts_per_tok: int = 2  # top-k routing
    n_shared_experts: int = 0  # always-on experts every token uses, 0 disables
    expert_d_ff: int = None  # per-expert FFN width. None -> falls back to d_ff
    moe_layer_freq: int = 1  # 1 = every layer is MoE; 2 = every other; etc.
    first_dense_layers: int = 0  # keep the first N layers dense (routing warmup)
    aux_loss_coef: float = 0.01  # load-balancing auxiliary loss weight
    router_z_loss_coef: float = 0.001  # router logit z-loss weight

    # Vision encoder config
    image_size: int = 256
    patch_size: int = 16
    vision_d_model: int = 1024
    vision_n_heads: int = 16
    vision_n_layers: int = 12

    # Diffusion config (image generation)
    diffusion_steps: int = 1000
    diffusion_channels: int = 256
    diffusion_resolution: int = 256
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Video generation config
    video_frames: int = 16
    video_resolution: int = 256
    temporal_layers: int = 8

    # Audio encoder config (speech / audio understanding)
    audio_sample_rate: int = 16000
    audio_n_mels: int = 80
    audio_n_fft: int = 400
    audio_hop_length: int = 160
    audio_max_frames: int = 1024
    audio_d_model: int = 1024
    audio_n_heads: int = 16
    audio_n_layers: int = 12

    # Audio generation config (text-to-audio / speech)
    audio_gen_frames: int = 128
    audio_gen_channels: int = 128

    # Code generation
    code_vocab_size: int = 50304  # shared vocab

    # Build scope. text_only skips the multimodal understanding + generation
    # submodules (vision/audio encoders, image/video/audio diffusion) so the LLM
    # core can be built, trained, and tested on its own — the focus of this pass.
    text_only: bool = False

    # Training
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    grad_clip: float = 1.0

    # Paths
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    log_dir: str = "logs"

    # Device / precision
    device: str = "auto"
    mixed_precision: bool = True
    precision: str = "bf16"  # "bf16" | "fp16" | "fp32". Chosen dtype for autocast.
    use_gradient_checkpointing: bool = False

    # Context extension (RoPE scaling)
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 1.0
    rope_scaling_type: str = "linear"  # "linear" | "ntk" | "yarn" | "none"

    # Identity
    preset: str = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        """Resolved number of key/value heads (GQA)."""
        return self.n_kv_heads if self.n_kv_heads else self.n_heads

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def validate(self) -> "FramerConfig":
        """Check the shape invariants the modules assume, and fail early if not.

        Without this a bad ``--d-model`` or a mistyped preset surfaces as an
        obscure error deep inside module construction (or, worse, only once a
        multi-hour training run reaches the multimodal decoders).
        """
        problems = []

        if self.d_model % self.n_heads:
            problems.append(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.kv_heads:
            problems.append(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.kv_heads}) for GQA"
            )
        if self.n_layers < 1:
            problems.append(f"n_layers ({self.n_layers}) must be at least 1")

        if self.use_moe:
            if self.n_experts < 1:
                problems.append("use_moe=True requires n_experts >= 1")
            if self.n_experts_per_tok > self.n_experts:
                problems.append(
                    f"n_experts_per_tok ({self.n_experts_per_tok}) exceeds "
                    f"n_experts ({self.n_experts})"
                )
            if self.first_dense_layers >= self.n_layers:
                problems.append(
                    f"first_dense_layers ({self.first_dense_layers}) leaves no MoE layer "
                    f"in n_layers ({self.n_layers})"
                )

        if not self.text_only:
            if self.image_size % self.patch_size:
                problems.append(
                    f"image_size ({self.image_size}) must be divisible by "
                    f"patch_size ({self.patch_size})"
                )
            if self.vision_d_model % self.vision_n_heads:
                problems.append("vision_d_model must be divisible by vision_n_heads")
            if self.audio_d_model % self.audio_n_heads:
                problems.append("audio_d_model must be divisible by audio_n_heads")
            # The diffusion blocks use GroupNorm(32, ch) and the video U-Net runs
            # at half the image channel width, so 64 is the real granularity.
            if self.diffusion_channels % 64:
                problems.append(
                    f"diffusion_channels ({self.diffusion_channels}) must be a multiple of 64 "
                    "(GroupNorm(32) over both the image and the half-width video U-Net)"
                )
            if self.audio_gen_channels % 32:
                problems.append(
                    f"audio_gen_channels ({self.audio_gen_channels}) must be a multiple of 32 "
                    "(GroupNorm(32))"
                )

        if problems:
            raise ValueError(
                f"Invalid FramerConfig ({self.preset or 'custom'}): " + "; ".join(problems)
            )
        return self

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Whether the transformer layer at ``layer_idx`` uses a MoE FFN."""
        if not self.use_moe or self.n_experts <= 0:
            return False
        if layer_idx < self.first_dense_layers:
            return False
        return (layer_idx - self.first_dense_layers) % max(1, self.moe_layer_freq) == 0

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "FramerConfig":
        """Build a config from a named preset (see ``model.configs.presets``)."""
        from .presets import build_preset_config

        return build_preset_config(name, **overrides)

    @classmethod
    def from_dict(cls, data: dict) -> "FramerConfig":
        """Build a config from a dict, ignoring unknown keys (forward-compat)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
