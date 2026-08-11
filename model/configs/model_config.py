from dataclasses import dataclass, fields

# RoPE context-extension strategies the backbone implements. Kept here, with no
# torch dependency, so both the config validator and the module can share one
# list instead of drifting apart.
ROPE_SCALING_TYPES = ("none", "linear", "ntk", "yarn")

# Decoder families. Each modality keeps its original implementation as the
# default so the small presets stay laptop-runnable, and opts in per preset.
IMAGE_GEN_ARCHS = ("unet", "latent_dit")
VIDEO_GEN_ARCHS = ("unet3d", "spacetime_dit")
AUDIO_GEN_ARCHS = ("mel_diffusion", "rvq_lm")
MM_TOKEN_PLACEMENTS = ("prefix", "interleaved")
VOCODER_ARCHS = ("griffin_lim", "istft")

# Aspect ratios a request may name. Kept here so validate() can check the
# configured default without importing the sizing helpers (and torch with them).
ASPECT_RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9")

DIFFUSION_OBJECTIVES = ("rectified_flow", "ddpm")
SAMPLER_METHODS = ("euler", "heun")


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
    # image_size is the encoder's input, and with tiling on it is the *tile*
    # size: effective resolution comes from the tile count instead.
    vision_tiling: bool = False
    vision_max_tiles: int = 12
    vision_thumbnail: bool = True

    # How encoded modality embeddings enter the token sequence. "prefix"
    # concatenates them ahead of the text, which loses their position relative
    # to it; "interleaved" writes them into placeholder token positions, so an
    # image can sit exactly where it was mentioned.
    mm_token_placement: str = "prefix"  # "prefix" | "interleaved"

    # Diffusion config (image generation)
    # image_gen_arch selects the decoder family. "unet" is the original
    # pixel-space U-Net, kept so the small presets stay laptop-runnable;
    # "latent_dit" is VAE + diffusion transformer + rectified flow, which is
    # what any real resolution needs.
    image_gen_arch: str = "unet"  # "unet" | "latent_dit"
    diffusion_steps: int = 1000
    diffusion_channels: int = 256
    # Resolution the image decoder is *trained* at. Distinct from image_size,
    # which is the vision encoder's input, and from the per-request size below.
    image_train_resolution: int = 512
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Per-request image sizing. Requests pick an aspect ratio at a size tier, or
    # give explicit dimensions; both are snapped to a legal multiple and capped.
    image_size_tier: int = 512  # square-equivalent side length
    image_default_aspect: str = "1:1"
    image_max_pixels: int = 1024 * 1024
    image_allow_custom_size: bool = True  # read sizing intent from the prompt

    # Latent diffusion (image_gen_arch="latent_dit")
    vae_latent_channels: int = 4
    vae_base_channels: int = 128
    vae_downsample: int = 8  # power of two
    dit_d_model: int = 1152
    dit_n_layers: int = 28
    dit_n_heads: int = 16
    dit_patch_size: int = 2  # patches over the latent grid, not the image
    diffusion_objective: str = "rectified_flow"  # "rectified_flow" | "ddpm"
    sampler_steps: int = 50
    sampler_method: str = "euler"  # "euler" | "heun"
    cfg_dropout_prob: float = 0.1  # conditioning dropout during training
    cfg_scale: float = 5.0  # default guidance strength at inference

    # Video generation config
    # video_gen_arch selects the decoder family. "unet3d" is the original 3D
    # U-Net, whose forward pass loops over frames in Python; "spacetime_dit" is
    # a causal 3D VAE plus a transformer with factorised spacetime attention.
    video_gen_arch: str = "unet3d"  # "unet3d" | "spacetime_dit"
    video_frames: int = 16
    video_resolution: int = 256
    video_fps: int = 24

    # Latent video (video_gen_arch="spacetime_dit")
    video_vae_latent_channels: int = 8
    video_vae_base_channels: int = 128
    video_vae_temporal_downsample: int = 4  # power of two
    video_vae_spatial_downsample: int = 8  # power of two
    video_dit_d_model: int = 1536
    video_dit_n_layers: int = 24
    video_dit_n_heads: int = 12
    video_dit_patch_size: tuple = (1, 2, 2)  # (t, h, w) over the latent grid

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
    # audio_gen_arch selects the decoder family. "mel_diffusion" treats a
    # spectrogram as an image and inverts it with Griffin-Lim, which caps
    # quality regardless of training; "rvq_lm" predicts discrete acoustic
    # tokens and decodes them with a learned codec.
    audio_gen_arch: str = "mel_diffusion"  # "mel_diffusion" | "rvq_lm"
    audio_gen_frames: int = 128
    audio_gen_channels: int = 128

    # Neural audio codec (audio_gen_arch="rvq_lm")
    codec_sample_rate: int = 24000
    codec_hop: int = 320  # 75 Hz acoustic frame rate at 24 kHz
    codec_base_channels: int = 64
    rvq_n_quantizers: int = 8
    rvq_codebook_size: int = 1024
    rvq_codebook_dim: int = 256
    rvq_quantizer_dropout: float = 0.0
    audio_lm_d_model: int = 1024
    audio_lm_n_layers: int = 12
    audio_lm_n_heads: int = 16
    vocoder_arch: str = "griffin_lim"  # "griffin_lim" | "istft"
    vocoder_d_model: int = 512
    vocoder_n_layers: int = 8
    speaker_embed_dim: int = 256
    use_speaker_conditioning: bool = False
    use_ctc_head: bool = False

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
    seed: int = 42  # RNG seed applied before model init and training for reproducibility

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
    # yarn only: the wavelength band, in rotations over the original context,
    # between "leave this frequency alone" and "interpolate it in full".
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_seq_len: int = None  # defaults to max_seq_len

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

        # An unrecognised scaling type used to fall through every branch and
        # silently apply no extension at all, which is worse than failing.
        if self.rope_scaling_type not in ROPE_SCALING_TYPES:
            problems.append(
                f"rope_scaling_type ('{self.rope_scaling_type}') must be one of "
                f"{', '.join(ROPE_SCALING_TYPES)}"
            )
        if self.rope_scaling_type == "yarn" and self.rope_high_freq_factor <= self.rope_low_freq_factor:
            problems.append(
                f"rope_high_freq_factor ({self.rope_high_freq_factor}) must exceed "
                f"rope_low_freq_factor ({self.rope_low_freq_factor})"
            )

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

            if self.image_default_aspect not in ASPECT_RATIOS:
                problems.append(
                    f"image_default_aspect ('{self.image_default_aspect}') must be one of "
                    f"{', '.join(ASPECT_RATIOS)}"
                )
            if self.image_max_pixels < 64 * 64:
                problems.append(
                    f"image_max_pixels ({self.image_max_pixels}) is below the smallest bucket"
                )

            if self.mm_token_placement not in MM_TOKEN_PLACEMENTS:
                problems.append(
                    f"mm_token_placement ('{self.mm_token_placement}') must be one of "
                    f"{', '.join(MM_TOKEN_PLACEMENTS)}"
                )
            if self.vision_tiling and self.vision_max_tiles < 1:
                problems.append("vision_tiling requires vision_max_tiles >= 1")

            if self.audio_gen_arch not in AUDIO_GEN_ARCHS:
                problems.append(
                    f"audio_gen_arch ('{self.audio_gen_arch}') must be one of "
                    f"{', '.join(AUDIO_GEN_ARCHS)}"
                )
            if self.vocoder_arch not in VOCODER_ARCHS:
                problems.append(
                    f"vocoder_arch ('{self.vocoder_arch}') must be one of "
                    f"{', '.join(VOCODER_ARCHS)}"
                )
            if self.audio_gen_arch == "rvq_lm":
                if self.rvq_n_quantizers < 1:
                    problems.append("rvq_lm requires rvq_n_quantizers >= 1")
                if self.audio_lm_d_model % self.audio_lm_n_heads:
                    problems.append("audio_lm_d_model must be divisible by audio_lm_n_heads")

            if self.video_gen_arch not in VIDEO_GEN_ARCHS:
                problems.append(
                    f"video_gen_arch ('{self.video_gen_arch}') must be one of "
                    f"{', '.join(VIDEO_GEN_ARCHS)}"
                )
            if self.video_gen_arch == "spacetime_dit":
                for factor, name in (
                    (self.video_vae_temporal_downsample, "video_vae_temporal_downsample"),
                    (self.video_vae_spatial_downsample, "video_vae_spatial_downsample"),
                ):
                    if factor < 1 or (factor & (factor - 1)):
                        problems.append(f"{name} ({factor}) must be a power of two")
                if self.video_vae_temporal_downsample > self.video_vae_spatial_downsample:
                    problems.append(
                        "video_vae_temporal_downsample cannot exceed "
                        "video_vae_spatial_downsample"
                    )
                if self.video_dit_d_model % self.video_dit_n_heads:
                    problems.append("video_dit_d_model must be divisible by video_dit_n_heads")
                if self.video_dit_d_model % 6:
                    problems.append(
                        f"video_dit_d_model ({self.video_dit_d_model}) must be divisible by 6 "
                        "(3D sin-cos positional embedding)"
                    )

            if self.image_gen_arch not in IMAGE_GEN_ARCHS:
                problems.append(
                    f"image_gen_arch ('{self.image_gen_arch}') must be one of "
                    f"{', '.join(IMAGE_GEN_ARCHS)}"
                )
            if self.diffusion_objective not in DIFFUSION_OBJECTIVES:
                problems.append(
                    f"diffusion_objective ('{self.diffusion_objective}') must be one of "
                    f"{', '.join(DIFFUSION_OBJECTIVES)}"
                )
            if self.sampler_method not in SAMPLER_METHODS:
                problems.append(
                    f"sampler_method ('{self.sampler_method}') must be one of "
                    f"{', '.join(SAMPLER_METHODS)}"
                )

            if self.image_gen_arch == "latent_dit":
                if self.vae_downsample < 1 or (self.vae_downsample & (self.vae_downsample - 1)):
                    problems.append(
                        f"vae_downsample ({self.vae_downsample}) must be a power of two"
                    )
                if self.dit_d_model % self.dit_n_heads:
                    problems.append("dit_d_model must be divisible by dit_n_heads")
                if self.dit_d_model % 4:
                    problems.append(
                        f"dit_d_model ({self.dit_d_model}) must be divisible by 4 "
                        "(2D sin-cos positional embedding)"
                    )
                if self.sampler_steps < 1:
                    problems.append(f"sampler_steps ({self.sampler_steps}) must be at least 1")
                if not 0.0 <= self.cfg_dropout_prob < 1.0:
                    problems.append(
                        f"cfg_dropout_prob ({self.cfg_dropout_prob}) must be in [0, 1)"
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
