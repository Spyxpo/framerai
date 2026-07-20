from dataclasses import dataclass


@dataclass
class FramerConfig:
    """Configuration for the FramerAI multimodal model."""

    # Transformer / Text config
    vocab_size: int = 50304
    max_seq_len: int = 2048
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 24
    d_ff: int = 4096
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5

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

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Paths
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    log_dir: str = "logs"

    # Device
    device: str = "auto"
    mixed_precision: bool = True

    # Context extension (RoPE scaling)
    rope_scaling_factor: float = 1.0
    rope_scaling_type: str = "linear"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2
