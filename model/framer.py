"""FramerAI: Unified multimodal model for text, code, image, video, and audio."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .configs import FramerConfig
from .modules.audio_encoder import AudioEncoder
from .modules.audio_generator import AudioGenerator
from .modules.diffusion import DiffusionModule
from .modules.moe import build_ffn
from .modules.multimodal_projector import MultimodalProjector
from .modules.transformer import CausalSelfAttention, FeedForward, RMSNorm, TransformerBlock
from .modules.video_generator import VideoGenerator
from .modules.vision_encoder import VisionEncoder


class FramerModel(nn.Module):
    """
    FramerAI unified multimodal model.

    Combines:
    - Autoregressive transformer (GQA, RoPE, SwiGLU, optional MoE) for text/code
    - Vision encoder for image understanding
    - Diffusion module for image generation
    - Video diffusion for video generation
    - Audio encoder for audio/speech understanding
    - Mel diffusion for audio/speech generation
    """

    def __init__(self, config: FramerConfig):
        super().__init__()
        self.config = config

        # Text/code token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer backbone (per-layer dense or MoE FFN)
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.max_seq_len, config.dropout,
                n_kv_heads=config.kv_heads,
                use_qk_norm=config.use_qk_norm,
                rope_theta=config.rope_theta,
                rope_scaling_factor=config.rope_scaling_factor,
                rope_scaling_type=config.rope_scaling_type,
                ffn=build_ffn(config, i, config.dropout),
            )
            for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)

        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embed.weight

        # text_only builds just the LLM core (no multimodal submodules).
        self.text_only = config.text_only
        if config.text_only:
            self._init_weights()
            return

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            config.image_size, config.patch_size,
            config.vision_d_model, config.vision_n_heads,
            config.vision_n_layers, config.dropout
        )
        self.vision_projector = MultimodalProjector(config.vision_d_model, config.d_model)

        # Diffusion for image generation
        self.diffusion = DiffusionModule(
            in_channels=3,
            base_channels=config.diffusion_channels,
            context_dim=config.d_model,
            num_steps=config.diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )

        # Video generator
        self.video_gen = VideoGenerator(
            frames=config.video_frames,
            resolution=config.video_resolution,
            base_channels=config.diffusion_channels // 2,
            context_dim=config.d_model,
            num_steps=config.diffusion_steps,
        )

        # Audio encoder (audio/speech understanding)
        self.audio_encoder = AudioEncoder(
            sample_rate=config.audio_sample_rate,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
            n_mels=config.audio_n_mels,
            d_model=config.audio_d_model,
            n_heads=config.audio_n_heads,
            n_layers=config.audio_n_layers,
            max_frames=config.audio_max_frames,
            dropout=config.dropout,
        )
        self.audio_projector = MultimodalProjector(config.audio_d_model, config.d_model)

        # Audio generator (text-to-audio / speech)
        self.audio_gen = AudioGenerator(
            n_mels=config.audio_n_mels,
            n_frames=config.audio_gen_frames,
            base_channels=config.audio_gen_channels,
            context_dim=config.d_model,
            num_steps=config.diffusion_steps,
            sample_rate=config.audio_sample_rate,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Depth-scaled init: shrink residual-output projections so the residual
        # stream stays well-conditioned as depth grows (GPT-2 / Llama style).
        scale = (2 * self.config.n_layers) ** -0.5
        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                module.out_proj.weight.data.mul_(scale)
            elif isinstance(module, FeedForward):
                module.w2.weight.data.mul_(scale)

    # ------------------------------------------------------------------
    # Transformer stack
    # ------------------------------------------------------------------

    def _run_layers(self, x, attention_mask=None, past_kvs=None, use_cache=False):
        """Run the transformer stack, aggregating MoE aux losses and KV cache."""
        presents = [] if use_cache else None
        aux_total = None
        use_ckpt = self.config.use_gradient_checkpointing and self.training and not use_cache

        for i, layer in enumerate(self.layers):
            past = past_kvs[i] if past_kvs is not None else None
            if use_ckpt:
                out = checkpoint(layer, x, attention_mask, None, False, use_reentrant=False)
            else:
                out = layer(x, attention_mask, past_kv=past, use_cache=use_cache)
            x = out["x"]
            if out["aux"] is not None:
                aux_total = out["aux"] if aux_total is None else aux_total + out["aux"]
            if use_cache:
                presents.append(out["present"])
        return x, aux_total, presents

    def forward_lm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        prefix_embeds: torch.Tensor = None,
        past_kvs: list = None,
        use_cache: bool = False,
    ) -> dict:
        """Core language-model forward.

        Returns a dict with ``logits`` (token positions only), ``hidden`` (the
        pre-head normalized hidden states over token positions), ``aux`` (summed
        MoE auxiliary loss or None), and ``past_kvs`` (updated cache or None).
        """
        B, T = input_ids.shape
        x = self.token_embed(input_ids)

        prefix_len = 0
        if prefix_embeds is not None:
            prefix_len = prefix_embeds.shape[1]
            x = torch.cat([prefix_embeds, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(B, prefix_len, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        x = self.embed_dropout(x)
        x, aux, presents = self._run_layers(x, attention_mask, past_kvs=past_kvs, use_cache=use_cache)
        x = self.norm(x)

        if prefix_len:
            x = x[:, prefix_len:]

        logits = self.lm_head(x)
        return {"logits": logits, "hidden": x, "aux": aux, "past_kvs": presents}

    def forward_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        prefix_embeds: torch.Tensor = None,
        image_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """Backward-compatible text forward returning logits only."""
        if prefix_embeds is None:
            prefix_embeds = image_embeds
        return self.forward_lm(input_ids, attention_mask, prefix_embeds)["logits"]

    def _require_multimodal(self):
        if self.text_only:
            raise RuntimeError(
                "This model was built text_only=True; multimodal submodules are "
                "absent. Rebuild with text_only=False to use image/audio."
            )

    def forward_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        self._require_multimodal()
        vis_features = self.vision_encoder(images)
        return self.vision_projector(vis_features)

    def forward_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio (waveform or log-mel) to language-space embeddings."""
        self._require_multimodal()
        audio_features = self.audio_encoder(audio)
        return self.audio_projector(audio_features)

    def forward_diffusion(self, images: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training forward for image generation (returns loss)."""
        self._require_multimodal()
        return self.diffusion(images, context)

    def forward_video(self, video: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training forward for video generation (returns loss)."""
        self._require_multimodal()
        return self.video_gen(video, context)

    def forward_audio_gen(self, target_audio: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training forward for audio generation (returns loss)."""
        self._require_multimodal()
        return self.audio_gen(target_audio, context)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        images: torch.Tensor = None,
        audio: torch.Tensor = None,
        target_images: torch.Tensor = None,
        target_video: torch.Tensor = None,
        target_audio: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        """Unified forward pass. Returns per-modality losses for present inputs."""
        results = {}

        # Encode input modalities and build the prefix prepended to the tokens
        prefix_parts = []
        if images is not None:
            prefix_parts.append(self.forward_vision(images))
        if audio is not None:
            prefix_parts.append(self.forward_audio(audio))
        prefix_embeds = torch.cat(prefix_parts, dim=1) if prefix_parts else None

        # Text/code generation (single LM pass reused for conditioning below)
        text_context = None
        if input_ids is not None:
            lm = self.forward_lm(input_ids, attention_mask, prefix_embeds)
            results["logits"] = lm["logits"]
            # Reuse the LM hidden states as generation conditioning (detached to
            # keep decoder losses from backpropagating through the whole stack).
            text_context = lm["hidden"].detach()
            if lm["aux"] is not None:
                results["aux_loss"] = lm["aux"]
            if labels is not None:
                loss = F.cross_entropy(
                    lm["logits"].reshape(-1, lm["logits"].size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                results["text_loss"] = loss

        # Image generation
        if target_images is not None:
            results["image_loss"] = self.forward_diffusion(target_images, text_context)

        # Video generation
        if target_video is not None:
            results["video_loss"] = self.forward_video(target_video, text_context)

        # Audio generation
        if target_audio is not None:
            results["audio_loss"] = self.forward_audio_gen(target_audio, text_context)

        # Total loss
        total_loss = sum(v for k, v in results.items() if k.endswith("_loss"))
        if total_loss is not None and not isinstance(total_loss, int):
            results["loss"] = total_loss

        return results
