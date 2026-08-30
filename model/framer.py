"""FramerAI: Unified multimodal model for text, code, image, video, and audio."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .configs import FramerConfig
from .modules.audio_encoder import AudioEncoder
from .modules.latent_diffusion import build_image_generator
from .modules.latent_video import build_video_generator
from .modules.moe import build_ffn
from .modules.multimodal_projector import MultimodalProjector
from .modules.rvq_audio import build_audio_generator
from .modules.transformer import CausalSelfAttention, FeedForward, RMSNorm, TransformerBlock
from .modules.vision_encoder import DynamicTiler, VisionEncoder


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

    # Placeholder token ids the interleaved path writes into. They live in the
    # tokenizer's reserved block, after the byte range, so their values are
    # stable across tokenizer versions.
    IMAGE_PLACEHOLDER_ID = 271
    AUDIO_PLACEHOLDER_ID = 272

    def __init__(self, config: FramerConfig, init_weights: bool = True):
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
                rope_low_freq_factor=config.rope_low_freq_factor,
                rope_high_freq_factor=config.rope_high_freq_factor,
                rope_original_max_seq_len=config.rope_original_max_seq_len,
                kv_cache_paged=config.kv_cache_paged,
                kv_cache_block_size=config.kv_cache_block_size,
                kv_cache_dtype=config.kv_cache_dtype,
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
            if init_weights:
                self.init_weights_()
            return

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            config.image_size, config.patch_size,
            config.vision_d_model, config.vision_n_heads,
            config.vision_n_layers, config.dropout
        )
        self.vision_projector = MultimodalProjector(config.vision_d_model, config.d_model)
        # With tiling on, image_size is the tile size and effective resolution
        # comes from the tile count instead.
        self.tiler = (
            DynamicTiler(config.image_size, config.vision_max_tiles, config.vision_thumbnail)
            if config.vision_tiling
            else None
        )

        # Image generation: pixel U-Net or latent diffusion transformer,
        # selected by config.image_gen_arch. Both expose the same
        # forward(images, context) -> loss and sample(shape, context, device).
        self.diffusion = build_image_generator(config)

        # Video generation: 3D U-Net or spacetime diffusion transformer,
        # selected by config.video_gen_arch. Both expose the same
        # forward(video, context) -> loss and sample(batch, context, device).
        self.video_gen = build_video_generator(config)

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

        # Audio generation: mel diffusion or RVQ acoustic tokens, selected by
        # config.audio_gen_arch. Both expose forward(target, context) -> loss.
        self.audio_gen = build_audio_generator(config)

        if init_weights:
            self.init_weights_()

    @classmethod
    def from_config_meta(cls, config) -> "FramerModel":
        """Build the model's *shapes* on the meta device, allocating nothing.

        A 2T preset is 3.6 TiB in bf16, so it cannot be constructed on one host
        even to be sharded. The meta build gives the sharding machinery a module
        tree to work from; materialize it per-rank afterwards::

            model = FramerModel.from_config_meta(config)
            # ... apply FSDP / expert-parallel sharding to the meta module ...
            model.to_empty(device="cuda")
            model.init_weights_(buffer_device="cuda")

        Weight initialization is skipped here because ``nn.init.normal_`` fails
        on meta tensors; call :meth:`init_weights_` once real storage exists.
        """
        with torch.device("meta"):
            return cls(config, init_weights=False)

    @torch.no_grad()
    def init_weights_(self, buffer_device=None) -> "FramerModel":
        """Initialize parameters and rebuild derived buffers, in place.

        Safe to call after ``to_empty(device=...)``, which allocates storage
        without contents: parameters get their distribution and every module
        owning a derived buffer (RoPE frequencies, noise schedules, mel
        filterbanks) recomputes it. Idempotent, so re-running it simply
        reinitializes.
        """
        # Let every module that knows how to initialize itself do so first. On a
        # normal build this repeats what __init__ already did; after to_empty()
        # it is the only thing standing between the convolutions, norms, and
        # positional parameters and uninitialized memory.
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Depth-scaled init: shrink residual-output projections so the residual
        # stream stays well-conditioned as depth grows.
        scale = (2 * self.config.n_layers) ** -0.5
        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                module.out_proj.weight.data.mul_(scale)
            elif isinstance(module, FeedForward):
                module.w2.weight.data.mul_(scale)

        self.reset_buffers(device=buffer_device)
        return self

    @torch.no_grad()
    def reset_buffers(self, device=None) -> "FramerModel":
        """Recompute every derived buffer in the tree.

        Buffers are computed, not learned, so ``to_empty()`` leaves them holding
        uninitialized memory. Each owning module implements ``reset_buffers``;
        this walks the tree and calls them.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "reset_buffers"):
                module.reset_buffers(device)
        return self

    # ------------------------------------------------------------------
    # Modality placement
    # ------------------------------------------------------------------

    @staticmethod
    def _scatter_modality_embeds(input_ids, x, embeds, placeholder_id):
        """Write modality embeddings into the positions holding a placeholder.

        The sequence already reserves one token per embedding, so this replaces
        rather than inserts: length, attention mask, and label alignment are all
        unchanged. A count mismatch means the sequence builder and the encoder
        disagree, which would otherwise corrupt the sequence silently, so it
        raises.
        """
        mask = input_ids == placeholder_id
        slots = int(mask.sum())
        if slots == 0:
            return x

        flat = embeds.reshape(-1, x.shape[-1])
        if flat.shape[0] != slots:
            raise ValueError(
                f"{slots} placeholder tokens for id {placeholder_id} but "
                f"{flat.shape[0]} embeddings to place"
            )
        return x.masked_scatter(mask.unsqueeze(-1), flat.to(x.dtype))

    def encode_image_tiles(self, images: torch.Tensor) -> torch.Tensor:
        """Encode an image as tiles plus a thumbnail, or as one whole image.

        Returns ``(B, tokens, d_model)``. With tiling on, every tile is encoded
        in one batched forward and the results are concatenated, so the token
        count grows with tile count while attention cost stays per-tile.
        """
        if not getattr(self, "tiler", None):
            return self.vision_projector(self.vision_encoder(images))

        tiles, _ = self.tiler(images)
        B, n_tiles = tiles.shape[:2]
        flat = tiles.reshape(B * n_tiles, *tiles.shape[2:])
        encoded = self.vision_projector(self.vision_encoder(flat))
        return encoded.reshape(B, n_tiles * encoded.shape[1], encoded.shape[2])

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
        modality_embeds: dict = None,
    ) -> dict:
        """Core language-model forward.

        Returns a dict with ``logits`` (token positions only), ``hidden`` (the
        pre-head normalized hidden states over token positions), ``aux`` (summed
        MoE auxiliary loss or None), and ``past_kvs`` (updated cache or None).

        ``modality_embeds`` maps a placeholder token id to the embeddings that
        replace it, used when ``mm_token_placement="interleaved"``. Placement
        then costs no extra sequence length and preserves where each modality
        appeared relative to the text, which prefix concatenation discards.
        """
        B, T = input_ids.shape
        x = self.token_embed(input_ids)

        if modality_embeds:
            for placeholder_id, embeds in modality_embeds.items():
                x = self._scatter_modality_embeds(input_ids, x, embeds, placeholder_id)

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
        """Encode images to embeddings, through the tiler when one is built.

        Training, contrastive pretraining and evaluation went through
        :meth:`encode_image_tiles` while inference came here and ran the plain
        encoder, so a page was seen at one tile's resolution however many tiles
        were configured. Delegating means one path: with no tiler the two are
        the same call, so nothing changes for the presets that have none.
        """
        self._require_multimodal()
        return self.encode_image_tiles(images)

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

    def forward_audio_gen(
        self,
        target_audio: torch.Tensor,
        context: torch.Tensor = None,
        target_waveform: torch.Tensor = None,
    ) -> torch.Tensor:
        """Training forward for audio generation (returns loss)."""
        self._require_multimodal()
        if target_waveform is not None:
            try:
                return self.audio_gen(target_audio, context, target_waveform=target_waveform)
            except TypeError:
                pass
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
        target_waveform: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        """Unified forward pass. Returns per-modality losses for present inputs."""
        results = {}

        # Encode input modalities. Under "prefix" they are concatenated ahead of
        # the tokens; under "interleaved" they are written into placeholder
        # positions, so an image sits where it was mentioned.
        interleaved = self.config.mm_token_placement == "interleaved"
        prefix_parts, modality_embeds = [], {}

        if images is not None:
            encoded = self.encode_image_tiles(images)
            if interleaved:
                modality_embeds[self.IMAGE_PLACEHOLDER_ID] = encoded
            else:
                prefix_parts.append(encoded)
        if audio is not None:
            encoded = self.forward_audio(audio)
            if interleaved:
                modality_embeds[self.AUDIO_PLACEHOLDER_ID] = encoded
            else:
                prefix_parts.append(encoded)

        prefix_embeds = torch.cat(prefix_parts, dim=1) if prefix_parts else None

        # Text/code generation (single LM pass reused for conditioning below)
        text_context = None
        if input_ids is not None:
            lm = self.forward_lm(
                input_ids, attention_mask, prefix_embeds,
                modality_embeds=modality_embeds or None,
            )
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
            results["audio_loss"] = self.forward_audio_gen(
                target_audio, text_context, target_waveform=target_waveform
            )


        # Total loss
        total_loss = sum(v for k, v in results.items() if k.endswith("_loss"))
        if total_loss is not None and not isinstance(total_loss, int):
            results["loss"] = total_loss

        return results
