"""FramerAI: Unified multimodal model for text, code, image, and video generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import FramerConfig
from .modules.transformer import TransformerBlock, RMSNorm
from .modules.vision_encoder import VisionEncoder
from .modules.diffusion import DiffusionModule
from .modules.video_generator import VideoGenerator
from .modules.multimodal_projector import MultimodalProjector


class FramerModel(nn.Module):
    """
    FramerAI unified multimodal model.

    Combines:
    - Autoregressive transformer for text/code generation
    - Vision encoder for image understanding
    - Diffusion module for image generation
    - Video diffusion for video generation
    """

    def __init__(self, config: FramerConfig):
        super().__init__()
        self.config = config

        # Text/code token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer backbone
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.max_seq_len, config.dropout
            )
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)

        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embed.weight

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

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for text/code generation."""
        B, T = input_ids.shape
        x = self.token_embed(input_ids)

        # Prepend image embeddings if provided
        if image_embeds is not None:
            x = torch.cat([image_embeds, x], dim=1)
            if attention_mask is not None:
                img_mask = torch.ones(B, image_embeds.shape[1], device=attention_mask.device)
                attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        x = self.embed_dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)

        if image_embeds is not None:
            x = x[:, image_embeds.shape[1]:]

        logits = self.lm_head(x)
        return logits

    def forward_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        vis_features = self.vision_encoder(images)
        return self.vision_projector(vis_features)

    def forward_diffusion(self, images: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training forward for image generation (returns loss)."""
        return self.diffusion(images, context)

    def forward_video(self, video: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Training forward for video generation (returns loss)."""
        return self.video_gen(video, context)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        images: torch.Tensor = None,
        target_images: torch.Tensor = None,
        target_video: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Unified forward pass.

        Returns dict with losses for each modality present in the input.
        """
        results = {}

        # Encode images if provided
        image_embeds = None
        if images is not None:
            image_embeds = self.forward_vision(images)

        # Text/code generation
        if input_ids is not None:
            logits = self.forward_text(input_ids, attention_mask, image_embeds)
            results["logits"] = logits
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                results["text_loss"] = loss

        # Get text context for conditioning generation
        text_context = None
        if input_ids is not None:
            with torch.no_grad():
                text_embeds = self.token_embed(input_ids)
                for layer in self.layers:
                    text_embeds = layer(text_embeds, attention_mask)
                text_context = self.norm(text_embeds)

        # Image generation
        if target_images is not None:
            img_loss = self.forward_diffusion(target_images, text_context)
            results["image_loss"] = img_loss

        # Video generation
        if target_video is not None:
            vid_loss = self.forward_video(target_video, text_context)
            results["video_loss"] = vid_loss

        # Total loss
        total_loss = sum(v for k, v in results.items() if k.endswith("_loss"))
        if total_loss:
            results["loss"] = total_loss

        return results
