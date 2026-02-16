"""Multimodal projector to align different modality embeddings."""

import torch
import torch.nn as nn


class MultimodalProjector(nn.Module):
    """Projects vision/audio embeddings into the language model's embedding space."""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else output_dim
            layers.extend([
                nn.Linear(in_d, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
            ])
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
