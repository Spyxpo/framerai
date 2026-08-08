"""Contrastive pretraining for the vision tower.

The vision encoder is otherwise trained only by whatever gradient reaches it
through the language-model loss, which is weak and indirect. Aligning image and
text embeddings directly with InfoNCE gives it a signal of its own, and is what
makes the projector produce embeddings the language model can actually use.

The temperature is learned as a log, so it stays positive without a clamp and
its gradient is well-scaled across the range that matters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveVisionTrainer(nn.Module):
    """InfoNCE between pooled image embeddings and pooled text embeddings."""

    def __init__(self, model, initial_temperature: float = 0.07):
        super().__init__()
        self.model = model
        self.log_temperature = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(1.0 / initial_temperature))))
        )

    def _image_features(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.model.encode_image_tiles(images)
        # The class token carries the pooled image; with tiling it is the
        # thumbnail's, which is the view that sees the whole image.
        return F.normalize(encoded[:, 0], dim=-1)

    def _text_features(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        hidden = self.model.forward_lm(input_ids, attention_mask)["hidden"]
        if attention_mask is not None:
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)
        return F.normalize(pooled, dim=-1)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask=None) -> dict:
        image_features = self._image_features(images)
        text_features = self._text_features(input_ids, attention_mask)

        scale = self.log_temperature.exp().clamp(max=100.0)
        logits = scale * image_features @ text_features.t()
        targets = torch.arange(logits.shape[0], device=logits.device)

        # Symmetric: image-to-text and text-to-image are both supervised.
        loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        return {"loss": loss, "accuracy": accuracy, "logits": logits}
