"""Image metrics: Frechet distance and text-image alignment.

The feature extractor is the model's *own* vision encoder rather than Inception.
That makes these scores comparable across FramerAI checkpoints and *not*
comparable to published FID numbers, which is stated plainly here and in the
guide rather than left for someone to discover.
"""

import torch

from .metrics import cosine_alignment, frechet_from_features


@torch.no_grad()
def extract_features(model, images: torch.Tensor, device="cpu") -> torch.Tensor:
    """Pooled vision-encoder features, one vector per image."""
    model.eval()
    encoded = model.encode_image_tiles(images.to(device))
    return encoded[:, 0]  # the class token


@torch.no_grad()
def fid(model, real: torch.Tensor, fake: torch.Tensor, device="cpu") -> float:
    """Frechet distance between real and generated images, in feature space."""
    return float(
        frechet_from_features(
            extract_features(model, real, device), extract_features(model, fake, device)
        )
    )


@torch.no_grad()
def alignment_score(model, images: torch.Tensor, input_ids: torch.Tensor, device="cpu") -> float:
    """Cosine similarity between an image and its caption, in the shared space.

    This is what measures prompt adherence; a low Frechet distance with poor
    alignment means the model makes plausible images of the wrong thing.
    """
    model.eval()
    image_features = extract_features(model, images, device)
    text_features = model.forward_lm(input_ids.to(device))["hidden"].mean(dim=1)
    return float(cosine_alignment(image_features, text_features))
