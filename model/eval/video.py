"""Video metrics: Frechet video distance and temporal consistency."""

import torch

from .metrics import frechet_from_features


@torch.no_grad()
def extract_features(model, video: torch.Tensor, device="cpu") -> torch.Tensor:
    """Pooled features per clip, from the video VAE encoder when available.

    Falls back to per-frame vision-encoder features, so the metric still runs on
    a preset whose video path is the 3D U-Net and has no VAE.
    """
    model.eval()
    video = video.to(device)
    vae = getattr(getattr(model, "video_gen", None), "vae", None)
    if vae is not None:
        latent = vae.encode(video).mean
        return latent.flatten(1).mean(dim=1, keepdim=True).expand(-1, latent.shape[1])

    B, C, T, H, W = video.shape
    frames = video.transpose(1, 2).reshape(B * T, C, H, W)
    features = model.encode_image_tiles(frames)[:, 0]
    return features.reshape(B, T, -1).mean(dim=1)


@torch.no_grad()
def fvd(model, real: torch.Tensor, fake: torch.Tensor, device="cpu") -> float:
    """Frechet distance between real and generated clips, in feature space."""
    return float(
        frechet_from_features(
            extract_features(model, real, device), extract_features(model, fake, device)
        )
    )


def temporal_consistency(video: torch.Tensor) -> float:
    """Mean absolute difference between consecutive frames.

    Flicker and identity drift both show up here, and neither is visible to a
    frame-wise quality metric. Lower is smoother, which is not automatically
    better: a still image scores zero.
    """
    if video.shape[2] < 2:
        return 0.0
    return float((video[:, :, 1:] - video[:, :, :-1]).abs().mean())
