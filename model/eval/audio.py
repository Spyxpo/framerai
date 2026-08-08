"""Audio metrics: spectral distance, SI-SDR, word error rate, speaker similarity."""

import torch

from .metrics import cosine_alignment, levenshtein


def mel_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """L1 distance between two log-mel spectrograms, on the shorter length."""
    length = min(a.shape[-1], b.shape[-1])
    return float((a[..., :length] - b[..., :length]).abs().mean())


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant signal-to-distortion ratio, in dB.

    Scale-invariant because a vocoder that gets the waveform right but the gain
    wrong is not making the error this is meant to catch.
    """
    length = min(estimate.shape[-1], reference.shape[-1])
    estimate, reference = estimate[..., :length], reference[..., :length]
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    scale = (estimate * reference).sum(-1, keepdim=True) / (
        reference.pow(2).sum(-1, keepdim=True) + eps
    )
    projection = scale * reference
    noise = estimate - projection
    ratio = projection.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps)
    return float(10 * torch.log10(ratio + eps).mean())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Edit distance over words, normalized by the reference length."""
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not hypothesis.split() else 1.0
    return levenshtein(ref_words, hypothesis.split()) / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(list(reference), list(hypothesis)) / len(reference)


def speaker_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two speaker embeddings."""
    return float(cosine_alignment(a, b))
