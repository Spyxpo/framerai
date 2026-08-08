"""Metric primitives shared across modalities.

No third-party dependency: the Frechet distance needs a matrix square root, and
``scipy.linalg.sqrtm`` is the usual way to get one. Covariances are symmetric
positive semi-definite, so ``torch.linalg.eigh`` gives the same answer by
square-rooting the eigenvalues, without adding scipy to the install.
"""

import torch


def _sqrtm_psd(matrix: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Square root of a symmetric positive semi-definite matrix."""
    matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix.double())
    # Numerical error can push near-zero eigenvalues slightly negative.
    eigenvalues = eigenvalues.clamp(min=eps)
    root = eigenvectors @ torch.diag_embed(eigenvalues.sqrt()) @ eigenvectors.transpose(-2, -1)
    return root.to(matrix.dtype)


def gaussian_statistics(features: torch.Tensor) -> tuple:
    """Mean and covariance of a ``(N, D)`` feature matrix."""
    if features.dim() != 2:
        raise ValueError(f"expected (N, D) features, got {tuple(features.shape)}")
    if features.shape[0] < 2:
        raise ValueError("Frechet statistics need at least two samples")
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.t() @ centered / (features.shape[0] - 1)
    return mean, covariance


def frechet_distance(
    mean1: torch.Tensor, cov1: torch.Tensor, mean2: torch.Tensor, cov2: torch.Tensor
) -> torch.Tensor:
    """Frechet distance between two multivariate Gaussians.

    ``||m1 - m2||^2 + tr(C1 + C2 - 2 * sqrt(C1 C2))``. The product of two PSD
    matrices is not symmetric, so it is symmetrized as ``sqrt(C1) C2 sqrt(C1)``,
    which has the same eigenvalues and is PSD.
    """
    diff = mean1 - mean2
    root_cov1 = _sqrtm_psd(cov1)
    product = _sqrtm_psd(root_cov1 @ cov2 @ root_cov1)
    return diff.dot(diff) + torch.trace(cov1) + torch.trace(cov2) - 2 * torch.trace(product)


def frechet_from_features(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Frechet distance straight from two feature matrices."""
    return frechet_distance(*gaussian_statistics(real), *gaussian_statistics(fake))


def cosine_alignment(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity between paired embeddings."""
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(-1).mean()


def levenshtein(reference: list, hypothesis: list) -> int:
    """Edit distance, over any sequence of comparable items."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_item != hyp_item),  # substitution
                )
            )
        previous = current
    return previous[-1]
