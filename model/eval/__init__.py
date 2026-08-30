"""Evaluation metrics and harness for FramerAI.

Feature-based scores (Frechet distance for images and video) use the model's own
encoders rather than Inception or I3D. They are therefore comparable *across
FramerAI checkpoints* and not comparable to published FID or FVD numbers.
"""

from .harness import EvalHarness, EvalReport, default_harness
from .longcontext import (
    aggregation_accuracy,
    length_buckets,
    multi_hop_accuracy,
    single_fact_accuracy,
)
from .metrics import (
    cosine_alignment,
    frechet_distance,
    frechet_from_features,
    gaussian_statistics,
    levenshtein,
)

__all__ = [
    "EvalHarness",
    "EvalReport",
    "default_harness",
    "frechet_distance",
    "frechet_from_features",
    "gaussian_statistics",
    "cosine_alignment",
    "levenshtein",
    "single_fact_accuracy",
    "multi_hop_accuracy",
    "aggregation_accuracy",
    "length_buckets",
]
