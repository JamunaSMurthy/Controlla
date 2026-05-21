"""Experiment metric entrypoints for Controlla."""

from .controllability import compute_controllability
from .cross_modal_consistency import compute_cross_modal_consistency
from .disentanglement import compute_disentanglement, latent_disentanglement_score
from .identity_preservation import compute_identity_preservation
from .latency_overhead import compute_latency_overhead
from .metric_utils import (
    cosine_similarity,
    first_available_column,
    load_prediction_manifest,
    mean_or_nan,
    normalize_prediction_frame,
    resolve_existing_paths,
    safe_float,
    stable_recall_at_k,
    stack_vector_column,
)
from .paper_metrics import (
    compute_core_paper_metrics,
    compute_retrieval_metrics,
    prepare_prediction_frame,
)

__all__ = [
    "compute_controllability",
    "compute_core_paper_metrics",
    "compute_cross_modal_consistency",
    "compute_disentanglement",
    "compute_identity_preservation",
    "compute_latency_overhead",
    "compute_retrieval_metrics",
    "cosine_similarity",
    "first_available_column",
    "latent_disentanglement_score",
    "load_prediction_manifest",
    "mean_or_nan",
    "normalize_prediction_frame",
    "prepare_prediction_frame",
    "resolve_existing_paths",
    "safe_float",
    "stable_recall_at_k",
    "stack_vector_column",
]