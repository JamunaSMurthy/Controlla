"""Identity matching between target and reference images."""

from __future__ import annotations

from typing import Any

import numpy as np


def _resample_vector(vector: np.ndarray, target_dim: int) -> np.ndarray:
    if vector.shape[0] == target_dim:
        return vector.astype(np.float32)
    if vector.shape[0] == 0:
        return np.zeros(target_dim, dtype=np.float32)
    source_axis = np.linspace(0.0, 1.0, num=vector.shape[0], dtype=np.float32)
    target_axis = np.linspace(0.0, 1.0, num=target_dim, dtype=np.float32)
    return np.interp(target_axis, source_axis, vector).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] != right.shape[0]:
        target_dim = max(left.shape[0], right.shape[0])
        left = _resample_vector(left, target_dim)
        right = _resample_vector(right, target_dim)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    cosine = float(np.dot(left, right) / denominator)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def match_identity(
    query_vector: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    query_identity_id: str | None = None,
    allow_pseudo_identity: bool = True,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_identity_id = candidate.get("identity_id")
        if query_identity_id and candidate_identity_id and candidate_identity_id == query_identity_id:
            score = 1.0
        elif allow_pseudo_identity:
            score = cosine_similarity(query_vector, candidate["_vector"])
        else:
            score = 0.0
        scored_candidates.append({**candidate, "identity_similarity": score})
    scored_candidates.sort(key=lambda item: item["identity_similarity"], reverse=True)
    return scored_candidates[:top_k]