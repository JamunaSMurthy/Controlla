"""Cross-modal embedding similarity helpers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .identity_matcher import cosine_similarity


TOKEN_RE = re.compile(r"[a-z0-9']+")


def text_to_embedding(text: str, dimension: int = 64) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return vector
    for token in tokens:
        index = sum((offset + 1) * ord(character) for offset, character in enumerate(token)) % dimension
        vector[index] += 1.0
    vector /= max(float(np.linalg.norm(vector)), 1.0)
    return vector.astype(np.float32)


def match_embeddings(
    query_vector: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    vector_key: str = "_vector",
    score_key: str = "similarity",
    top_k: int = 16,
) -> list[dict[str, Any]]:
    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        score = cosine_similarity(query_vector, candidate[vector_key])
        scored_candidates.append({**candidate, score_key: score})
    scored_candidates.sort(key=lambda item: item[score_key], reverse=True)
    return scored_candidates[:top_k]