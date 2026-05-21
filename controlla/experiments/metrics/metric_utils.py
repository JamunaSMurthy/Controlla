"""Shared helpers for Controlla experiment metrics."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from controlla.utils import load_manifest_frame


def load_prediction_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load a prediction manifest or source manifest for evaluation.

    Supported formats are delegated to controlla.utils.load_manifest_frame.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction manifest not found: {path}")

    frame = load_manifest_frame(str(path))
    if len(frame) == 0:
        raise ValueError(f"Prediction manifest is empty: {path}")

    return normalize_prediction_frame(frame)


def normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column aliases used across baseline manifests."""

    frame = frame.copy()

    alias_map = {
        "target_emotion": ["target_emotion", "unified_emotion", "emotion", "emotion_label"],
        "prompt": ["target_prompt", "prompt", "text", "caption"],
        "generated_image_path": ["generated_image_path", "output_image_path", "prediction_path", "image_path"],
        "reference_image_path": ["reference_image_path", "source_image_path", "ref_image_path", "image_path"],
        "audio_path": ["audio_path", "audio_feature_path", "output_audio_path", "generated_audio_path"],
        "identity_id": ["identity_id", "reference_identity", "person_id", "subject_id"],
        "predicted_emotion": ["predicted_emotion", "generated_emotion", "output_emotion"],
    }

    for canonical, aliases in alias_map.items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            if alias in frame.columns:
                frame[canonical] = frame[alias]
                break

    return frame


def first_available_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    default: str | float = "",
) -> pd.Series:
    """Return the first available column or a default-filled series."""

    for column in candidates:
        if column in frame.columns:
            return frame[column].fillna(default)

    return pd.Series([default] * len(frame), index=frame.index)


def has_column(frame: pd.DataFrame, candidates: Iterable[str]) -> bool:
    """Return True if any candidate column exists."""

    return any(column in frame.columns for column in candidates)


def safe_float(value: Any) -> float | None:
    """Convert value to float, returning None for invalid/NaN values."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(number) or math.isinf(number):
        return None

    return number


def mean_or_nan(values: Iterable[float | None]) -> float:
    """Mean over valid numeric values."""

    numeric = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.mean(numeric)) if numeric else float("nan")


def resolve_existing_paths(
    values: Iterable[Any],
    fallback_prefix: str,
) -> list[str]:
    """Replace missing paths with deterministic fallback strings.

    Fallback strings allow deterministic hash-based smoke-test encoders to run.
    They are not substitutes for real generated images in paper metrics.
    """

    resolved: list[str] = []

    for index, value in enumerate(values):
        if value is None:
            resolved.append(f"{fallback_prefix}::{index}")
            continue

        text = str(value).strip()

        if text and text.lower() not in {"nan", "none", "null"} and Path(text).exists():
            resolved.append(text)
        else:
            resolved.append(f"{fallback_prefix}::{index}")

    return resolved


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity."""

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"cosine inputs must be [N, D], got {left.shape} and {right.shape}")

    dim = min(left.shape[1], right.shape[1])
    left = left[:, :dim]
    right = right[:, :dim]

    left = left / (np.linalg.norm(left, axis=1, keepdims=True) + 1e-8)
    right = right / (np.linalg.norm(right, axis=1, keepdims=True) + 1e-8)

    return np.sum(left * right, axis=1)


def normalize_rows(array: np.ndarray) -> np.ndarray:
    """L2-normalize rows."""

    array = np.asarray(array, dtype=np.float32)
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-8)


def parse_vector_column(value: Any) -> np.ndarray | None:
    """Parse vector stored as JSON/list/string.

    Accepts values like:
        "[0.1, 0.2]"
        "0.1 0.2"
        np.ndarray
        list[float]
    """

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value.astype(np.float32).flatten()

    if isinstance(value, list | tuple):
        try:
            return np.asarray(value, dtype=np.float32).flatten()
        except ValueError:
            return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    try:
        parsed = json.loads(text)
        return np.asarray(parsed, dtype=np.float32).flatten()
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        return np.asarray(parsed, dtype=np.float32).flatten()
    except Exception:
        pass

    try:
        return np.asarray([float(part) for part in text.replace(",", " ").split()], dtype=np.float32)
    except Exception:
        return None


def stack_vector_column(frame: pd.DataFrame, candidates: Iterable[str]) -> np.ndarray | None:
    """Stack first available vector column into [N, D]."""

    for column in candidates:
        if column not in frame.columns:
            continue

        vectors = [parse_vector_column(value) for value in frame[column].tolist()]
        valid = [vector for vector in vectors if vector is not None]

        if not valid:
            continue

        dim = min(vector.shape[0] for vector in valid)
        stacked = []

        for vector in vectors:
            if vector is None:
                stacked.append(np.zeros(dim, dtype=np.float32))
            else:
                stacked.append(vector[:dim].astype(np.float32))

        return np.stack(stacked, axis=0)

    return None


def stable_recall_at_k(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    k_values: tuple[int, ...] = (1, 5),
    prefix: str = "R",
) -> dict[str, float]:
    """Compute recall@K assuming query i matches target i."""

    query_embeddings = normalize_rows(query_embeddings)
    target_embeddings = normalize_rows(target_embeddings)

    scores = query_embeddings @ target_embeddings.T
    ranking = np.argsort(-scores, axis=1)

    metrics: dict[str, float] = {}

    for k in k_values:
        effective_k = min(k, target_embeddings.shape[0])
        hits = [
            float(index in ranking[index, :effective_k])
            for index in range(query_embeddings.shape[0])
        ]
        metrics[f"{prefix}_R@{k}"] = float(np.mean(hits)) if hits else float("nan")

    return metrics