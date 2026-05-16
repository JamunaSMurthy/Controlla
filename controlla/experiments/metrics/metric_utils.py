"""Shared helpers for Controlla experiment metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from controlla.utils import load_manifest_frame


def load_prediction_manifest(manifest_path: str) -> pd.DataFrame:
    """Load a prediction manifest or source manifest for evaluation."""
    return load_manifest_frame(manifest_path)


def first_available_column(frame: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    """Return the first available column or a default-filled series."""
    for column in candidates:
        if column in frame.columns:
            return frame[column].fillna(default)
    return pd.Series([default] * len(frame), index=frame.index)


def resolve_existing_paths(values: list[str], fallback_prefix: str) -> list[str]:
    """Replace missing paths with deterministic fallbacks for encoder hashing."""
    resolved = []
    for index, value in enumerate(values):
        if value and Path(value).exists():
            resolved.append(value)
        else:
            resolved.append(f"{fallback_prefix}::{index}")
    return resolved