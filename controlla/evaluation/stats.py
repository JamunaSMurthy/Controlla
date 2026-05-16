"""Statistics helpers for aggregating experiment results."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def aggregate_seed_metrics(values: Iterable[float]) -> dict[str, float]:
    """Return mean and standard deviation for a metric across seeds."""
    array = np.asarray(list(values), dtype=np.float32)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    """Format values for CSV and LaTeX-ready table cells."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"