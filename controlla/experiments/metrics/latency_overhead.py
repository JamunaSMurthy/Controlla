"""Latency and overhead metrics for Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from .metric_utils import first_available_column, load_prediction_manifest


def compute_latency_overhead(manifest_path: str, baseline_latency_ms: float = 1.0) -> dict[str, float]:
    """Aggregate latency statistics from a prediction manifest."""
    frame = load_prediction_manifest(manifest_path)
    latency_series = first_available_column(frame, ["latency_ms"], default="0")
    latencies = latency_series.astype(float).to_numpy(dtype=np.float32)
    if latencies.size == 0:
        latencies = np.zeros(1, dtype=np.float32)
    baseline = max(baseline_latency_ms, 1e-6)
    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "overhead_ratio": float(np.mean(latencies) / baseline),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute latency and overhead metrics")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--baseline-latency-ms", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(compute_latency_overhead(args.manifest_path, baseline_latency_ms=args.baseline_latency_ms))