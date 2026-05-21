"""Aggregate Controlla experiment results across seeds.

This script searches for metrics.json files under a results directory and
aggregates numeric metrics by method/variant/topology/regularization/etc.

Example:
    python -m experiments.reporting.aggregate_results \
        --results-root experiments/outputs/results \
        --output-path experiments/outputs/reporting/aggregated.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from controlla.evaluation.stats import aggregate_seed_metrics
from controlla.utils import ensure_directory


DEFAULT_ID_COLUMNS = [
    "section",
    "dataset",
    "split",
    "method",
    "variant",
    "baseline",
    "regularization",
    "ot_regularization",
    "graph_perturbation",
    "graph_topology",
    "traversal_mode",
    "k",
    "seed",
]


EXCLUDE_COLUMNS = {
    "prediction_manifest",
    "used_fallback_manifest",
    "settings",
    "config",
    "notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics across seeds")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="**/metrics.json")
    parser.add_argument(
        "--keep-seed",
        action="store_true",
        help="Keep seed as a grouping column instead of aggregating across seeds.",
    )
    return parser.parse_args()


def _flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric dictionaries."""

    flat: dict[str, Any] = {}

    for key, value in payload.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=new_key))
        else:
            flat[new_key] = value

    return flat


def collect_metric_files(results_root: str | Path, pattern: str = "**/metrics.json") -> list[Path]:
    """Collect metric JSON files."""

    root = Path(results_root)

    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")

    return sorted(root.glob(pattern))


def _load_metric_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Metric file must contain a JSON object: {path}")

    flat = _flatten_dict(payload)
    flat["_metrics_path"] = str(path)

    # Infer useful metadata from directory structure if not present.
    parts = path.parts
    if "method" not in flat:
        for candidate in reversed(parts):
            if candidate.startswith("seed_"):
                continue
            if candidate not in {"metrics.json", "results", "outputs"}:
                flat.setdefault("method", candidate)
                break

    return flat


def _is_numeric_series(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().any()


def aggregate_results(
    results_root: str | Path,
    output_path: str | Path,
    pattern: str = "**/metrics.json",
    keep_seed: bool = False,
) -> pd.DataFrame:
    """Aggregate metric JSON files and save aggregated CSV."""

    metric_files = collect_metric_files(results_root, pattern=pattern)

    if not metric_files:
        raise FileNotFoundError(f"No metric files found under {results_root} with pattern {pattern}")

    rows = [_load_metric_file(path) for path in metric_files]
    frame = pd.DataFrame.from_records(rows)

    if frame.empty:
        raise ValueError("No metric rows loaded.")

    id_candidates = [column for column in DEFAULT_ID_COLUMNS if column in frame.columns]
    if not keep_seed and "seed" in id_candidates:
        id_candidates.remove("seed")

    id_columns = [
        column
        for column in id_candidates
        if column not in EXCLUDE_COLUMNS
    ]

    metric_columns = [
        column
        for column in frame.columns
        if column not in set(id_columns)
        and column not in EXCLUDE_COLUMNS
        and not column.startswith("_")
        and _is_numeric_series(frame[column])
    ]

    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    aggregated_rows: list[dict[str, Any]] = []

    grouped = frame.groupby(id_columns, dropna=False) if id_columns else [((), frame)]

    for keys, group in grouped:
        row: dict[str, Any] = {}

        if id_columns:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row.update(dict(zip(id_columns, keys)))

        for metric in metric_columns:
            values = group[metric].dropna().tolist()

            if not values:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_n"] = 0
                continue

            stats = aggregate_seed_metrics(values)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_n"] = len(values)

        aggregated_rows.append(row)

    aggregated = pd.DataFrame.from_records(aggregated_rows)

    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    aggregated.to_csv(output_path, index=False)

    print(f"read {len(metric_files)} metric files")
    print(f"wrote aggregated metrics to {output_path}")

    return aggregated


def main() -> None:
    args = parse_args()
    aggregate_results(
        results_root=args.results_root,
        output_path=args.output_path,
        pattern=args.pattern,
        keep_seed=args.keep_seed,
    )


if __name__ == "__main__":
    main()