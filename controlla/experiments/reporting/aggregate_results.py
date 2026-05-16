"""Aggregate per-seed Controlla experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from controlla.evaluation.stats import aggregate_seed_metrics
from controlla.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics across seeds")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    metrics_files = sorted(results_root.glob("**/metrics.json"))
    rows = []
    for path in metrics_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    frame = pd.DataFrame.from_records(rows)
    id_columns = [column for column in ["method", "variant", "regularization", "graph_perturbation", "graph_topology"] if column in frame.columns]
    metric_columns = [column for column in frame.columns if column not in set(id_columns + ["seed", "prediction_manifest", "used_fallback_manifest", "settings"])]

    aggregated_rows = []
    grouped = frame.groupby(id_columns, dropna=False) if id_columns else [((), frame)]
    for keys, group in grouped:
        row = {}
        if id_columns:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row.update(dict(zip(id_columns, keys)))
        for metric in metric_columns:
            if pd.api.types.is_numeric_dtype(group[metric]):
                stats = aggregate_seed_metrics(group[metric].tolist())
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
        aggregated_rows.append(row)
    output_path = Path(args.output_path)
    ensure_directory(output_path.parent)
    pd.DataFrame.from_records(aggregated_rows).to_csv(output_path, index=False)
    print(f"wrote aggregated metrics to {output_path}")


if __name__ == "__main__":
    main()