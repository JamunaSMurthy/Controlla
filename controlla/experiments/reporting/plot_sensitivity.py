"""Plot sensitivity study outputs for Controlla experiments.

The plots are designed for the graph sensitivity / OT regularization studies
reported in the Controlla paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from controlla.utils import ensure_directory


DEFAULT_METRICS = [
    "Acc_mean",
    "TS_mean",
    "LDS_mean",
    "GC_mean",
    "ID_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sensitivity study curves")
    parser.add_argument("--aggregated-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="Acc_mean")
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Plot all default paper sensitivity metrics.",
    )
    return parser.parse_args()


def _resolve_metric(frame: pd.DataFrame, metric: str) -> str:
    candidates = [
        metric,
        f"{metric}_mean",
        metric.lower(),
        f"{metric.lower()}_mean",
    ]

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise KeyError(f"Metric {metric} not found. Available columns: {list(frame.columns)}")


def _resolve_x_column(frame: pd.DataFrame) -> str:
    for column in ["ot_regularization", "regularization", "graph_perturbation", "k"]:
        if column in frame.columns:
            return column

    raise KeyError(
        "Could not find an x-axis column. Expected one of: "
        "ot_regularization, regularization, graph_perturbation, k"
    )


def _resolve_group_column(frame: pd.DataFrame) -> str | None:
    for column in ["graph_topology", "topology", "variant", "method"]:
        if column in frame.columns:
            return column
    return None


def plot_sensitivity(
    aggregated_path: str | Path,
    output_dir: str | Path,
    metric: str = "Acc_mean",
) -> Path:
    """Plot one sensitivity curve."""

    frame = pd.read_csv(aggregated_path)
    output_dir = ensure_directory(output_dir)

    metric_column = _resolve_metric(frame, metric)
    x_column = _resolve_x_column(frame)
    group_column = _resolve_group_column(frame)

    figure_path = Path(output_dir) / f"sensitivity_{metric_column}.png"

    plt.figure(figsize=(7, 4))

    if group_column is not None:
        for group_name, group in frame.groupby(group_column, dropna=False):
            ordered = group.sort_values(x_column)
            plt.plot(
                ordered[x_column],
                ordered[metric_column],
                marker="o",
                label=str(group_name),
            )
        plt.legend()
    else:
        ordered = frame.sort_values(x_column)
        plt.plot(
            ordered[x_column],
            ordered[metric_column],
            marker="o",
        )

    plt.xlabel(x_column.replace("_", " "))
    plt.ylabel(metric_column.replace("_mean", "").replace("_", " "))
    plt.title(f"Controlla sensitivity: {metric_column.replace('_mean', '')}")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"wrote sensitivity plot to {figure_path}")

    return figure_path


def main() -> None:
    args = parse_args()

    if args.all_metrics:
        frame = pd.read_csv(args.aggregated_path)
        for metric in DEFAULT_METRICS:
            if metric in frame.columns:
                plot_sensitivity(args.aggregated_path, args.output_dir, metric=metric)
    else:
        plot_sensitivity(args.aggregated_path, args.output_dir, metric=args.metric)


if __name__ == "__main__":
    main()