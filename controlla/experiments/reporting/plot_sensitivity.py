"""Plot sensitivity study outputs for Controlla experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from controlla.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sensitivity study curves")
    parser.add_argument("--aggregated-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="controllability_score_mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.aggregated_path)
    output_dir = ensure_directory(args.output_dir)
    figure_path = Path(output_dir) / "sensitivity_plot.png"
    plt.figure(figsize=(7, 4))
    for topology, group in frame.groupby("graph_topology"):
        if "regularization" not in group.columns or args.metric not in group.columns:
            continue
        ordered = group.sort_values("regularization")
        plt.plot(ordered["regularization"], ordered[args.metric], marker="o", label=str(topology))
    plt.xlabel("OT regularization")
    plt.ylabel(args.metric)
    plt.title("Controlla Sensitivity Study")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    print(f"wrote sensitivity plot to {figure_path}")


if __name__ == "__main__":
    main()