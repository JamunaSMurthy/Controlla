"""Generate CSV and LaTeX-ready result tables for Controlla experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from controlla.evaluation.stats import format_mean_std
from controlla.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV and LaTeX tables from aggregated results")
    parser.add_argument("--aggregated-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.aggregated_path)
    output_dir = ensure_directory(args.output_dir)
    formatted = frame.copy()
    metric_bases = sorted({column[:-5] for column in frame.columns if column.endswith("_mean")})
    for base in metric_bases:
        std_column = f"{base}_std"
        if std_column in frame.columns:
            formatted[base] = [format_mean_std(mean, std) for mean, std in zip(frame[f"{base}_mean"], frame[std_column])]
    keep_columns = [column for column in formatted.columns if not column.endswith("_mean") and not column.endswith("_std")]
    table = formatted[keep_columns]
    csv_path = Path(output_dir) / "main_table.csv"
    tex_path = Path(output_dir) / "main_table.tex"
    table.to_csv(csv_path, index=False)
    tex_path.write_text(table.to_latex(index=False), encoding="utf-8")
    print(f"wrote tables to {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()