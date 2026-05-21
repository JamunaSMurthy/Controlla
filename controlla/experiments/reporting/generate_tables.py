"""Generate CSV and LaTeX-ready result tables for Controlla experiments.

This script converts an aggregated metrics CSV into paper-style tables.

Example:
    python -m experiments.reporting.generate_tables \
        --aggregated-path experiments/outputs/reporting/aggregated.csv \
        --output-dir experiments/outputs/reporting/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from controlla.evaluation.stats import format_mean_std
from controlla.utils import ensure_directory


PAPER_TABLES: dict[str, dict[str, list[str]]] = {
    "main_results": {
        "id_columns": ["method"],
        "metrics": ["Acc", "TS", "CLIP", "IB", "H"],
    },
    "cross_dataset": {
        "id_columns": ["dataset", "method"],
        "metrics": ["Acc", "TS", "ID", "GC"],
    },
    "architecture": {
        "id_columns": ["method"],
        "metrics": ["Acc", "TS", "LDS", "ID"],
    },
    "geometry": {
        "id_columns": ["method"],
        "metrics": ["LDS", "GC", "ID", "TS"],
    },
    "retrieval": {
        "id_columns": ["method"],
        "metrics": ["I2T_R@1", "I2T_R@5", "I2A_R@1", "I2A_R@5"],
    },
    "ablations": {
        "id_columns": ["variant"],
        "metrics": ["Acc", "TS", "LDS", "GC", "ID"],
    },
    "sensitivity": {
        "id_columns": ["graph_topology", "ot_regularization", "graph_perturbation"],
        "metrics": ["Acc", "TS", "LDS", "GC", "ID"],
    },
}


METHOD_DISPLAY = {
    "styleclip": "StyleCLIP",
    "controlnet": "ControlNet",
    "controlnetpp": "ControlNet++",
    "icedit": "ICEdit",
    "flux_kontext": "FLUX.1 Kontext",
    "dreambooth": "DreamBooth",
    "dreambooth_controlnetpp": "DreamBooth + CNet++",
    "sdxl": "SDXL",
    "sdxl_controlnetpp": "SDXL + CNet++",
    "clip_retrieval": "CLIP",
    "imagebind_retrieval": "ImageBind",
    "controlla": "Controlla",
    "controlla_full": "Controlla",
}


METRIC_DISPLAY = {
    "Acc": "Acc",
    "TS": "TS",
    "CLIP": "CLIP",
    "IB": "IB",
    "H": "H",
    "LDS": "LDS",
    "GC": "GC",
    "ID": "ID",
    "I2T_R@1": "I2T R@1",
    "I2T_R@5": "I2T R@5",
    "I2A_R@1": "I2A R@1",
    "I2A_R@5": "I2A R@5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV and LaTeX tables from aggregated results")
    parser.add_argument("--aggregated-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--table",
        type=str,
        default="all",
        choices=["all", *PAPER_TABLES.keys()],
        help="Which table to generate.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for mean ± std formatting.",
    )
    return parser.parse_args()


def _metric_mean_column(frame: pd.DataFrame, metric: str) -> str | None:
    candidates = [
        f"{metric}_mean",
        f"{metric.lower()}_mean",
        metric,
        metric.lower(),
    ]

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    return None


def _metric_std_column(frame: pd.DataFrame, metric: str) -> str | None:
    candidates = [
        f"{metric}_std",
        f"{metric.lower()}_std",
    ]

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    return None


def _format_metric(frame: pd.DataFrame, metric: str, precision: int) -> pd.Series:
    mean_column = _metric_mean_column(frame, metric)
    std_column = _metric_std_column(frame, metric)

    if mean_column is None:
        return pd.Series(["--"] * len(frame), index=frame.index)

    means = pd.to_numeric(frame[mean_column], errors="coerce")

    if std_column is not None:
        stds = pd.to_numeric(frame[std_column], errors="coerce")
        return pd.Series(
            [
                "--" if pd.isna(mean) else format_mean_std(float(mean), float(std) if not pd.isna(std) else 0.0, precision=precision)
                for mean, std in zip(means, stds)
            ],
            index=frame.index,
        )

    return pd.Series(
        ["--" if pd.isna(mean) else f"{float(mean):.{precision}f}" for mean in means],
        index=frame.index,
    )


def _filter_for_table(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    filtered = frame.copy()

    if "section" in filtered.columns:
        section_aliases = {
            "main_results": ["main_results", "main", "main_table"],
            "cross_dataset": ["cross_dataset", "cross_dataset_generalization"],
            "architecture": ["architecture", "architecture_comparison"],
            "geometry": ["geometry", "geometry_evaluation"],
            "retrieval": ["retrieval", "retrieval_evaluation"],
            "ablations": ["ablation", "ablations", "component_ablations", "modality_ablations"],
            "sensitivity": ["sensitivity", "graph_sensitivity"],
        }
        aliases = section_aliases.get(table_name, [table_name])
        section = filtered["section"].astype(str).str.lower()
        filtered = filtered[section.isin([alias.lower() for alias in aliases])].copy()

    return filtered


def _display_value(column: str, value: object) -> object:
    if column == "method":
        return METHOD_DISPLAY.get(str(value), str(value))
    return value


def build_table(
    frame: pd.DataFrame,
    table_name: str,
    precision: int = 3,
) -> pd.DataFrame:
    """Build one paper-style table."""

    if table_name not in PAPER_TABLES:
        raise KeyError(f"Unknown table: {table_name}")

    spec = PAPER_TABLES[table_name]
    filtered = _filter_for_table(frame, table_name)

    id_columns = [column for column in spec["id_columns"] if column in filtered.columns]
    metrics = spec["metrics"]

    rows = pd.DataFrame(index=filtered.index)

    for column in id_columns:
        rows[column] = filtered[column].map(lambda value, col=column: _display_value(col, value))

    for metric in metrics:
        rows[METRIC_DISPLAY.get(metric, metric)] = _format_metric(filtered, metric, precision)

    if rows.empty:
        rows = pd.DataFrame(columns=[*id_columns, *[METRIC_DISPLAY.get(metric, metric) for metric in metrics]])

    return rows.reset_index(drop=True)


def _write_table(table: pd.DataFrame, output_dir: Path, name: str) -> None:
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"

    table.to_csv(csv_path, index=False)
    tex_path.write_text(
        table.to_latex(index=False, escape=False),
        encoding="utf-8",
    )

    print(f"wrote {name}: {csv_path}, {tex_path}")


def generate_tables(
    aggregated_path: str | Path,
    output_dir: str | Path,
    table: str = "all",
    precision: int = 3,
) -> dict[str, pd.DataFrame]:
    """Generate CSV and LaTeX paper tables."""

    frame = pd.read_csv(aggregated_path)
    output_dir = ensure_directory(output_dir)

    table_names: Iterable[str]
    if table == "all":
        table_names = PAPER_TABLES.keys()
    else:
        table_names = [table]

    generated: dict[str, pd.DataFrame] = {}

    for table_name in table_names:
        built = build_table(frame, table_name, precision=precision)
        _write_table(built, Path(output_dir), table_name)
        generated[table_name] = built

    return generated


def main() -> None:
    args = parse_args()
    generate_tables(
        aggregated_path=args.aggregated_path,
        output_dir=args.output_dir,
        table=args.table,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()