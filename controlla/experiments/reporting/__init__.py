"""Reporting utilities for Controlla experiments.

This package turns per-run metric JSON files into:
- aggregated CSV files,
- LaTeX-ready result tables,
- sensitivity plots,
- paper-style tables for Controlla experiments.
"""

from .aggregate_results import aggregate_results, collect_metric_files
from .generate_tables import generate_tables
from .plot_sensitivity import plot_sensitivity

__all__ = [
    "aggregate_results",
    "collect_metric_files",
    "generate_tables",
    "plot_sensitivity",
]