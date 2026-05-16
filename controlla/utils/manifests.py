"""Manifest loading and split-aware path resolution helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_project_path(project_root: str | Path, raw_path: str | Path | None) -> Path | None:
    """Resolve a possibly-relative path against the project root."""
    if raw_path in (None, ""):
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (Path(project_root) / candidate).resolve()


def resolve_manifest_path(
    project_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    split_dir: str | Path | None = None,
    split_name: str | None = None,
    default_split_name: str | None = None,
    default_extension: str = ".jsonl",
) -> Path:
    """Resolve a manifest path from either an explicit file or a split directory."""
    resolved_manifest = resolve_project_path(project_root, manifest_path)
    if resolved_manifest is not None:
        return resolved_manifest

    resolved_split_dir = resolve_project_path(project_root, split_dir)
    effective_split_name = split_name or default_split_name
    if resolved_split_dir is None or not effective_split_name:
        raise ValueError("No manifest path or split directory configured")

    split_path = Path(effective_split_name)
    if split_path.suffix:
        file_name = split_path.name
    else:
        suffix = default_extension if default_extension.startswith(".") else f".{default_extension}"
        file_name = f"{effective_split_name}{suffix}"
    return (resolved_split_dir / file_name).resolve()


def resolve_data_manifest_path(
    config: dict[str, Any],
    *,
    train: bool = True,
    manifest_path: str | Path | None = None,
    split_dir: str | Path | None = None,
    split_name: str | None = None,
) -> Path:
    """Resolve the training data manifest configured for Controlla."""
    data_config = config["data"]
    configured_manifest = manifest_path or data_config.get("manifest_path") or data_config.get("csv_path")
    configured_split_dir = split_dir or data_config.get("split_dir")
    configured_split_name = split_name or data_config.get("split_name")
    default_split_name = "train" if train else "val"
    return resolve_manifest_path(
        config["project_root"],
        manifest_path=configured_manifest,
        split_dir=configured_split_dir,
        split_name=configured_split_name,
        default_split_name=default_split_name,
    )


def resolve_experiment_manifest_path(
    config: dict[str, Any],
    *,
    manifest_path: str | Path | None = None,
    split_dir: str | Path | None = None,
    split_name: str | None = None,
    default_split_name: str = "test",
) -> Path:
    """Resolve the evaluation or experiment manifest configured for a runner."""
    configured_manifest = manifest_path or config.get("manifest_path")
    configured_split_dir = split_dir or config.get("split_dir")
    configured_split_name = split_name or config.get("split_name")
    return resolve_manifest_path(
        config["project_root"],
        manifest_path=configured_manifest,
        split_dir=configured_split_dir,
        split_name=configured_split_name,
        default_split_name=default_split_name,
    )


def load_manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load manifest records from CSV or JSONL."""
    path = Path(manifest_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported manifest format: {path}")


def load_manifest_frame(manifest_path: str | Path) -> pd.DataFrame:
    """Load a manifest into a pandas dataframe from CSV or JSONL."""
    path = Path(manifest_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported manifest format: {path}")