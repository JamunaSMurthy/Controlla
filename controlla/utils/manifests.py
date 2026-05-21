"""Manifest loading and split-aware path resolution helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_project_path(
    project_root: str | Path,
    raw_path: str | Path | None,
) -> Path | None:
    """Resolve a possibly-relative path against project root."""

    if raw_path in (None, ""):
        return None

    candidate = Path(str(raw_path))

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
    """Resolve a manifest path from explicit file or split directory."""

    resolved_manifest = resolve_project_path(project_root, manifest_path)

    if resolved_manifest is not None:
        if not resolved_manifest.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {resolved_manifest}")
        return resolved_manifest

    resolved_split_dir = resolve_project_path(project_root, split_dir)
    effective_split_name = split_name or default_split_name

    if resolved_split_dir is None or not effective_split_name:
        raise ValueError("No manifest path or split directory configured")

    suffix = default_extension if default_extension.startswith(".") else f".{default_extension}"
    split_path = Path(str(effective_split_name))

    file_name = split_path.name if split_path.suffix else f"{effective_split_name}{suffix}"
    resolved = (resolved_split_dir / file_name).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {resolved}")

    return resolved


def resolve_data_manifest_path(
    config: dict[str, Any],
    *,
    train: bool = True,
    manifest_path: str | Path | None = None,
    split_dir: str | Path | None = None,
    split_name: str | None = None,
) -> Path:
    """Resolve the train/validation data manifest configured for Controlla."""

    data_config = config["data"]

    configured_manifest = (
        manifest_path
        or data_config.get("manifest_path")
        or data_config.get("csv_path")
    )
    configured_split_dir = split_dir or data_config.get("split_dir")

    if split_name is not None:
        configured_split_name = split_name
    elif train:
        configured_split_name = data_config.get("split_name")
    else:
        configured_split_name = data_config.get("val_split_name") or "val"

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
    """Resolve evaluation/experiment manifest for a runner."""

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
    """Load manifest records from CSV, JSONL, JSON, or Parquet."""

    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "records" in payload:
            return list(payload["records"])
        raise ValueError(f"Unsupported JSON manifest structure: {path}")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")

    raise ValueError(f"Unsupported manifest format: {path}")


def load_manifest_frame(manifest_path: str | Path) -> pd.DataFrame:
    """Load a manifest into a pandas DataFrame."""

    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame.from_records(payload)
        if isinstance(payload, dict) and "records" in payload:
            return pd.DataFrame.from_records(payload["records"])
        return pd.DataFrame.from_records([payload])

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported manifest format: {path}")


def write_manifest_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write manifest rows to CSV or JSONL."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if suffix == ".csv":
        if not rows:
            output_path.write_text("", encoding="utf-8")
            return

        fieldnames = sorted({key for row in rows for key in row.keys()})

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    raise ValueError(f"Unsupported manifest output format: {output_path}")