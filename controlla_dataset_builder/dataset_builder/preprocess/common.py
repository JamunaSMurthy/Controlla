"""Shared helpers for Step 3 raw dataset preprocessing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from dataset_builder.constants import NORMALIZED_INDEX_COLUMNS
from dataset_builder.schemas import NormalizedRecord
from dataset_builder.utils.io import write_json, write_jsonl
from dataset_builder.utils.paths import ensure_directory


IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_SUFFIXES: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac")


def resolve_existing_path(base_dir: str | Path, configured_name: str | Path) -> Path | None:
    """Resolve a configured dataset path while tolerating naming drift like trailing spaces."""
    base_path = Path(base_dir).resolve()
    configured_path = Path(configured_name)
    if configured_path.is_absolute():
        return configured_path.resolve() if configured_path.exists() else None

    exact_match = (base_path / configured_path).resolve()
    if exact_match.exists():
        return exact_match

    normalized_target = configured_path.as_posix().strip().lower()
    for child in base_path.iterdir():
        if child.name.strip().lower() == normalized_target:
            return child.resolve()
    return None


def resolve_dataset_file(raw_datasets_root: str | Path, configured_path: str | Path) -> Path:
    """Resolve a dataset config file path under the raw datasets root."""
    path = Path(configured_path)
    if path.is_absolute():
        return path.resolve()
    return (Path(raw_datasets_root).resolve() / path).resolve()


def count_files(root: Path, suffixes: Sequence[str]) -> int:
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffix_set)


def infer_split_from_code(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"train", "training", "1"}:
        return "train"
    if lowered in {"val", "valid", "validation", "dev", "2"}:
        return "val"
    if lowered in {"test", "testing", "3"}:
        return "test"
    return None


def write_normalized_outputs(
    dataset_name: str,
    records: Iterable[NormalizedRecord],
    normalized_root: str | Path,
) -> dict[str, str | int]:
    """Validate and write normalized records in both CSV and JSONL formats."""
    validated_records = [NormalizedRecord.model_validate(record) for record in records]
    output_dir = ensure_directory(Path(normalized_root) / dataset_name)
    csv_path = output_dir / f"{dataset_name}_normalized.csv"
    jsonl_path = output_dir / f"{dataset_name}_normalized.jsonl"
    summary_path = output_dir / f"{dataset_name}_summary.json"

    rows = []
    for record in validated_records:
        dumped = record.model_dump()
        rows.append({column: dumped.get(column) for column in NORMALIZED_INDEX_COLUMNS})

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_INDEX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})

    write_jsonl(rows, jsonl_path)
    write_json(
        {
            "dataset": dataset_name,
            "num_records": len(rows),
            "csv_path": str(csv_path),
            "jsonl_path": str(jsonl_path),
        },
        summary_path,
    )
    return {
        "dataset": dataset_name,
        "num_records": len(rows),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
    }


def build_image_only_records(dataset_name: str, raw_root: Path, default_emotion: str = "neutral") -> list[NormalizedRecord]:
    """Index image-only datasets that will be used later as reference-image pools."""
    records: list[NormalizedRecord] = []
    for image_path in sorted(raw_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        records.append(
            NormalizedRecord(
                source_dataset=dataset_name,
                sample_id=image_path.stem,
                image_path=str(image_path.resolve()),
                raw_label="unlabeled",
                unified_emotion=default_emotion,
            )
        )
    return records