"""Load and normalize split-builder metadata from CSV or JSONL."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"sample_id", "unified_emotion"}


def _clean_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in {"", "null", "None", "nan", "NaN"}:
        return None
    return value


def _derive_dataset_source(record: dict[str, Any]) -> str | None:
    if record.get("dataset_source"):
        return str(record["dataset_source"])
    nested = record.get("dataset_sources")
    if isinstance(nested, dict):
        sources = [nested.get("image_source"), nested.get("audio_source"), nested.get("text_source")]
        available = sorted({str(item) for item in sources if item})
        if len(available) == 1:
            return available[0]
        if available:
            return "+".join(available)
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    record = {key: _clean_value(value) for key, value in raw.items()}
    record["sample_id"] = str(record.get("sample_id") or "").strip()
    record["split"] = str(record.get("split") or "").strip().lower() or None
    record["unified_emotion"] = str(record.get("unified_emotion") or "unknown").strip().lower()
    record["dataset_source"] = _derive_dataset_source(record)
    nested = record.get("dataset_sources") if isinstance(record.get("dataset_sources"), dict) else {}
    record["image_dataset_source"] = record.get("image_dataset_source") or nested.get("image_source") or record["dataset_source"]
    record["audio_dataset_source"] = record.get("audio_dataset_source") or nested.get("audio_source") or record["dataset_source"]
    record["text_dataset_source"] = record.get("text_dataset_source") or nested.get("text_source") or record["dataset_source"]
    record["text"] = str(record.get("text") or "")
    record["identity_id"] = str(record["identity_id"]).strip() if record.get("identity_id") is not None else None
    record["speaker_id"] = str(record["speaker_id"]).strip() if record.get("speaker_id") is not None else None
    record["has_image"] = bool(record.get("image_path") or record.get("reference_image_path"))
    record["has_audio"] = bool(record.get("audio_path"))
    record["has_text"] = bool(record.get("text"))
    return record


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No records found in input metadata")
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if not record.get(field)]
        if missing:
            raise ValueError(f"Record {index} is missing required fields: {missing}")


def load_metadata(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input metadata not found: {source}")
    if source.suffix.lower() == ".jsonl":
        records = _load_jsonl(source)
    elif source.suffix.lower() == ".csv":
        records = _load_csv(source)
    else:
        raise ValueError(f"Unsupported metadata format: {source.suffix}")
    normalized = [normalize_record(record) for record in records]
    validate_records(normalized)
    return normalized