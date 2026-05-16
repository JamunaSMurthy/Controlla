"""Persist split-builder outputs to JSONL and JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _strip_transient_fields(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in record.items():
        if key.startswith("_"):
            continue
        if hasattr(value, "tolist"):
            cleaned[key] = value.tolist()
        else:
            cleaned[key] = value
    return cleaned


def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_strip_transient_fields(record), ensure_ascii=True) + "\n")
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_split_outputs(output_dir: str | Path, assigned_groups: list[dict[str, Any]], dropped_groups: list[dict[str, Any]]) -> dict[str, str]:
    target_dir = Path(output_dir)
    split_records = {"train": [], "val": [], "test": []}
    grouped_records: list[dict[str, Any]] = []
    dropped_records: list[dict[str, Any]] = []
    for group in assigned_groups:
        for sample in group["samples"]:
            updated = {**sample, "split": group["assigned_split"], "group_id": group["group_id"]}
            split_records[group["assigned_split"]].append(updated)
            grouped_records.append(updated)

    for group in dropped_groups:
        samples = group.get("samples") or []
        if samples:
            for sample in samples:
                dropped_records.append(
                    {
                        **sample,
                        "group_id": group.get("group_id"),
                        "drop_reason": group.get("drop_reason") or group.get("reason"),
                        "drop_stage": group.get("drop_stage"),
                        "original_split": group.get("assigned_split"),
                    }
                )
        else:
            dropped_records.append(group)

    outputs = {
        "train": str(write_jsonl(split_records["train"], target_dir / "train.jsonl")),
        "val": str(write_jsonl(split_records["val"], target_dir / "val.jsonl")),
        "test": str(write_jsonl(split_records["test"], target_dir / "test.jsonl")),
        "grouped_metadata": str(write_jsonl(grouped_records, target_dir / "grouped_metadata.jsonl")),
        "dropped_samples": str(write_jsonl(dropped_records, target_dir / "dropped_samples.jsonl")),
    }
    return outputs