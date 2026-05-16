"""Prepare RAF-DB normalized index."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.schemas import NormalizedRecord
from dataset_builder.utils.logging import get_logger

from .common import resolve_dataset_file
from .unify_labels import normalize_emotion


LOGGER = get_logger("dataset_builder.preprocess.rafdb")


def _resolve_rafdb_image(raw_root: Path, image_name: str, split: str, label: str) -> Path | None:
    candidates = [
        raw_root / "DATASET" / split / label / image_name,
        raw_root / "DATASET" / split / image_name,
        raw_root / "DATASET" / image_name,
        raw_root / image_name,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _read_label_file(raw_root: Path, labels_path: Path, split: str) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    missing_images = 0
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = row["image"]
            image_path = _resolve_rafdb_image(raw_root, image_name, split, str(row["label"]))
            if image_path is None:
                missing_images += 1
                continue
            records.append(
                NormalizedRecord(
                    source_dataset="rafdb",
                    sample_id=Path(image_name).stem,
                    split=split,
                    image_path=str(image_path),
                    raw_label=row.get("label"),
                    unified_emotion=normalize_emotion("rafdb", row.get("label")),
                )
            )
    if missing_images:
        LOGGER.warning("Skipped %s RAF-DB rows from %s due to missing images", missing_images, labels_path.name)
    return records


def build_records(raw_root: Path, dataset_settings: dict[str, object], raw_datasets_root: Path) -> list[NormalizedRecord]:
    train_labels = resolve_dataset_file(raw_datasets_root, str(dataset_settings["train_labels_file"]))
    test_labels = resolve_dataset_file(raw_datasets_root, str(dataset_settings["test_labels_file"]))
    return _read_label_file(raw_root, train_labels, "train") + _read_label_file(raw_root, test_labels, "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RAF-DB normalized index")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    paths = load_builder_paths(args.config)
    config = load_yaml_config(args.config)
    dataset_settings = config["datasets"]["rafdb"]
    records = build_records(paths.raw_datasets_root / str(dataset_settings["raw_root"]), dataset_settings, paths.raw_datasets_root)
    LOGGER.info("Prepared %s RAF-DB records", len(records))


if __name__ == "__main__":
    main()