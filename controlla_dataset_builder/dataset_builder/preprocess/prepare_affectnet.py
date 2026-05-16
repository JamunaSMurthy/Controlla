"""Prepare AffectNet normalized index."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.schemas import NormalizedRecord
from dataset_builder.utils.logging import get_logger

from .common import resolve_dataset_file
from .unify_labels import normalize_emotion


LOGGER = get_logger("dataset_builder.preprocess.affectnet")


def resolve_affectnet_image_path(raw_root: Path, relative_path: str) -> tuple[Path | None, str | None]:
    candidates = [
        (raw_root / relative_path, None),
        (raw_root / "Train" / relative_path, "train"),
        (raw_root / "Test" / relative_path, "test"),
    ]
    for path, split in candidates:
        if path.exists():
            return path.resolve(), split
    return None, None


def build_records(raw_root: Path, dataset_settings: dict[str, object], raw_datasets_root: Path) -> list[NormalizedRecord]:
    labels_path = resolve_dataset_file(raw_datasets_root, str(dataset_settings.get("labels_file", raw_root / "labels.csv")))
    records: list[NormalizedRecord] = []
    missing_images = 0
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = row["pth"]
            image_path, split = resolve_affectnet_image_path(raw_root, relative_path)
            if image_path is None:
                missing_images += 1
                continue
            records.append(
                NormalizedRecord(
                    source_dataset="affectnet",
                    sample_id=image_path.stem,
                    split=split,
                    image_path=str(image_path),
                    raw_label=row.get("label"),
                    unified_emotion=normalize_emotion("affectnet", row.get("label")),
                )
            )
    if missing_images:
        LOGGER.warning("Skipped %s AffectNet rows with missing image files", missing_images)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AffectNet normalized index")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    paths = load_builder_paths(args.config)
    config = load_yaml_config(args.config)
    dataset_settings = config["datasets"]["affectnet"]
    records = build_records(paths.raw_datasets_root / str(dataset_settings["raw_root"]), dataset_settings, paths.raw_datasets_root)
    LOGGER.info("Prepared %s AffectNet records", len(records))


if __name__ == "__main__":
    main()