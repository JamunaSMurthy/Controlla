"""Prepare EmoBank normalized index."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.schemas import NormalizedRecord
from dataset_builder.utils.logging import get_logger

from .common import infer_split_from_code, resolve_dataset_file
from .unify_labels import normalize_emotion


LOGGER = get_logger("dataset_builder.preprocess.emobank")


def build_records(raw_root: Path, dataset_settings: dict[str, object], raw_datasets_root: Path) -> list[NormalizedRecord]:
    csv_path = resolve_dataset_file(raw_datasets_root, str(dataset_settings["csv_file"]))
    records: list[NormalizedRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            valence = float(row["V"])
            arousal = float(row["A"])
            records.append(
                NormalizedRecord(
                    source_dataset="emobank",
                    sample_id=row["id"],
                    split=infer_split_from_code(row.get("split")),
                    text=row["text"].strip(),
                    raw_text=row["text"],
                    raw_label="vad_heuristic",
                    unified_emotion=normalize_emotion("emobank", None, valence=valence, arousal=arousal),
                    valence=valence,
                    arousal=arousal,
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EmoBank normalized index")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    paths = load_builder_paths(args.config)
    config = load_yaml_config(args.config)
    dataset_settings = config["datasets"]["emobank"]
    records = build_records(paths.raw_datasets_root / str(dataset_settings["raw_root"]), dataset_settings, paths.raw_datasets_root)
    LOGGER.info("Prepared %s EmoBank records", len(records))


if __name__ == "__main__":
    main()