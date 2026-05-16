"""Prepare VoxCeleb normalized index."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.schemas import NormalizedRecord
from dataset_builder.utils.logging import get_logger


LOGGER = get_logger("dataset_builder.preprocess.voxceleb")


def build_records(raw_root: Path, dataset_settings: dict[str, object], raw_datasets_root: Path) -> list[NormalizedRecord]:
    del dataset_settings, raw_datasets_root
    sample_audio = next((path for path in raw_root.rglob("*.wav")), None)
    if sample_audio is None:
        LOGGER.warning("VoxCeleb root %s contains split metadata but no media files; returning zero records", raw_root)
        return []
    raise NotImplementedError("Local VoxCeleb media layout differs from the metadata-only copy currently present")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VoxCeleb normalized index")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    paths = load_builder_paths(args.config)
    config = load_yaml_config(args.config)
    dataset_settings = config["datasets"]["voxceleb"]
    records = build_records(paths.raw_datasets_root / str(dataset_settings["raw_root"]), dataset_settings, paths.raw_datasets_root)
    LOGGER.info("Prepared %s VoxCeleb records", len(records))


if __name__ == "__main__":
    main()