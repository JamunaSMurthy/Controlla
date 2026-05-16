"""Prepare RAVDESS normalized index."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.schemas import NormalizedRecord
from dataset_builder.utils.logging import get_logger

from .unify_labels import normalize_emotion


LOGGER = get_logger("dataset_builder.preprocess.ravdess")


def parse_ravdess_filename(filename: str) -> dict[str, str]:
    modality, channel, emotion, intensity, statement, repetition, actor = Path(filename).stem.split("-")
    return {
        "modality": modality,
        "channel": channel,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
    }


def build_records(raw_root: Path, dataset_settings: dict[str, object], raw_datasets_root: Path) -> list[NormalizedRecord]:
    del dataset_settings, raw_datasets_root
    records: list[NormalizedRecord] = []
    for audio_path in sorted(raw_root.rglob("*.wav")):
        parsed = parse_ravdess_filename(audio_path.name)
        actor_id = f"actor_{parsed['actor']}"
        records.append(
            NormalizedRecord(
                source_dataset="ravdess",
                sample_id=audio_path.stem,
                audio_path=str(audio_path.resolve()),
                raw_label=parsed["emotion"],
                unified_emotion=normalize_emotion("ravdess", parsed["emotion"]),
                identity_id=actor_id,
                speaker_id=actor_id,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RAVDESS normalized index")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    paths = load_builder_paths(args.config)
    config = load_yaml_config(args.config)
    dataset_settings = config["datasets"]["ravdess"]
    records = build_records(paths.raw_datasets_root / str(dataset_settings["raw_root"]), dataset_settings, paths.raw_datasets_root)
    LOGGER.info("Prepared %s RAVDESS records", len(records))


if __name__ == "__main__":
    main()