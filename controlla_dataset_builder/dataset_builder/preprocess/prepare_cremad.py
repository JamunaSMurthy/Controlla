"""Prepare CREMA-D normalized index."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.schemas import NormalizedRecord
from dataset_builder.utils.logging import get_logger

from .unify_labels import normalize_emotion


LOGGER = get_logger("dataset_builder.preprocess.cremad")


def parse_cremad_filename(filename: str) -> dict[str, str]:
    speaker_id, sentence_code, emotion_code, intensity = Path(filename).stem.split("_")
    return {
        "speaker_id": speaker_id,
        "sentence_code": sentence_code,
        "emotion_code": emotion_code,
        "intensity": intensity,
    }


def build_records(raw_root: Path, dataset_settings: dict[str, object], raw_datasets_root: Path) -> list[NormalizedRecord]:
    del dataset_settings, raw_datasets_root
    records: list[NormalizedRecord] = []
    for audio_path in sorted(raw_root.glob("*.wav")):
        parsed = parse_cremad_filename(audio_path.name)
        records.append(
            NormalizedRecord(
                source_dataset="cremad",
                sample_id=audio_path.stem,
                audio_path=str(audio_path.resolve()),
                raw_text=parsed["sentence_code"],
                raw_label=parsed["emotion_code"],
                unified_emotion=normalize_emotion("cremad", parsed["emotion_code"]),
                identity_id=parsed["speaker_id"],
                speaker_id=parsed["speaker_id"],
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CREMA-D normalized index")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    paths = load_builder_paths(args.config)
    config = load_yaml_config(args.config)
    dataset_settings = config["datasets"]["cremad"]
    records = build_records(paths.raw_datasets_root / str(dataset_settings["raw_root"]), dataset_settings, paths.raw_datasets_root)
    LOGGER.info("Prepared %s CREMA-D records", len(records))


if __name__ == "__main__":
    main()