"""Build feature caches for normalized datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from dataset_builder.config import load_builder_paths
from dataset_builder.preprocess.audio_features import extract_audio_embedding
from dataset_builder.preprocess.face_identity import ArcFaceIdentityEncoder, extract_identity_features
from dataset_builder.preprocess.text_sources import load_normalized_records
from dataset_builder.utils.io import read_jsonl, write_json, write_jsonl
from dataset_builder.utils.logging import get_logger


LOGGER = get_logger("dataset_builder.runners.build_features")


def _load_text_manifest(dataset_name: str, text_root: Path) -> dict[str, dict[str, Any]]:
    jsonl_path = text_root / dataset_name / f"{dataset_name}_text_manifest.jsonl"
    if not jsonl_path.exists():
        return {}
    return {record["sample_id"]: record for record in read_jsonl(jsonl_path)}


def _write_enriched_normalized(records: list[dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "source_dataset",
        "sample_id",
        "split",
        "image_path",
        "reference_image_path",
        "audio_path",
        "audio_feature_path",
        "text",
        "raw_text",
        "raw_label",
        "unified_emotion",
        "identity_id",
        "speaker_id",
        "valence",
        "arousal",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: "" if record.get(key) is None else record.get(key) for key in fieldnames})
    return path


def build_features(config_path: str | Path = "configs/datasets.yaml", datasets: set[str] | None = None) -> dict[str, object]:
    paths = load_builder_paths(config_path)
    text_root = paths.project_root / "outputs" / "text"
    encoder = ArcFaceIdentityEncoder()
    summary: dict[str, object] = {"datasets": {}, "feature_cache_root": str(paths.feature_cache_root)}

    for dataset_dir in sorted(path for path in paths.normalized_root.iterdir() if path.is_dir()):
        dataset_name = dataset_dir.name
        if datasets is not None and dataset_name not in datasets:
            continue

        normalized_csv = dataset_dir / f"{dataset_name}_normalized.csv"
        if not normalized_csv.exists():
            continue
        normalized_records = load_normalized_records(normalized_csv)
        text_map = _load_text_manifest(dataset_name, text_root)

        feature_dir = paths.feature_cache_root / dataset_name
        audio_dir = feature_dir / "audio"
        identity_dir = feature_dir / "identity"
        audio_dir.mkdir(parents=True, exist_ok=True)
        identity_dir.mkdir(parents=True, exist_ok=True)

        audio_manifest: list[dict[str, Any]] = []
        identity_manifest: list[dict[str, Any]] = []
        enriched_records: list[dict[str, Any]] = []

        for record in normalized_records:
            text_record = text_map.get(record.sample_id, {})
            enriched = record.model_dump()

            if record.audio_path:
                audio_output_path = audio_dir / f"{record.sample_id}.npy"
                audio_embedding, metadata = extract_audio_embedding(record.audio_path)
                npy_metadata = dict(metadata)
                import numpy as np

                np.save(audio_output_path, audio_embedding)
                enriched["audio_feature_path"] = str(audio_output_path.resolve())
                audio_manifest.append(
                    {
                        "source_dataset": dataset_name,
                        "sample_id": record.sample_id,
                        "audio_path": record.audio_path,
                        "audio_feature_path": str(audio_output_path.resolve()),
                        "embedding_dim": int(audio_embedding.size),
                        "duration_sec": npy_metadata["duration_sec"],
                        "sample_rate": int(npy_metadata["sample_rate"]),
                        "unified_emotion": text_record.get("unified_emotion", record.unified_emotion),
                        "conditioning_text": text_record.get("conditioning_text"),
                    }
                )

            if record.image_path:
                identity_output_path = identity_dir / f"{record.sample_id}.npy"
                identity_feature_path, inferred_identity = extract_identity_features(
                    record.image_path,
                    output_path=identity_output_path,
                    encoder=encoder,
                )
                enriched["identity_id"] = record.identity_id or inferred_identity
                identity_manifest.append(
                    {
                        "source_dataset": dataset_name,
                        "sample_id": record.sample_id,
                        "image_path": record.image_path,
                        "identity_feature_path": identity_feature_path,
                        "identity_id": enriched["identity_id"],
                        "speaker_id": record.speaker_id,
                        "unified_emotion": text_record.get("unified_emotion", record.unified_emotion),
                        "conditioning_text": text_record.get("conditioning_text"),
                    }
                )

            enriched_records.append(enriched)

        audio_manifest_path = write_jsonl(audio_manifest, feature_dir / f"{dataset_name}_audio_features.jsonl")
        identity_manifest_path = write_jsonl(identity_manifest, feature_dir / f"{dataset_name}_identity_features.jsonl")
        enriched_path = _write_enriched_normalized(enriched_records, feature_dir / f"{dataset_name}_normalized_with_features.csv")

        dataset_summary = {
            "dataset": dataset_name,
            "num_records": len(normalized_records),
            "audio_feature_records": len(audio_manifest),
            "identity_feature_records": len(identity_manifest),
            "audio_manifest_path": str(audio_manifest_path),
            "identity_manifest_path": str(identity_manifest_path),
            "enriched_normalized_path": str(enriched_path),
        }
        write_json(dataset_summary, feature_dir / f"{dataset_name}_feature_summary.json")
        summary["datasets"][dataset_name] = dataset_summary
        LOGGER.info(
            "Built %s audio and %s identity feature records for %s",
            len(audio_manifest),
            len(identity_manifest),
            dataset_name,
        )

    write_json(summary, paths.feature_cache_root / "build_features_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audio and identity feature caches")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    selected = set(args.datasets) if args.datasets else None
    build_features(args.config, selected)


if __name__ == "__main__":
    main()