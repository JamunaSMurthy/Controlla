"""Build the aligned Controlla multimodal dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from dataset_builder.config import load_build_dataset_config, load_builder_paths
from dataset_builder.constants import OUTPUT_DATASET_MODES
from dataset_builder.schemas import AlignedSample
from dataset_builder.utils.hashing import stable_hash
from dataset_builder.utils.io import read_jsonl, write_json, write_jsonl
from dataset_builder.utils.logging import get_logger

from dataset_builder.align.embedding_matcher import match_embeddings, text_to_embedding
from dataset_builder.align.emotion_matcher import match_emotion, related_emotions
from dataset_builder.align.filtering import filter_tuples
from dataset_builder.align.identity_matcher import match_identity
from dataset_builder.align.multimodal_consistency import MultimodalConsistencyScorer
from dataset_builder.align.score_fusion import fuse_scores, load_alignment_config, passes_thresholds
from dataset_builder.align.tuple_builder import build_tuple


LOGGER = get_logger("dataset_builder.runners.build_aligned_dataset")


def _load_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _load_vector(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    return np.load(path).astype(np.float32)


def _group_by_emotion(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["unified_emotion"], []).append(record)
    return grouped


def _candidate_subset(records: list[dict[str, Any]], anchor_key: str, limit: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    start = int(stable_hash(anchor_key), 16) % len(records)
    step = max(1, len(records) // limit)
    indices = [(start + offset * step) % len(records) for offset in range(limit)]
    return [records[index] for index in indices]


def _emotion_candidate_pool(
    emotion: str,
    grouped_records: dict[str, list[dict[str, Any]]],
    *,
    allow_related_emotions: bool,
) -> list[dict[str, Any]]:
    candidate_emotions = {emotion}
    if allow_related_emotions:
        candidate_emotions.update(related_emotions(emotion))
    pool: list[dict[str, Any]] = []
    for candidate_emotion in candidate_emotions:
        pool.extend(grouped_records.get(candidate_emotion, []))
    return pool


def _resolve_split(anchor_record: dict[str, Any], build_config) -> str:
    if anchor_record.get("split") in {"train", "val", "test"}:
        return str(anchor_record["split"])
    value = (int(stable_hash(anchor_record["sample_id"]), 16) % 10_000) / 10_000.0
    train_cutoff = build_config.split.train_ratio
    val_cutoff = train_cutoff + build_config.split.val_ratio
    if value < train_cutoff:
        return "train"
    if value < val_cutoff:
        return "val"
    return "test"


def _to_json_record(sample: AlignedSample) -> dict[str, Any]:
    return sample.model_dump(mode="json")


def build_aligned_dataset(
    datasets_config_path: str | Path = "configs/datasets.yaml",
    build_config_path: str | Path = "configs/build_dataset.yaml",
    alignment_config_path: str | Path = "configs/alignment.yaml",
    anchor_datasets: set[str] | None = None,
) -> dict[str, Any]:
    paths = load_builder_paths(datasets_config_path)
    build_config = load_build_dataset_config(build_config_path)
    alignment_config = load_alignment_config(str(alignment_config_path))
    consistency_scorer = MultimodalConsistencyScorer()

    feature_root = paths.feature_cache_root
    text_root = paths.project_root / "outputs" / "text"
    output_root = paths.aligned_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    identity_records: list[dict[str, Any]] = []
    audio_records: list[dict[str, Any]] = []
    text_records: list[dict[str, Any]] = []

    for dataset_dir in sorted(path for path in feature_root.iterdir() if path.is_dir()):
        dataset_name = dataset_dir.name
        identity_records.extend(_load_optional_jsonl(dataset_dir / f"{dataset_name}_identity_features.jsonl"))
        audio_records.extend(_load_optional_jsonl(dataset_dir / f"{dataset_name}_audio_features.jsonl"))
        text_records.extend(_load_optional_jsonl(text_root / dataset_name / f"{dataset_name}_text_manifest.jsonl"))

    for record in identity_records:
        record["_vector"] = _load_vector(record["identity_feature_path"])
    for record in audio_records:
        record["_vector"] = _load_vector(record["audio_feature_path"])
    for record in text_records:
        record["_vector"] = text_to_embedding(record.get("conditioning_text") or record.get("text") or "")

    image_anchors = [
        record
        for record in identity_records
        if record.get("source_dataset") in (anchor_datasets or {"affectnet", "rafdb"})
    ]
    reference_pool = [record for record in identity_records]
    text_by_emotion = _group_by_emotion(text_records)
    audio_by_emotion = _group_by_emotion(audio_records)

    aligned_samples: list[AlignedSample] = []
    for anchor in image_anchors:
        anchor_emotion = anchor["unified_emotion"]
        anchor_vector = anchor["_vector"]

        text_pool = _emotion_candidate_pool(
            anchor_emotion,
            text_by_emotion,
            allow_related_emotions=alignment_config.matching.allow_related_emotions,
        )
        sampled_text_pool = _candidate_subset(text_pool, f"text:{anchor['sample_id']}", 24)
        matched_text_pool = match_emotion(
            anchor_emotion,
            sampled_text_pool,
            exact_emotion_only=alignment_config.matching.exact_emotion_only,
            allow_related_emotions=alignment_config.matching.allow_related_emotions,
            top_k=24,
        )
        text_candidates = match_embeddings(anchor_vector, matched_text_pool, vector_key="_vector", score_key="clip_proxy_similarity", top_k=8)
        if not text_candidates:
            continue
        text_candidate = max(
            text_candidates,
            key=lambda item: item["emotion_match_score"] * 0.8 + item["clip_proxy_similarity"] * 0.2 + (0.05 if item.get("text_origin") in {"hybrid", "original"} else 0.0),
        )

        audio_candidate = None
        audio_emotion_score = text_candidate["emotion_match_score"]
        if audio_records:
            audio_pool = _emotion_candidate_pool(
                anchor_emotion,
                audio_by_emotion,
                allow_related_emotions=alignment_config.matching.allow_related_emotions,
            )
            sampled_audio_pool = _candidate_subset(audio_pool, f"audio:{anchor['sample_id']}", 24)
            matched_audio_pool = match_emotion(
                anchor_emotion,
                sampled_audio_pool,
                exact_emotion_only=alignment_config.matching.exact_emotion_only,
                allow_related_emotions=alignment_config.matching.allow_related_emotions,
                top_k=24,
            )
            audio_candidates = match_embeddings(anchor_vector, matched_audio_pool, vector_key="_vector", score_key="imagebind_proxy_similarity", top_k=8)
            if audio_candidates:
                audio_candidate = max(
                    audio_candidates,
                    key=lambda item: item["emotion_match_score"] * 0.7 + item["imagebind_proxy_similarity"] * 0.3,
                )
                audio_emotion_score = audio_candidate["emotion_match_score"]

        reference_candidates = [record for record in _candidate_subset(reference_pool, f"ref:{anchor['sample_id']}", 24) if record["sample_id"] != anchor["sample_id"]]
        identity_candidates = match_identity(
            anchor_vector,
            reference_candidates,
            query_identity_id=anchor.get("identity_id"),
            allow_pseudo_identity=alignment_config.matching.allow_pseudo_identity,
            top_k=8,
        )
        reference_candidate = identity_candidates[0] if identity_candidates else None
        identity_similarity = reference_candidate["identity_similarity"] if reference_candidate else 0.0

        emotion_score = (text_candidate["emotion_match_score"] + audio_emotion_score) / 2.0 if audio_candidate else text_candidate["emotion_match_score"]
        consistency_scores = consistency_scorer.score(
            image_record=anchor,
            text_record=text_candidate,
            audio_record=audio_candidate,
        )
        scores = fuse_scores(
            emotion_match_score=emotion_score,
            identity_similarity=identity_similarity,
            clip_similarity=consistency_scores.image_text_similarity,
            imagebind_similarity=consistency_scores.image_audio_similarity,
            text_audio_similarity=consistency_scores.text_audio_similarity,
            config=alignment_config,
        )
        scores.clip_backend = consistency_scores.clip_backend
        scores.imagebind_backend = consistency_scores.imagebind_backend
        if not passes_thresholds(scores, alignment_config, has_reference_image=reference_candidate is not None):
            continue

        aligned_samples.append(
            build_tuple(
                image_record=anchor,
                text_record=text_candidate,
                audio_record=audio_candidate,
                reference_record=reference_candidate,
                split=_resolve_split(anchor, build_config),
                alignment_scores=scores,
            )
        )

    filtered_samples = filter_tuples(
        aligned_samples,
        minimum_final_score=alignment_config.thresholds.minimum_final_score,
        require_audio=build_config.requirements.require_audio,
        require_text=build_config.requirements.require_text,
        require_reference_image=build_config.requirements.require_reference_image,
        max_samples_per_class=build_config.alignment.max_samples_per_class,
    )

    all_records = [_to_json_record(sample) for sample in filtered_samples]
    all_manifest_path = write_jsonl(all_records, output_root / "aligned_dataset.jsonl")

    mode_outputs: dict[str, str] = {}
    for mode in OUTPUT_DATASET_MODES:
        if mode == "eval_bench":
            continue
        mode_records: list[dict[str, Any]] = []
        for sample in filtered_samples:
            if mode == "full_multimodal" and not (sample.modality_mask.has_image and sample.modality_mask.has_audio and sample.modality_mask.has_text):
                continue
            if mode == "image_text" and not (sample.modality_mask.has_image and sample.modality_mask.has_text):
                continue
            if mode == "image_audio" and not (sample.modality_mask.has_image and sample.modality_mask.has_audio):
                continue
            if mode == "text_audio_image_reference" and not (
                sample.modality_mask.has_image
                and sample.modality_mask.has_audio
                and sample.modality_mask.has_text
                and sample.modality_mask.has_reference_image
            ):
                continue
            mode_records.append(_to_json_record(sample))
        mode_path = output_root / f"{mode}.jsonl"
        write_jsonl(mode_records, mode_path)
        mode_outputs[mode] = str(mode_path)

    emotion_counts: dict[str, int] = {}
    for sample in filtered_samples:
        emotion_counts[sample.unified_emotion] = emotion_counts.get(sample.unified_emotion, 0) + 1

    summary = {
        "num_candidates": len(aligned_samples),
        "num_aligned_samples": len(filtered_samples),
        "aligned_manifest_path": str(all_manifest_path),
        "mode_outputs": mode_outputs,
        "emotion_counts": emotion_counts,
    }
    write_json(summary, output_root / "build_aligned_dataset_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the aligned Controlla multimodal dataset")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--build-config", default="configs/build_dataset.yaml")
    parser.add_argument("--alignment-config", default="configs/alignment.yaml")
    parser.add_argument("--anchor-datasets", nargs="*", default=None)
    args = parser.parse_args()
    anchor_datasets = set(args.anchor_datasets) if args.anchor_datasets else None
    build_aligned_dataset(args.datasets_config, args.build_config, args.alignment_config, anchor_datasets)


if __name__ == "__main__":
    main()
