"""Validate split assignments for leakage and coverage metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .build_groups import _should_hard_link_audio_hash_bucket
from .config import SplitBuilderConfig
from .similarity_utils import cosine_similarity, hamming_distance, image_perceptual_hash, resolve_embedding, sha256_file, sha256_text, vector_signature


def _records_by_split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["split"]].append(record)
    return grouped


def _overlap_counts(records: list[dict[str, Any]], key_name: str) -> tuple[int, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    sample_count = 0
    for record in records:
        key = record.get(key_name)
        if not key:
            continue
        grouped[str(key)].add(record["split"])
    leakage_keys = [key for key, splits in grouped.items() if len(splits) > 1]
    for key in leakage_keys:
        sample_count += sum(1 for record in records if str(record.get(key_name) or "") == key)
    return len(leakage_keys), sample_count


def _exact_overlap(records: list[dict[str, Any]], field_name: str) -> int:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = record.get(field_name)
        if key:
            grouped[str(key)].add(record["split"])
    return sum(1 for splits in grouped.values() if len(splits) > 1)


def _audio_overlap_metrics(records: list[dict[str, Any]], config: SplitBuilderConfig) -> dict[str, int]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = record.get("audio_hash")
        if key:
            buckets[str(key)].append(record)

    raw_exact_overlap = 0
    actionable_exact_overlap = 0
    ambiguous_overlap = 0
    for bucket in buckets.values():
        if len({record["split"] for record in bucket}) < 2:
            continue
        raw_exact_overlap += 1
        should_link, _, _ = _should_hard_link_audio_hash_bucket(bucket, config)
        if should_link:
            actionable_exact_overlap += 1
        else:
            ambiguous_overlap += 1
    return {
        "raw_exact_audio_duplicate_overlap": raw_exact_overlap,
        "actionable_exact_audio_duplicate_overlap": actionable_exact_overlap,
        "ambiguous_audio_reuse_overlap_count": ambiguous_overlap,
    }


def _near_duplicate_overlap(records: list[dict[str, Any]], config: SplitBuilderConfig) -> int:
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("image_path"):
            continue
        phash = record.get("_image_phash") or record.get("image_phash") or image_perceptual_hash(record.get("image_path"))
        if not phash:
            continue
        record["_validation_phash"] = phash
        bucketed[phash[:6]].append(record)
    overlap_count = 0
    for bucket in bucketed.values():
        if len(bucket) > config.similarity.max_bucket_size:
            bucket = sorted(bucket, key=lambda item: item["sample_id"])[: config.similarity.max_bucket_size]
        comparisons = 0
        for index, left in enumerate(bucket):
            for right in bucket[index + 1 :]:
                comparisons += 1
                if comparisons > config.similarity.max_pair_comparisons_per_bucket:
                    break
                if left["split"] == right["split"]:
                    continue
                hamming = hamming_distance(left.get("_validation_phash"), right.get("_validation_phash"))
                if hamming is not None and hamming <= config.similarity.image_phash_hamming_threshold:
                    overlap_count += 1
            if comparisons > config.similarity.max_pair_comparisons_per_bucket:
                break
    return overlap_count


def _cross_dataset_overlap(records: list[dict[str, Any]], config: SplitBuilderConfig) -> int:
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("image_path"):
            continue
        vector = resolve_embedding(record, kind="image")
        record["_validation_image_embedding"] = vector
        bucketed[vector_signature(vector)].append(record)
    count = 0
    for bucket in bucketed.values():
        if len(bucket) > config.similarity.max_bucket_size:
            bucket = sorted(bucket, key=lambda item: item["sample_id"])[: config.similarity.max_bucket_size]
        comparisons = 0
        for index, left in enumerate(bucket):
            for right in bucket[index + 1 :]:
                comparisons += 1
                if comparisons > config.similarity.max_pair_comparisons_per_bucket:
                    break
                if left["split"] == right["split"]:
                    continue
                if left.get("dataset_source") == right.get("dataset_source"):
                    continue
                similarity = cosine_similarity(left["_validation_image_embedding"], right["_validation_image_embedding"])
                if similarity >= config.similarity.image_similarity_threshold:
                    count += 1
            if comparisons > config.similarity.max_pair_comparisons_per_bucket:
                break
    return count


def validate_split_assignments(records: list[dict[str, Any]], config: SplitBuilderConfig) -> dict[str, Any]:
    for record in records:
        record["image_hash"] = record.get("image_hash") or record.get("_image_hash") or sha256_file(record.get("image_path"))
        record["audio_hash"] = record.get("audio_hash") or record.get("_audio_hash") or sha256_file(record.get("audio_path"))
        record["text_hash"] = record.get("text_hash") or sha256_text(record.get("text"))

    identity_overlap_count, identity_affected_samples = _overlap_counts(records, "identity_id")
    speaker_overlap_count, speaker_affected_samples = _overlap_counts(records, "speaker_id")
    exact_image_overlap = _exact_overlap(records, "image_hash")
    audio_overlap_metrics = _audio_overlap_metrics(records, config)
    exact_audio_overlap = audio_overlap_metrics["raw_exact_audio_duplicate_overlap"]
    near_duplicate_overlap = _near_duplicate_overlap(records, config)
    cross_dataset_overlap = _cross_dataset_overlap(records, config)

    total_records = len(records)
    split_sizes = {split: len(items) for split, items in _records_by_split(records).items()}
    emotion_distribution: dict[str, dict[str, int]] = defaultdict(dict)
    dataset_distribution: dict[str, dict[str, int]] = defaultdict(dict)
    identities_per_split: dict[str, int] = defaultdict(int)
    speakers_per_split: dict[str, int] = defaultdict(int)
    for split_name, split_records in _records_by_split(records).items():
        emotion_distribution[split_name] = dict(Counter(record.get("unified_emotion") for record in split_records))
        dataset_distribution[split_name] = dict(Counter(record.get("dataset_source") for record in split_records if record.get("dataset_source")))
        identities_per_split[split_name] = len({record.get("identity_id") for record in split_records if record.get("identity_id")})
        speakers_per_split[split_name] = len({record.get("speaker_id") for record in split_records if record.get("speaker_id")})

    identity_leakage_percent = 0.0 if total_records == 0 else round(100.0 * identity_affected_samples / total_records, 4)
    speaker_leakage_percent = 0.0 if total_records == 0 else round(100.0 * speaker_affected_samples / total_records, 4)
    status = "PASS"
    if any(
        [
            identity_overlap_count > 0,
            speaker_overlap_count > 0,
            exact_image_overlap > 0,
            audio_overlap_metrics["actionable_exact_audio_duplicate_overlap"] > 0,
            near_duplicate_overlap > 0,
            cross_dataset_overlap > 0,
        ]
    ):
        status = "WARNING"
    non_empty_splits = sum(1 for split_name in ("train", "val", "test") if split_sizes.get(split_name, 0) > 0)
    if non_empty_splits < 3:
        status = "WARNING"

    return {
        "identity_leakage_percent": identity_leakage_percent,
        "speaker_leakage_percent": speaker_leakage_percent,
        "identity_overlap_count": identity_overlap_count,
        "speaker_overlap_count": speaker_overlap_count,
        "exact_image_duplicate_overlap": exact_image_overlap,
        "exact_audio_duplicate_overlap": exact_audio_overlap,
        "actionable_exact_audio_duplicate_overlap": audio_overlap_metrics["actionable_exact_audio_duplicate_overlap"],
        "ambiguous_audio_reuse_overlap_count": audio_overlap_metrics["ambiguous_audio_reuse_overlap_count"],
        "near_duplicate_overlap_count": near_duplicate_overlap,
        "cross_dataset_overlap_count": cross_dataset_overlap,
        "non_empty_split_count": non_empty_splits,
        "split_size_counts": split_sizes,
        "emotion_distribution_per_split": dict(emotion_distribution),
        "dataset_source_distribution_per_split": dict(dataset_distribution),
        "identities_per_split": dict(identities_per_split),
        "speakers_per_split": dict(speakers_per_split),
        "status": status,
    }