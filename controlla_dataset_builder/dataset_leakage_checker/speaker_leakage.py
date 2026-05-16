"""Speaker and cross-modal leakage detection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

try:
    from .embedding_utils import (
        audio_embedding,
        batch_iterable,
        cosine_similarity,
        normalize_text,
        percentage,
        representative_examples,
        sha256_text,
        vector_signature,
    )
except ImportError:
    from embedding_utils import (
        audio_embedding,
        batch_iterable,
        cosine_similarity,
        normalize_text,
        percentage,
        representative_examples,
        sha256_text,
        vector_signature,
    )


def detect_speaker_id_leakage(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        speaker_id = sample.get("speaker_id")
        if speaker_id:
            grouped[str(speaker_id)].append(sample)

    leaks: list[dict[str, Any]] = []
    affected_sample_ids: set[str] = set()
    for speaker_id, group in grouped.items():
        splits = sorted({sample["split"] for sample in group})
        if len(splits) < 2:
            continue
        affected_sample_ids.update(sample["sample_id"] for sample in group)
        leaks.append(
            {
                "speaker_id": speaker_id,
                "splits": splits,
                "sample_ids": [sample["sample_id"] for sample in group],
                "dataset_sources": sorted({sample.get("audio_dataset_source") or sample.get("dataset_source") for sample in group if sample.get("audio_dataset_source") or sample.get("dataset_source")}),
            }
        )
    return {
        "leaked_speakers": leaks,
        "num_leaked_speakers": len(leaks),
        "affected_samples": len(affected_sample_ids),
        "leakage_percent": percentage(len(affected_sample_ids), len(samples)),
        "examples": representative_examples(leaks),
    }


def detect_speaker_embedding_leakage(samples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    fallback_samples = [sample for sample in samples if not sample.get("speaker_id") and sample.get("audio_path")]
    if not fallback_samples:
        return {
            "num_cross_split_clusters": 0,
            "affected_samples": 0,
            "leakage_percent": 0.0,
            "clusters": [],
            "examples": [],
        }

    threshold = float(config["speaker_similarity_threshold"])
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batch_iterable(fallback_samples, int(config["batch_size"])):
        for sample in batch:
            embedding = audio_embedding(
                sample.get("audio_path"),
                feature_path=sample.get("audio_feature_path"),
                dimension=int(config["audio_embedding_dim"]),
            )
            sample["_speaker_embedding"] = embedding
            bucketed[vector_signature(embedding)].append(sample)

    clusters: list[dict[str, Any]] = []
    affected_sample_ids: set[str] = set()
    cluster_index = 0
    for bucket_samples in bucketed.values():
        if len(bucket_samples) < 2:
            continue
        used: set[str] = set()
        for anchor in bucket_samples:
            if anchor["sample_id"] in used:
                continue
            members = [anchor]
            used.add(anchor["sample_id"])
            for candidate in bucket_samples:
                if candidate["sample_id"] in used:
                    continue
                similarity = cosine_similarity(anchor["_speaker_embedding"], candidate["_speaker_embedding"])
                if similarity >= threshold:
                    members.append(candidate)
                    used.add(candidate["sample_id"])
            splits = sorted({sample["split"] for sample in members})
            if len(members) < 2 or len(splits) < 2:
                continue
            cluster_index += 1
            affected_sample_ids.update(sample["sample_id"] for sample in members)
            clusters.append(
                {
                    "cluster_id": f"speaker_cluster_{cluster_index}",
                    "splits": splits,
                    "sample_ids": [sample["sample_id"] for sample in members],
                    "audio_paths": [str(sample.get("audio_path")) for sample in members],
                }
            )

    return {
        "num_cross_split_clusters": len(clusters),
        "affected_samples": len(affected_sample_ids),
        "leakage_percent": percentage(len(affected_sample_ids), len(samples)),
        "clusters": clusters,
        "examples": representative_examples(clusters),
    }


def detect_cross_modal_leakage(samples: list[dict[str, Any]]) -> dict[str, Any]:
    identity_emotion: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    speaker_emotion: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    text_emotion: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    audio_emotion: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        emotion = str(sample.get("unified_emotion") or "")
        if sample.get("identity_id"):
            identity_emotion[(str(sample["identity_id"]), emotion)].append(sample)
        if sample.get("speaker_id"):
            speaker_emotion[(str(sample["speaker_id"]), emotion)].append(sample)
        normalized_text = normalize_text(sample.get("text"))
        if normalized_text:
            text_emotion[(sha256_text(normalized_text), emotion)].append(sample)
        if sample.get("audio_exact_hash"):
            audio_emotion[(str(sample["audio_exact_hash"]), emotion)].append(sample)

    def collect(name: str, grouped: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
        overlaps: list[dict[str, Any]] = []
        affected_sample_ids: set[str] = set()
        for (key, emotion), group in grouped.items():
            splits = sorted({sample["split"] for sample in group})
            if len(splits) < 2:
                continue
            affected_sample_ids.update(sample["sample_id"] for sample in group)
            overlaps.append(
                {
                    name: key,
                    "emotion": emotion,
                    "splits": splits,
                    "sample_ids": [sample["sample_id"] for sample in group],
                }
            )
        return {
            "count": len(overlaps),
            "affected_samples": len(affected_sample_ids),
            "examples": representative_examples(overlaps),
            "items": overlaps,
        }

    return {
        "identity_emotion_overlap": collect("identity_id", identity_emotion),
        "speaker_emotion_overlap": collect("speaker_id", speaker_emotion),
        "text_emotion_overlap": collect("text_hash", text_emotion),
        "audio_emotion_overlap": collect("audio_hash", audio_emotion),
    }


def run_speaker_and_cross_modal_checks(samples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    explicit = detect_speaker_id_leakage(samples)
    fallback = detect_speaker_embedding_leakage(samples, config)
    cross_modal = detect_cross_modal_leakage(samples)
    affected = max(explicit["affected_samples"], fallback["affected_samples"])
    return {
        "speaker_leakage": {
            "explicit_speaker_id": explicit,
            "fallback_embedding_clusters": fallback,
            "affected_samples": affected,
            "leakage_percent": percentage(affected, len(samples)),
        },
        "cross_modal_leakage": cross_modal,
    }