"""Duplicate, near-duplicate, and cross-dataset overlap detection."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

try:
    from .embedding_utils import (
        audio_embedding,
        audio_exact_hash,
        batch_iterable,
        cosine_similarity,
        hamming_distance,
        image_embedding,
        image_exact_hash,
        image_perceptual_hash,
        normalize_text,
        percentage,
        representative_examples,
        sha256_text,
        text_embedding,
        vector_signature,
    )
except ImportError:
    from embedding_utils import (
        audio_embedding,
        audio_exact_hash,
        batch_iterable,
        cosine_similarity,
        hamming_distance,
        image_embedding,
        image_exact_hash,
        image_perceptual_hash,
        normalize_text,
        percentage,
        representative_examples,
        sha256_text,
        text_embedding,
        vector_signature,
    )


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    return int(config.get(key, default))


def _record_duplicate_group(label: str, group: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        label: group[0].get(label),
        "splits": sorted({sample["split"] for sample in group}),
        "sample_ids": [sample["sample_id"] for sample in group],
    }


def detect_exact_image_duplicates(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        for image_field in ("image_path", "reference_image_path"):
            image_path = sample.get(image_field)
            if not image_path:
                continue
            exact_hash = image_exact_hash(image_path)
            if not exact_hash:
                continue
            grouped[exact_hash].append({**sample, "image_hash": exact_hash, "image_field": image_field})

    duplicates: list[dict[str, Any]] = []
    affected_sample_ids: set[str] = set()
    for exact_hash, group in grouped.items():
        splits = {sample["split"] for sample in group}
        if len(group) < 2 or len(splits) < 2:
            continue
        affected_sample_ids.update(sample["sample_id"] for sample in group)
        duplicates.append(
            {
                "image_hash": exact_hash,
                "splits": sorted(splits),
                "sample_ids": [sample["sample_id"] for sample in group],
                "image_paths": [sample[image_field] for sample, image_field in [(item, item["image_field"]) for item in group]],
            }
        )
    return {
        "count": len(duplicates),
        "affected_samples": len(affected_sample_ids),
        "examples": representative_examples(duplicates),
        "items": duplicates,
    }


def detect_near_duplicate_images(samples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    candidates = [sample for sample in samples if sample.get("image_path")]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batch_iterable(candidates, int(config["batch_size"])):
        for sample in batch:
            phash = image_perceptual_hash(sample.get("image_path"))
            if not phash:
                continue
            sample["_image_phash"] = phash
            sample["_image_embedding"] = image_embedding(sample.get("image_path"), dimension=int(config["image_embedding_dim"]))
            buckets[phash[: int(config["image_phash_prefix_length"]) ]].append(sample)

    matches: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    affected_sample_ids: set[str] = set()
    for bucket_samples in buckets.values():
        max_bucket_size = _config_int(config, "max_bucket_size", 256)
        max_pair_comparisons = _config_int(config, "max_pair_comparisons_per_bucket", 4096)
        if len(bucket_samples) > max_bucket_size:
            bucket_samples = sorted(bucket_samples, key=lambda item: item["sample_id"])[:max_bucket_size]
        comparisons = 0
        for index, left in enumerate(bucket_samples):
            for right in bucket_samples[index + 1 :]:
                comparisons += 1
                if comparisons > max_pair_comparisons:
                    break
                if left["split"] == right["split"]:
                    continue
                pair_key = tuple(sorted((left["sample_id"], right["sample_id"])))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                hamming = hamming_distance(left.get("_image_phash"), right.get("_image_phash"))
                similarity = cosine_similarity(left["_image_embedding"], right["_image_embedding"])
                if hamming is None:
                    continue
                if hamming <= int(config["image_phash_hamming_threshold"]) or similarity >= float(config["image_similarity_threshold"]):
                    affected_sample_ids.update(pair_key)
                    matches.append(
                        {
                            "left_sample_id": left["sample_id"],
                            "right_sample_id": right["sample_id"],
                            "left_split": left["split"],
                            "right_split": right["split"],
                            "left_image_path": left.get("image_path"),
                            "right_image_path": right.get("image_path"),
                            "phash_hamming_distance": hamming,
                            "embedding_similarity": round(similarity, 6),
                        }
                    )
            if comparisons > max_pair_comparisons:
                break
    return {
        "count": len(matches),
        "affected_samples": len(affected_sample_ids),
        "examples": representative_examples(matches),
        "items": matches,
    }


def detect_audio_duplicates(samples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    exact_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates = [sample for sample in samples if sample.get("audio_path")]
    for sample in candidates:
        exact_hash = audio_exact_hash(sample.get("audio_path"))
        sample["audio_exact_hash"] = exact_hash
        if exact_hash:
            exact_grouped[exact_hash].append(sample)

    exact_duplicates: list[dict[str, Any]] = []
    affected_sample_ids: set[str] = set()
    for exact_hash, group in exact_grouped.items():
        splits = {sample["split"] for sample in group}
        if len(group) < 2 or len(splits) < 2:
            continue
        affected_sample_ids.update(sample["sample_id"] for sample in group)
        exact_duplicates.append(
            {
                "audio_hash": exact_hash,
                "splits": sorted(splits),
                "sample_ids": [sample["sample_id"] for sample in group],
            }
        )

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batch_iterable(candidates, int(config["batch_size"])):
        for sample in batch:
            embedding = audio_embedding(
                sample.get("audio_path"),
                feature_path=sample.get("audio_feature_path"),
                dimension=int(config["audio_embedding_dim"]),
            )
            sample["_audio_embedding"] = embedding
            buckets[vector_signature(embedding)].append(sample)

    near_duplicates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for bucket_samples in buckets.values():
        max_bucket_size = _config_int(config, "max_bucket_size", 256)
        max_pair_comparisons = _config_int(config, "max_pair_comparisons_per_bucket", 4096)
        if len(bucket_samples) > max_bucket_size:
            bucket_samples = sorted(bucket_samples, key=lambda item: item["sample_id"])[:max_bucket_size]
        comparisons = 0
        for index, left in enumerate(bucket_samples):
            for right in bucket_samples[index + 1 :]:
                comparisons += 1
                if comparisons > max_pair_comparisons:
                    break
                if left["split"] == right["split"]:
                    continue
                pair_key = tuple(sorted((left["sample_id"], right["sample_id"])))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                similarity = cosine_similarity(left["_audio_embedding"], right["_audio_embedding"])
                if similarity >= float(config["audio_similarity_threshold"]):
                    affected_sample_ids.update(pair_key)
                    near_duplicates.append(
                        {
                            "left_sample_id": left["sample_id"],
                            "right_sample_id": right["sample_id"],
                            "left_split": left["split"],
                            "right_split": right["split"],
                            "embedding_similarity": round(similarity, 6),
                        }
                    )
            if comparisons > max_pair_comparisons:
                break
    return {
        "exact_duplicates": {
            "count": len(exact_duplicates),
            "items": exact_duplicates,
            "examples": representative_examples(exact_duplicates),
        },
        "near_duplicates": {
            "count": len(near_duplicates),
            "items": near_duplicates,
            "examples": representative_examples(near_duplicates),
        },
        "affected_samples": len(affected_sample_ids),
        "leakage_percent": percentage(len(affected_sample_ids), len(samples)),
    }


def detect_text_duplicates(samples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    exact_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        normalized = normalize_text(sample.get("text"))
        if not normalized:
            continue
        sample["_normalized_text"] = normalized
        text_hash = sha256_text(normalized)
        sample["_text_hash"] = text_hash
        exact_grouped[text_hash].append(sample)

    exact_duplicates: list[dict[str, Any]] = []
    affected_sample_ids: set[str] = set()
    exact_duplicate_hashes: set[str] = set()
    for text_hash, group in exact_grouped.items():
        splits = {sample["split"] for sample in group}
        if len(group) < 2 or len(splits) < 2:
            continue
        exact_duplicate_hashes.add(text_hash)
        affected_sample_ids.update(sample["sample_id"] for sample in group)
        exact_duplicates.append(
            {
                "text_hash": text_hash,
                "splits": sorted(splits),
                "sample_ids": [sample["sample_id"] for sample in group],
                "text_preview": group[0]["_normalized_text"][:160],
            }
        )

    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        normalized = sample.get("_normalized_text")
        if not normalized:
            continue
        if sample.get("_text_hash") in exact_duplicate_hashes:
            continue
        tokens = normalized.split()
        if not tokens:
            continue
        sample["_text_embedding"] = text_embedding(normalized, dimension=int(config["text_embedding_dim"]))
        first_token = tokens[0]
        second_token = tokens[1] if len(tokens) > 1 else ""
        buckets[(first_token, second_token, min(len(tokens) // 8, 20), sample.get("unified_emotion") or "")].append(sample)

    fuzzy_duplicates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for bucket_samples in buckets.values():
        max_bucket_size = _config_int(config, "max_bucket_size", 256)
        max_pair_comparisons = _config_int(config, "max_pair_comparisons_per_bucket", 4096)
        if len(bucket_samples) > max_bucket_size:
            bucket_samples = sorted(bucket_samples, key=lambda item: item["sample_id"])[:max_bucket_size]
        comparisons = 0
        for index, left in enumerate(bucket_samples):
            for right in bucket_samples[index + 1 :]:
                comparisons += 1
                if comparisons > max_pair_comparisons:
                    break
                if left["split"] == right["split"]:
                    continue
                pair_key = tuple(sorted((left["sample_id"], right["sample_id"])))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                similarity = cosine_similarity(left["_text_embedding"], right["_text_embedding"])
                ratio = SequenceMatcher(None, left["_normalized_text"], right["_normalized_text"]).ratio()
                if similarity >= float(config["text_similarity_threshold"]) or ratio >= float(config["text_fuzzy_ratio_threshold"]):
                    affected_sample_ids.update(pair_key)
                    fuzzy_duplicates.append(
                        {
                            "left_sample_id": left["sample_id"],
                            "right_sample_id": right["sample_id"],
                            "left_split": left["split"],
                            "right_split": right["split"],
                            "embedding_similarity": round(similarity, 6),
                            "fuzzy_ratio": round(ratio, 6),
                            "left_text_preview": left["_normalized_text"][:120],
                            "right_text_preview": right["_normalized_text"][:120],
                        }
                    )
            if comparisons > max_pair_comparisons:
                break
    return {
        "exact_duplicates": {
            "count": len(exact_duplicates),
            "items": exact_duplicates,
            "examples": representative_examples(exact_duplicates),
        },
        "fuzzy_duplicates": {
            "count": len(fuzzy_duplicates),
            "items": fuzzy_duplicates,
            "examples": representative_examples(fuzzy_duplicates),
        },
        "affected_samples": len(affected_sample_ids),
        "leakage_percent": percentage(len(affected_sample_ids), len(samples)),
    }


def detect_cross_dataset_overlap_from_matches(
    samples: list[dict[str, Any]],
    image_matches: list[dict[str, Any]],
    audio_matches: list[dict[str, Any]],
    text_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_index = {sample["sample_id"]: sample for sample in samples}
    overlaps: list[dict[str, Any]] = []
    normalized_matches: list[dict[str, Any]] = []
    for match in image_matches + audio_matches + text_matches:
        if "left_sample_id" in match and "right_sample_id" in match:
            normalized_matches.append(match)
            continue
        sample_ids = match.get("sample_ids") or []
        for index, left_sample_id in enumerate(sample_ids):
            for right_sample_id in sample_ids[index + 1 :]:
                normalized_matches.append({"left_sample_id": left_sample_id, "right_sample_id": right_sample_id})

    for match in normalized_matches:
        left = sample_index.get(match["left_sample_id"])
        right = sample_index.get(match["right_sample_id"])
        if left is None or right is None:
            continue
        left_sources = {
            left.get("image_dataset_source"),
            left.get("audio_dataset_source"),
            left.get("text_dataset_source"),
            left.get("dataset_source"),
        }
        right_sources = {
            right.get("image_dataset_source"),
            right.get("audio_dataset_source"),
            right.get("text_dataset_source"),
            right.get("dataset_source"),
        }
        if left_sources == right_sources:
            continue
        overlaps.append(
            {
                "left_sample_id": left["sample_id"],
                "right_sample_id": right["sample_id"],
                "left_sources": sorted(source for source in left_sources if source),
                "right_sources": sorted(source for source in right_sources if source),
            }
        )
    unique_pairs = {(item["left_sample_id"], item["right_sample_id"]) for item in overlaps}
    return {
        "count": len(unique_pairs),
        "items": overlaps,
        "examples": representative_examples(overlaps),
    }


def run_duplicate_checks(samples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    image_exact = detect_exact_image_duplicates(samples)
    image_near = detect_near_duplicate_images(samples, config)
    audio = detect_audio_duplicates(samples, config)
    text = detect_text_duplicates(samples, config)
    cross_dataset = detect_cross_dataset_overlap_from_matches(
        samples,
        image_near["items"],
        audio["near_duplicates"]["items"],
        text["fuzzy_duplicates"]["items"] + text["exact_duplicates"]["items"],
    )
    return {
        "image_duplicates": {
            "exact_duplicates": image_exact,
            "near_duplicates": image_near,
            "count": image_exact["count"] + image_near["count"],
        },
        "audio_duplicates": audio,
        "text_duplicates": text,
        "cross_dataset_overlap": cross_dataset,
    }