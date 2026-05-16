"""Identity leakage detection for multimodal datasets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

try:
    from .embedding_utils import (
        batch_iterable,
        cosine_similarity,
        image_embedding,
        percentage,
        representative_examples,
        vector_signature,
    )
except ImportError:
    from embedding_utils import (
        batch_iterable,
        cosine_similarity,
        image_embedding,
        percentage,
        representative_examples,
        vector_signature,
    )


def _sample_image_path(sample: dict[str, Any]) -> str | None:
    return sample.get("image_path") or sample.get("reference_image_path")


def detect_identity_id_leakage(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        identity_id = sample.get("identity_id")
        if identity_id:
            grouped[str(identity_id)].append(sample)

    leaked_identities: list[dict[str, Any]] = []
    affected_sample_ids: set[str] = set()
    for identity_id, group in grouped.items():
        splits = sorted({sample["split"] for sample in group})
        if len(splits) < 2:
            continue
        affected_sample_ids.update(sample["sample_id"] for sample in group)
        leaked_identities.append(
            {
                "identity_id": identity_id,
                "splits": splits,
                "sample_ids": [sample["sample_id"] for sample in group],
                "dataset_sources": sorted({sample.get("image_dataset_source") or sample.get("dataset_source") for sample in group if sample.get("image_dataset_source") or sample.get("dataset_source")}),
            }
        )

    return {
        "leaked_identities": leaked_identities,
        "num_leaked_identities": len(leaked_identities),
        "affected_samples": len(affected_sample_ids),
        "leakage_percent": percentage(len(affected_sample_ids), len(samples)),
        "examples": representative_examples(leaked_identities),
    }


def detect_identity_embedding_leakage(samples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    fallback_samples = [sample for sample in samples if not sample.get("identity_id") and _sample_image_path(sample)]
    if not fallback_samples:
        return {
            "num_cross_split_clusters": 0,
            "affected_samples": 0,
            "leakage_percent": 0.0,
            "clusters": [],
            "examples": [],
        }

    threshold = float(config["identity_similarity_threshold"])
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batch_iterable(fallback_samples, int(config["batch_size"])):
        for sample in batch:
            embedding = image_embedding(_sample_image_path(sample), dimension=int(config["image_embedding_dim"]))
            sample["_identity_embedding"] = embedding
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
            cluster_members = [anchor]
            used.add(anchor["sample_id"])
            for candidate in bucket_samples:
                if candidate["sample_id"] in used:
                    continue
                similarity = cosine_similarity(anchor["_identity_embedding"], candidate["_identity_embedding"])
                if similarity >= threshold:
                    cluster_members.append(candidate)
                    used.add(candidate["sample_id"])
            splits = sorted({sample["split"] for sample in cluster_members})
            if len(cluster_members) < 2 or len(splits) < 2:
                continue
            cluster_index += 1
            affected_sample_ids.update(sample["sample_id"] for sample in cluster_members)
            clusters.append(
                {
                    "cluster_id": f"identity_cluster_{cluster_index}",
                    "splits": splits,
                    "sample_ids": [sample["sample_id"] for sample in cluster_members],
                    "image_paths": [str(_sample_image_path(sample)) for sample in cluster_members],
                }
            )

    return {
        "num_cross_split_clusters": len(clusters),
        "affected_samples": len(affected_sample_ids),
        "leakage_percent": percentage(len(affected_sample_ids), len(samples)),
        "clusters": clusters,
        "examples": representative_examples(clusters),
    }


def run_identity_leakage_check(samples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    explicit = detect_identity_id_leakage(samples)
    fallback = detect_identity_embedding_leakage(samples, config)
    affected = max(explicit["affected_samples"], fallback["affected_samples"])
    return {
        "explicit_identity_id": explicit,
        "fallback_embedding_clusters": fallback,
        "affected_samples": affected,
        "leakage_percent": percentage(affected, len(samples)),
    }