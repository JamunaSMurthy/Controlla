"""Hard-edge graph construction and connected-component grouping."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import SplitBuilderConfig
from .similarity_utils import image_perceptual_hash, normalize_text, sha256_file


class UnionFind:
    """Deterministic union-find structure for hard leakage grouping."""

    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _link_group(
    uf: UnionFind,
    sample_ids: list[str],
    hard_edges: list[dict[str, Any]],
    edge_counts: Counter[str],
    *,
    rule: str,
    value: str,
) -> None:
    if len(sample_ids) < 2:
        return
    anchor = sample_ids[0]
    for sample_id in sample_ids[1:]:
        if uf.union(anchor, sample_id):
            hard_edges.append({"left": anchor, "right": sample_id, "rule": rule, "value": value})
            edge_counts[rule] += 1


def _explicit_metadata_links(sample: dict[str, Any]) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for key in (
        "reference_sample_id",
        "target_sample_id",
        "source_sample_id",
        "pair_id",
        "pair_group_id",
        "paired_sample_id",
        "reference_pair_id",
        "target_pair_id",
    ):
        value = sample.get(key)
        if value:
            links.append((key, str(value)))
    return links


def _memoized_sha256_file(path: str | None, hash_cache: dict[str, str | None]) -> str | None:
    if not path:
        return None
    cache_key = str(Path(path))
    if cache_key not in hash_cache:
        hash_cache[cache_key] = sha256_file(path)
    return hash_cache[cache_key]


def _annotate_hard_keys(samples: list[dict[str, Any]]) -> dict[str, int]:
    hash_cache: dict[str, str | None] = {}
    image_phash_paths: set[str] = set()
    face_phash_paths: set[str] = set()
    for sample in samples:
        sample["_image_hash"] = sample.get("image_hash") or _memoized_sha256_file(sample.get("image_path"), hash_cache)
        sample["_audio_hash"] = sample.get("audio_hash") or _memoized_sha256_file(sample.get("audio_path"), hash_cache)
        sample["_reference_hash"] = _memoized_sha256_file(sample.get("reference_image_path"), hash_cache)
        image_path = sample.get("image_path")
        face_path = sample.get("image_path") or sample.get("reference_image_path")
        sample["_image_phash"] = sample.get("image_phash") or image_perceptual_hash(image_path)
        sample["_face_phash"] = sample.get("face_phash") or sample.get("_image_phash") or image_perceptual_hash(face_path)
        sample["_explicit_metadata_links"] = _explicit_metadata_links(sample)
        if image_path:
            image_phash_paths.add(str(Path(image_path)))
        if face_path:
            face_phash_paths.add(str(Path(face_path)))
    return {
        "memoized_file_hash_paths": len(hash_cache),
        "precomputed_image_phash_paths": len(image_phash_paths),
        "precomputed_face_phash_paths": len(face_phash_paths),
    }


def _audio_hash_bucket_stats(bucket: list[dict[str, Any]]) -> dict[str, int]:
    normalized_texts = {
        normalize_text(sample.get("text"))
        for sample in bucket
        if normalize_text(sample.get("text"))
    }
    distinct_identities = {
        str(sample.get("identity_id"))
        for sample in bucket
        if sample.get("identity_id") is not None and str(sample.get("identity_id"))
    }
    distinct_speakers = {
        str(sample.get("speaker_id"))
        for sample in bucket
        if sample.get("speaker_id") is not None and str(sample.get("speaker_id"))
    }
    return {
        "size": len(bucket),
        "distinct_identities": len(distinct_identities),
        "distinct_speakers": len(distinct_speakers),
        "distinct_texts": len(normalized_texts),
    }


def _should_hard_link_audio_hash_bucket(
    bucket: list[dict[str, Any]],
    config: SplitBuilderConfig,
) -> tuple[bool, str | None, dict[str, int]]:
    stats = _audio_hash_bucket_stats(bucket)
    if not config.grouping.enable_audio_hash_hard_grouping:
        return False, "disabled", stats
    if stats["size"] > config.grouping.audio_hash_max_group_size_for_hard_link:
        return False, "group_size", stats
    if stats["distinct_identities"] > config.grouping.audio_hash_max_distinct_identities_for_hard_link:
        return False, "distinct_identities", stats
    if stats["distinct_texts"] > config.grouping.audio_hash_max_distinct_texts_for_hard_link:
        return False, "distinct_texts", stats
    return True, None, stats


def _build_hard_edges(samples: list[dict[str, Any]], uf: UnionFind, config: SplitBuilderConfig) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    hard_edges: list[dict[str, Any]] = []
    edge_counts: Counter[str] = Counter()
    annotation_stats = _annotate_hard_keys(samples)

    by_sample_id: dict[str, list[str]] = defaultdict(list)
    by_identity: dict[str, list[str]] = defaultdict(list)
    by_speaker: dict[str, list[str]] = defaultdict(list)
    by_image_hash: dict[str, list[str]] = defaultdict(list)
    by_audio_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_explicit_pair_metadata: dict[str, list[str]] = defaultdict(list)
    by_explicit_reference_target_metadata: dict[str, list[str]] = defaultdict(list)
    skipped_bucket_counts: Counter[str] = Counter()
    skipped_audio_hash_reasons: Counter[str] = Counter()
    skipped_audio_hash_records = 0
    skipped_audio_hash_examples: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = sample["sample_id"]
        by_sample_id[sample_id].append(sample_id)
        if sample.get("identity_id"):
            by_identity[str(sample["identity_id"])].append(sample_id)
        if sample.get("speaker_id"):
            by_speaker[str(sample["speaker_id"])].append(sample_id)
        if sample.get("_image_hash"):
            by_image_hash[str(sample["_image_hash"])].append(sample_id)
        if sample.get("_audio_hash"):
            by_audio_hash[str(sample["_audio_hash"])].append(sample)
        for link_key, link_value in sample.get("_explicit_metadata_links", []):
            if "pair" in link_key:
                by_explicit_pair_metadata[f"{link_key}:{link_value}"].append(sample_id)
            else:
                by_explicit_reference_target_metadata[f"{link_key}:{link_value}"].append(sample_id)

    for value, sample_ids in sorted(by_sample_id.items()):
        _link_group(uf, sorted(sample_ids), hard_edges, edge_counts, rule="sample_id", value=value)
    for value, sample_ids in sorted(by_identity.items()):
        _link_group(uf, sorted(sample_ids), hard_edges, edge_counts, rule="identity_id", value=value)
    for value, sample_ids in sorted(by_speaker.items()):
        _link_group(uf, sorted(sample_ids), hard_edges, edge_counts, rule="speaker_id", value=value)
    for value, sample_ids in sorted(by_image_hash.items()):
        _link_group(uf, sorted(sample_ids), hard_edges, edge_counts, rule="image_hash", value=value)
    for value, bucket in sorted(by_audio_hash.items()):
        should_link, skip_reason, stats = _should_hard_link_audio_hash_bucket(bucket, config)
        if should_link:
            _link_group(
                uf,
                sorted(sample["sample_id"] for sample in bucket),
                hard_edges,
                edge_counts,
                rule="audio_hash",
                value=value,
            )
            continue
        skipped_bucket_counts["audio_hash"] += 1
        if skip_reason:
            skipped_audio_hash_reasons[skip_reason] += 1
        skipped_audio_hash_records += stats["size"]
        if len(skipped_audio_hash_examples) < config.grouping.report_top_component_limit:
            skipped_audio_hash_examples.append(
                {
                    "audio_hash": value[:12],
                    "size": stats["size"],
                    "distinct_identities": stats["distinct_identities"],
                    "distinct_speakers": stats["distinct_speakers"],
                    "distinct_texts": stats["distinct_texts"],
                    "skip_reason": skip_reason,
                    "sample_ids": sorted(sample["sample_id"] for sample in bucket[:5]),
                }
            )
    for value, sample_ids in sorted(by_explicit_pair_metadata.items()):
        _link_group(uf, sorted(sample_ids), hard_edges, edge_counts, rule="explicit_pair_metadata", value=value)
    for value, sample_ids in sorted(by_explicit_reference_target_metadata.items()):
        _link_group(uf, sorted(sample_ids), hard_edges, edge_counts, rule="reference_target_link", value=value)

    diagnostics = {
        **annotation_stats,
        "skipped_bucket_counts_by_rule": dict(skipped_bucket_counts),
        "audio_hash_skip_reasons": dict(skipped_audio_hash_reasons),
        "skipped_audio_hash_record_count": skipped_audio_hash_records,
        "skipped_audio_hash_examples": skipped_audio_hash_examples,
    }
    return hard_edges, dict(edge_counts), diagnostics


def _component_analysis(groups: list[dict[str, Any]], total_samples: int, config: SplitBuilderConfig) -> dict[str, Any]:
    ordered_groups = sorted(groups, key=lambda item: (-item["size"], item["group_id"]))
    sizes = [group["size"] for group in ordered_groups]
    top_groups = []
    for group in ordered_groups[: config.grouping.report_top_component_limit]:
        top_groups.append(
            {
                "group_id": group["group_id"],
                "size": group["size"],
                "fraction_of_dataset": round(group["size"] / max(total_samples, 1), 6),
                "edge_counts_by_rule": group["edge_counts_by_rule"],
                "dataset_counts": group["dataset_counts"],
                "emotion_counts": group["emotion_counts"],
            }
        )

    groups_over_size_limit = sum(1 for group in groups if group["size"] > config.grouping.max_group_size)
    groups_over_fraction_limit = 0
    if total_samples >= config.grouping.min_total_samples_for_fraction_rule:
        groups_over_fraction_limit = sum(
            1 for group in groups if group["size"] / max(total_samples, 1) > config.grouping.max_group_fraction
        )

    return {
        "largest_hard_group_size": sizes[0] if sizes else 0,
        "largest_hard_group_fraction": 0.0 if not sizes else round(sizes[0] / max(total_samples, 1), 6),
        "median_hard_group_size": sizes[len(sizes) // 2] if sizes else 0,
        "top_hard_group_sizes": sizes[: config.grouping.report_top_component_limit],
        "num_groups_over_size_limit": groups_over_size_limit,
        "num_groups_over_fraction_limit": groups_over_fraction_limit,
        "largest_groups": top_groups,
    }


def build_groups(samples: list[dict[str, Any]], config: SplitBuilderConfig) -> dict[str, Any]:
    uf = UnionFind([sample["sample_id"] for sample in samples])
    hard_edges, edge_counts_by_rule, hard_edge_diagnostics = _build_hard_edges(samples, uf, config)

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        components[uf.find(sample["sample_id"])].append(sample)

    sample_to_group: dict[str, str] = {}
    group_edge_counts: dict[str, Counter[str]] = defaultdict(Counter)
    groups: list[dict[str, Any]] = []

    for index, root in enumerate(sorted(components), start=1):
        members = sorted(components[root], key=lambda item: item["sample_id"])
        group_id = f"group_{index:06d}"
        for member in members:
            sample_to_group[member["sample_id"]] = group_id

        emotion_counts = Counter(sample.get("unified_emotion") for sample in members)
        dataset_counts = Counter(sample.get("dataset_source") for sample in members if sample.get("dataset_source"))
        modality_counts = {
            "image": sum(int(bool(sample.get("has_image"))) for sample in members),
            "audio": sum(int(bool(sample.get("has_audio"))) for sample in members),
            "text": sum(int(bool(sample.get("has_text"))) for sample in members),
        }
        groups.append(
            {
                "group_id": group_id,
                "root_sample_id": root,
                "sample_ids": [sample["sample_id"] for sample in members],
                "samples": members,
                "size": len(members),
                "emotion_counts": dict(emotion_counts),
                "dataset_counts": dict(dataset_counts),
                "modality_counts": modality_counts,
                "existing_splits": sorted({sample["split"] for sample in members if sample.get("split")}),
                "edge_counts_by_rule": {},
            }
        )

    for edge in hard_edges:
        group_id = sample_to_group.get(edge["left"])
        if group_id:
            group_edge_counts[group_id][edge["rule"]] += 1

    for group in groups:
        group["edge_counts_by_rule"] = dict(group_edge_counts[group["group_id"]])

    analysis = _component_analysis(groups, len(samples), config)
    hard_group_report = {
        "num_input_samples": len(samples),
        "num_hard_groups": len(groups),
        "largest_hard_group_size": analysis["largest_hard_group_size"],
        "largest_hard_group_fraction": analysis["largest_hard_group_fraction"],
        "edge_counts_by_rule": edge_counts_by_rule,
        "hard_edge_diagnostics": hard_edge_diagnostics,
        "component_analysis": analysis,
    }

    return {
        "groups": groups,
        "hard_edges": hard_edges,
        "hard_edge_counts_by_rule": edge_counts_by_rule,
        "num_groups": len(groups),
        "component_analysis": analysis,
        "hard_group_report": hard_group_report,
    }
