"""Post-split soft overlap analysis used for warnings and limited repair."""

from __future__ import annotations

from collections import Counter, defaultdict
from time import perf_counter
from typing import Any

from .build_groups import _should_hard_link_audio_hash_bucket
from .config import SplitBuilderConfig
from .similarity_utils import cosine_similarity, hamming_distance, image_perceptual_hash, resolve_embedding, sha256_text, vector_signature


def _group_conflict_by_type_factory() -> defaultdict[str, Counter]:
    return defaultdict(Counter)


class _DisjointSet:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        if item not in self._parent:
            self._parent[item] = item

    def find(self, item: str) -> str:
        self.add(item)
        parent = self._parent[item]
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def components(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in sorted(self._parent):
            grouped[self.find(item)].append(item)
        return list(grouped.values())


def _init_report() -> dict[str, Any]:
    return {
        "warning_counts_by_type": Counter(),
        "example_warnings": [],
        "group_warning_counts": defaultdict(Counter),
        "group_conflict_split_counts": defaultdict(Counter),
        "group_conflict_split_counts_by_type": defaultdict(_group_conflict_by_type_factory),
        "group_total_warning_counts": Counter(),
        "quarantine_candidate_groups": Counter(),
        "repair_components_by_type": defaultdict(list),
        "total_warning_count": 0,
        "analysis_timings_seconds": {},
        "analysis_stats": {},
    }


def _add_directional_group_warning(
    report: dict[str, Any],
    warning_type: str,
    *,
    group_id: str,
    split_name: str,
    conflicting_split: str,
    example_payload: dict[str, Any] | None,
    example_limit: int,
    mark_quarantine: bool = False,
) -> None:
    if split_name == conflicting_split:
        return
    report["warning_counts_by_type"][warning_type] += 1
    report["group_warning_counts"][group_id][warning_type] += 1
    report["group_conflict_split_counts"][group_id][conflicting_split] += 1
    report["group_conflict_split_counts_by_type"][group_id][warning_type][conflicting_split] += 1
    report["group_total_warning_counts"][group_id] += 1
    report["total_warning_count"] += 1
    if mark_quarantine:
        report["quarantine_candidate_groups"][group_id] += 1
    if example_payload and len(report["example_warnings"]) < example_limit:
        payload = {"type": warning_type, **example_payload}
        report["example_warnings"].append(payload)


def _add_warning(
    report: dict[str, Any],
    warning_type: str,
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    score: float | None = None,
    metadata: dict[str, Any] | None = None,
    example_limit: int,
) -> None:
    if left["split"] == right["split"]:
        return
    left_group = str(left["group_id"])
    right_group = str(right["group_id"])
    report["warning_counts_by_type"][warning_type] += 1
    report["group_warning_counts"][left_group][warning_type] += 1
    report["group_warning_counts"][right_group][warning_type] += 1
    report["group_conflict_split_counts"][left_group][right["split"]] += 1
    report["group_conflict_split_counts"][right_group][left["split"]] += 1
    report["group_conflict_split_counts_by_type"][left_group][warning_type][right["split"]] += 1
    report["group_conflict_split_counts_by_type"][right_group][warning_type][left["split"]] += 1
    report["group_total_warning_counts"][left_group] += 1
    report["group_total_warning_counts"][right_group] += 1
    report["total_warning_count"] += 1
    if len(report["example_warnings"]) < example_limit:
        payload = {
            "type": warning_type,
            "left_sample_id": left["sample_id"],
            "right_sample_id": right["sample_id"],
            "left_group_id": left_group,
            "right_group_id": right_group,
            "left_split": left["split"],
            "right_split": right["split"],
        }
        if score is not None:
            payload["score"] = round(float(score), 6)
        if metadata:
            payload.update(metadata)
        report["example_warnings"].append(payload)


def _record_analysis_pass(report: dict[str, Any], pass_name: str, *, elapsed_seconds: float, stats: dict[str, Any]) -> None:
    report["analysis_timings_seconds"][pass_name] = round(float(elapsed_seconds), 6)
    report["analysis_stats"][pass_name] = stats


def _group_entries_by_split(bucket: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in sorted(bucket, key=lambda item: (item["split"], item["group_id"], item["sample_id"])):
        split_name = str(record["split"])
        group_id = str(record["group_id"])
        if group_id not in grouped[split_name]:
            grouped[split_name][group_id] = {
                "count": 0,
                "example": record,
            }
        grouped[split_name][group_id]["count"] += 1
    return grouped


def _representative_example_payload(
    *,
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "left_sample_id": left_record["sample_id"],
        "right_sample_id": right_record["sample_id"],
        "left_group_id": str(left_record["group_id"]),
        "right_group_id": str(right_record["group_id"]),
        "left_split": left_record["split"],
        "right_split": right_record["split"],
    }
    if metadata:
        payload.update(metadata)
    return payload


def _emit_group_split_conflicts(
    report: dict[str, Any],
    *,
    warning_type: str,
    groups_by_split: dict[str, dict[str, dict[str, Any]]],
    example_limit: int,
    metadata_factory,
    mark_quarantine: bool = False,
) -> int:
    emitted = 0
    ordered_splits = sorted(groups_by_split)
    for split_name in ordered_splits:
        other_splits = [candidate for candidate in ordered_splits if candidate != split_name]
        for group_id, entry in sorted(groups_by_split[split_name].items()):
            for other_split in other_splits:
                if not groups_by_split[other_split]:
                    continue
                other_group_id, other_entry = next(iter(sorted(groups_by_split[other_split].items())))
                metadata = metadata_factory(
                    split_name=split_name,
                    group_id=group_id,
                    other_split=other_split,
                    other_group_id=other_group_id,
                    left_entry=entry,
                    right_entry=other_entry,
                )
                _add_directional_group_warning(
                    report,
                    warning_type,
                    group_id=group_id,
                    split_name=split_name,
                    conflicting_split=other_split,
                    example_payload=_representative_example_payload(
                        left_record=entry["example"],
                        right_record=other_entry["example"],
                        metadata=metadata,
                    ),
                    example_limit=example_limit,
                    mark_quarantine=mark_quarantine,
                )
                emitted += 1
    return emitted


def _build_repair_component_summary(
    records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    group_entries: dict[str, dict[str, Any]] = {}
    for record in records:
        group_id = str(record["group_id"])
        if group_id not in group_entries:
            group_entries[group_id] = {
                "split": str(record["split"]),
                "record_count": 0,
            }
        group_entries[group_id]["record_count"] += 1
    split_names = {entry["split"] for entry in group_entries.values()}
    if len(split_names) < 2:
        return None

    group_ids_by_split: dict[str, list[str]] = defaultdict(list)
    record_count_by_split: Counter[str] = Counter()
    group_count_by_split: Counter[str] = Counter()
    for group_id, entry in sorted(group_entries.items()):
        split_name = str(entry["split"])
        group_ids_by_split[split_name].append(group_id)
        group_count_by_split[split_name] += 1
        record_count_by_split[split_name] += int(entry["record_count"])

    summary: dict[str, Any] = {
        "group_ids": sorted(group_entries),
        "group_ids_by_split": {split_name: sorted(group_ids) for split_name, group_ids in group_ids_by_split.items()},
        "group_count_by_split": dict(group_count_by_split),
        "record_count_by_split": dict(record_count_by_split),
        "num_groups": len(group_entries),
        "num_records": int(sum(record_count_by_split.values())),
    }
    if metadata:
        summary.update(metadata)
    return summary


def _append_repair_component(
    report: dict[str, Any],
    warning_type: str,
    records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> bool:
    summary = _build_repair_component_summary(records, metadata=metadata)
    if summary is None:
        return False
    report["repair_components_by_type"][warning_type].append(summary)
    return True


def _record_pairwise_repair_components(
    report: dict[str, Any],
    *,
    warning_type: str,
    records_by_group: dict[str, list[dict[str, Any]]],
    disjoint_set: _DisjointSet,
) -> int:
    component_count = 0
    for group_ids in disjoint_set.components():
        component_records: list[dict[str, Any]] = []
        for group_id in group_ids:
            component_records.extend(records_by_group.get(group_id, []))
        if _append_repair_component(report, warning_type, component_records):
            component_count += 1
    return component_count


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "warning_counts_by_type": dict(report["warning_counts_by_type"]),
        "example_warnings": report["example_warnings"],
        "group_warning_counts": {group_id: dict(counts) for group_id, counts in report["group_warning_counts"].items()},
        "group_conflict_split_counts": {
            group_id: dict(counts) for group_id, counts in report["group_conflict_split_counts"].items()
        },
        "group_conflict_split_counts_by_type": {
            group_id: {warning_type: dict(counts) for warning_type, counts in warning_counts.items()}
            for group_id, warning_counts in report["group_conflict_split_counts_by_type"].items()
        },
        "group_total_warning_counts": dict(report["group_total_warning_counts"]),
        "quarantine_candidate_groups": dict(report["quarantine_candidate_groups"]),
        "repair_components_by_type": {
            warning_type: list(components) for warning_type, components in report["repair_components_by_type"].items()
        },
        "analysis_timings_seconds": dict(report["analysis_timings_seconds"]),
        "analysis_stats": dict(report["analysis_stats"]),
        "total_warning_count": int(report["total_warning_count"]),
    }


def _analyze_exact_text_duplicates(records: list[dict[str, Any]], report: dict[str, Any], config: SplitBuilderConfig) -> dict[str, Any]:
    if not config.enable_text_similarity:
        return {"enabled": False}
    by_text_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = {
        "enabled": True,
        "num_text_hashes": 0,
        "duplicate_buckets": 0,
        "cross_split_buckets": 0,
        "largest_bucket_size": 0,
        "max_group_span_per_bucket": 0,
        "directional_conflicts_emitted": 0,
    }
    for record in records:
        text_hash = record.get("text_hash") or sha256_text(record.get("text"))
        if text_hash:
            record["text_hash"] = text_hash
            by_text_hash[text_hash].append(record)
    for text_hash, bucket in by_text_hash.items():
        stats["num_text_hashes"] += 1
        if len(bucket) < 2:
            continue
        stats["duplicate_buckets"] += 1
        stats["largest_bucket_size"] = max(stats["largest_bucket_size"], len(bucket))
        groups_by_split = _group_entries_by_split(bucket)
        total_groups = sum(len(group_entries) for group_entries in groups_by_split.values())
        stats["max_group_span_per_bucket"] = max(stats["max_group_span_per_bucket"], total_groups)
        if len(groups_by_split) < 2:
            continue
        stats["cross_split_buckets"] += 1
        stats["directional_conflicts_emitted"] += _emit_group_split_conflicts(
            report,
            warning_type="text_duplicate",
            groups_by_split=groups_by_split,
            example_limit=config.report_example_limit,
            metadata_factory=lambda **kwargs: {"text_hash": text_hash},
        )
    return stats


def _audio_bucket_key(record: dict[str, Any]) -> str | None:
    return record.get("_audio_hash") or record.get("audio_hash") or record.get("audio_path")


def _analyze_exact_audio_duplicates(records: list[dict[str, Any]], report: dict[str, Any], config: SplitBuilderConfig) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = {
        "num_audio_buckets": 0,
        "duplicate_buckets": 0,
        "cross_split_buckets": 0,
        "actionable_cross_split_buckets": 0,
        "largest_bucket_size": 0,
        "directional_conflicts_emitted": 0,
        "repair_component_count": 0,
    }
    for record in records:
        key = _audio_bucket_key(record)
        if key:
            buckets[str(key)].append(record)
    for audio_key, bucket in buckets.items():
        stats["num_audio_buckets"] += 1
        if len(bucket) < 2:
            continue
        stats["duplicate_buckets"] += 1
        stats["largest_bucket_size"] = max(stats["largest_bucket_size"], len(bucket))
        groups_by_split = _group_entries_by_split(bucket)
        if len(groups_by_split) < 2:
            continue
        stats["cross_split_buckets"] += 1
        should_link, _, bucket_stats = _should_hard_link_audio_hash_bucket(bucket, config)
        if not should_link:
            continue
        stats["actionable_cross_split_buckets"] += 1
        stats["directional_conflicts_emitted"] += _emit_group_split_conflicts(
            report,
            warning_type="exact_audio_duplicate",
            groups_by_split=groups_by_split,
            example_limit=config.report_example_limit,
            metadata_factory=lambda **kwargs: {
                "audio_bucket": audio_key[:12],
                "bucket_size": bucket_stats["size"],
                "distinct_identities": bucket_stats["distinct_identities"],
                "distinct_texts": bucket_stats["distinct_texts"],
            },
        )
        if _append_repair_component(
            report,
            "exact_audio_duplicate",
            bucket,
            metadata={
                "audio_bucket": audio_key[:12],
                "bucket_size": bucket_stats["size"],
                "distinct_identities": bucket_stats["distinct_identities"],
                "distinct_texts": bucket_stats["distinct_texts"],
            },
        ):
            stats["repair_component_count"] += 1
    return stats


def _analyze_ambiguous_audio_reuse(records: list[dict[str, Any]], report: dict[str, Any], config: SplitBuilderConfig) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = {
        "num_audio_buckets": 0,
        "ambiguous_buckets": 0,
        "cross_split_ambiguous_buckets": 0,
        "largest_ambiguous_bucket": 0,
        "directional_conflicts_emitted": 0,
    }
    for record in records:
        key = _audio_bucket_key(record)
        if key:
            buckets[str(key)].append(record)
    for audio_key, bucket in buckets.items():
        stats["num_audio_buckets"] += 1
        if len(bucket) < 2:
            continue
        should_link, _, bucket_stats = _should_hard_link_audio_hash_bucket(bucket, config)
        if should_link:
            continue
        stats["ambiguous_buckets"] += 1
        stats["largest_ambiguous_bucket"] = max(stats["largest_ambiguous_bucket"], len(bucket))
        groups_by_split = _group_entries_by_split(bucket)
        if len(groups_by_split) < 2:
            continue
        stats["cross_split_ambiguous_buckets"] += 1
        stats["directional_conflicts_emitted"] += _emit_group_split_conflicts(
            report,
            warning_type="ambiguous_audio_reuse",
            groups_by_split=groups_by_split,
            example_limit=config.report_example_limit,
            metadata_factory=lambda **kwargs: {
                "audio_bucket": audio_key[:12],
                "bucket_size": bucket_stats["size"],
                "distinct_identities": bucket_stats["distinct_identities"],
                "distinct_texts": bucket_stats["distinct_texts"],
            },
            mark_quarantine=True,
        )
    return stats


def _has_real_embedding_candidate(record: dict[str, Any], *, kind: str) -> bool:
    if kind == "face":
        return any(record.get(key) for key in ("arcface_embedding_path", "identity_feature_path", "image_embedding_path"))
    if kind == "image":
        return any(record.get(key) for key in ("clip_embedding_path", "image_embedding_path"))
    return True


def _surrogate_image_path(record: dict[str, Any], *, kind: str) -> str | None:
    if kind == "face":
        return record.get("image_path") or record.get("reference_image_path")
    if kind == "image":
        return record.get("image_path")
    return None


def _precomputed_surrogate_phash(record: dict[str, Any], *, kind: str) -> str | None:
    if kind == "face":
        return record.get("_face_phash") or record.get("face_phash") or record.get("_image_phash") or record.get("image_phash")
    if kind == "image":
        return record.get("_image_phash") or record.get("image_phash")
    return None


def _analyze_phash_surrogate_overlap(
    records: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    kind: str,
    warning_type: str,
    config: SplitBuilderConfig,
    filter_fn,
    extra_warning_type: str | None = None,
    require_cross_dataset: bool = False,
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if filter_fn(record) and not _has_real_embedding_candidate(record, kind=kind) and _surrogate_image_path(record, kind=kind)
    ]
    if len(candidates) < 2:
        return {
            "candidates": len(candidates),
            "bucket_count": 0,
            "comparisons": 0,
            "warnings_added": 0,
            "repair_component_count": 0,
            "mode": "phash_surrogate",
        }

    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repair_components = _DisjointSet()
    warnings_before = report["total_warning_count"]
    comparisons = 0
    for record in candidates:
        phash = _precomputed_surrogate_phash(record, kind=kind)
        if not phash:
            path = _surrogate_image_path(record, kind=kind)
            phash = image_perceptual_hash(path)
        if not phash:
            continue
        record[f"_soft_{kind}_phash"] = phash
        bucketed[phash[:6]].append(record)
        records_by_group[str(record["group_id"])].append(record)

    for bucket in bucketed.values():
        if len(bucket) > config.similarity.max_bucket_size:
            bucket = sorted(bucket, key=lambda item: item["sample_id"])[: config.similarity.max_bucket_size]
        bucket_comparisons = 0
        for index, left in enumerate(bucket):
            for right in bucket[index + 1 :]:
                bucket_comparisons += 1
                comparisons += 1
                if bucket_comparisons > config.similarity.max_pair_comparisons_per_bucket:
                    break
                if left["split"] == right["split"]:
                    continue
                if require_cross_dataset and left.get("dataset_source") == right.get("dataset_source"):
                    continue
                hamming = hamming_distance(left.get(f"_soft_{kind}_phash"), right.get(f"_soft_{kind}_phash"))
                if hamming is None or hamming > config.similarity.image_phash_hamming_threshold:
                    continue
                repair_components.union(str(left["group_id"]), str(right["group_id"]))
                _add_warning(
                    report,
                    warning_type,
                    left,
                    right,
                    score=max(0.0, 1.0 - hamming / 64.0),
                    example_limit=config.report_example_limit,
                )
                if extra_warning_type and hamming == 0:
                    _add_warning(
                        report,
                        extra_warning_type,
                        left,
                        right,
                        score=1.0,
                        example_limit=config.report_example_limit,
                    )
            if bucket_comparisons > config.similarity.max_pair_comparisons_per_bucket:
                break
    repair_component_count = _record_pairwise_repair_components(
        report,
        warning_type=warning_type,
        records_by_group=records_by_group,
        disjoint_set=repair_components,
    )
    return {
        "candidates": len(candidates),
        "bucket_count": len(bucketed),
        "comparisons": comparisons,
        "warnings_added": int(report["total_warning_count"] - warnings_before),
        "repair_component_count": repair_component_count,
        "mode": "phash_surrogate",
    }


def _embedding_cache_key(record: dict[str, Any], *, kind: str) -> tuple[str, str]:
    if kind == "audio":
        source = (
            record.get("audio_embedding_path")
            or record.get("audio_feature_path")
            or record.get("_audio_hash")
            or record.get("audio_hash")
            or record.get("audio_path")
            or record["sample_id"]
        )
        return kind, str(source)
    if kind == "face":
        source = (
            record.get("arcface_embedding_path")
            or record.get("identity_feature_path")
            or record.get("image_embedding_path")
            or record.get("image_path")
            or record.get("reference_image_path")
            or record["sample_id"]
        )
        return kind, str(source)
    if kind == "image":
        source = record.get("clip_embedding_path") or record.get("image_embedding_path") or record.get("image_path") or record["sample_id"]
        return kind, str(source)
    return kind, str(record["sample_id"])


def _resolve_embedding_cached(
    record: dict[str, Any],
    *,
    kind: str,
    embedding_cache: dict[tuple[str, str], Any],
) -> Any:
    cache_key = _embedding_cache_key(record, kind=kind)
    if cache_key not in embedding_cache:
        embedding_cache[cache_key] = resolve_embedding(record, kind=kind)
    return embedding_cache[cache_key]


def _analyze_embedding_overlap(
    records: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    kind: str,
    threshold: float,
    warning_type: str,
    config: SplitBuilderConfig,
    embedding_cache: dict[tuple[str, str], Any],
    filter_fn,
    extra_warning_type: str | None = None,
    extra_threshold: float | None = None,
    require_cross_dataset: bool = False,
) -> dict[str, Any]:
    candidates = [record for record in records if filter_fn(record)]
    if kind in {"face", "image"}:
        candidates = [record for record in candidates if _has_real_embedding_candidate(record, kind=kind)]
    if len(candidates) < 2:
        return {
            "candidates": len(candidates),
            "bucket_count": 0,
            "comparisons": 0,
            "warnings_added": 0,
            "repair_component_count": 0,
            "mode": "embedding",
        }

    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repair_components = _DisjointSet()
    comparisons = 0
    warnings_before = report["total_warning_count"]
    for record in candidates:
        vector = _resolve_embedding_cached(record, kind=kind, embedding_cache=embedding_cache)
        record[f"_soft_{kind}_embedding"] = vector
        bucketed[vector_signature(vector)].append(record)
        records_by_group[str(record["group_id"])].append(record)

    for bucket in bucketed.values():
        if len(bucket) > config.similarity.max_bucket_size:
            bucket = sorted(bucket, key=lambda item: item["sample_id"])[: config.similarity.max_bucket_size]
        bucket_comparisons = 0
        for index, left in enumerate(bucket):
            for right in bucket[index + 1 :]:
                bucket_comparisons += 1
                comparisons += 1
                if bucket_comparisons > config.similarity.max_pair_comparisons_per_bucket:
                    break
                if left["split"] == right["split"]:
                    continue
                if require_cross_dataset and left.get("dataset_source") == right.get("dataset_source"):
                    continue
                similarity = cosine_similarity(left[f"_soft_{kind}_embedding"], right[f"_soft_{kind}_embedding"])
                if similarity >= threshold:
                    repair_components.union(str(left["group_id"]), str(right["group_id"]))
                    _add_warning(
                        report,
                        warning_type,
                        left,
                        right,
                        score=similarity,
                        example_limit=config.report_example_limit,
                    )
                if extra_warning_type and extra_threshold is not None and similarity >= extra_threshold:
                    _add_warning(
                        report,
                        extra_warning_type,
                        left,
                        right,
                        score=similarity,
                        example_limit=config.report_example_limit,
                    )
            if bucket_comparisons > config.similarity.max_pair_comparisons_per_bucket:
                break
    repair_component_count = _record_pairwise_repair_components(
        report,
        warning_type=warning_type,
        records_by_group=records_by_group,
        disjoint_set=repair_components,
    )
    return {
        "candidates": len(candidates),
        "bucket_count": len(bucketed),
        "comparisons": comparisons,
        "warnings_added": int(report["total_warning_count"] - warnings_before),
        "repair_component_count": repair_component_count,
        "mode": "embedding",
    }


def analyze_soft_overlaps(records: list[dict[str, Any]], config: SplitBuilderConfig) -> dict[str, Any]:
    report = _init_report()
    embedding_cache: dict[tuple[str, str], Any] = {}

    timer = perf_counter()
    exact_audio_stats = _analyze_exact_audio_duplicates(records, report, config)
    _record_analysis_pass(report, "exact_audio_duplicates", elapsed_seconds=perf_counter() - timer, stats=exact_audio_stats)

    timer = perf_counter()
    text_stats = _analyze_exact_text_duplicates(records, report, config)
    _record_analysis_pass(report, "exact_text_duplicates", elapsed_seconds=perf_counter() - timer, stats=text_stats)

    timer = perf_counter()
    ambiguous_audio_stats = _analyze_ambiguous_audio_reuse(records, report, config)
    _record_analysis_pass(report, "ambiguous_audio_reuse", elapsed_seconds=perf_counter() - timer, stats=ambiguous_audio_stats)

    if config.enable_face_similarity:
        timer = perf_counter()
        face_surrogate_stats = _analyze_phash_surrogate_overlap(
            records,
            report,
            kind="face",
            warning_type="face_similarity",
            extra_warning_type="proxy_identity_cluster",
            config=config,
            filter_fn=lambda record: bool(record.get("image_path") or record.get("reference_image_path")),
        )
        _record_analysis_pass(report, "face_similarity_surrogate", elapsed_seconds=perf_counter() - timer, stats=face_surrogate_stats)

        timer = perf_counter()
        face_stats = _analyze_embedding_overlap(
            records,
            report,
            kind="face",
            threshold=config.similarity.face_similarity_threshold,
            warning_type="face_similarity",
            extra_warning_type="proxy_identity_cluster",
            extra_threshold=config.similarity.proxy_identity_threshold,
            config=config,
            embedding_cache=embedding_cache,
            filter_fn=lambda record: bool(record.get("image_path") or record.get("reference_image_path")),
        )
        _record_analysis_pass(report, "face_similarity_embeddings", elapsed_seconds=perf_counter() - timer, stats=face_stats)
    if config.enable_audio_similarity:
        timer = perf_counter()
        audio_stats = _analyze_embedding_overlap(
            records,
            report,
            kind="audio",
            threshold=config.similarity.audio_similarity_threshold,
            warning_type="audio_similarity",
            extra_warning_type="proxy_speaker_cluster",
            extra_threshold=config.similarity.proxy_speaker_threshold,
            config=config,
            embedding_cache=embedding_cache,
            filter_fn=lambda record: bool(record.get("audio_path") or record.get("audio_feature_path") or record.get("audio_embedding_path")),
        )
        _record_analysis_pass(report, "audio_similarity", elapsed_seconds=perf_counter() - timer, stats=audio_stats)
    if config.enable_image_similarity:
        timer = perf_counter()
        image_surrogate_stats = _analyze_phash_surrogate_overlap(
            records,
            report,
            kind="image",
            warning_type="image_similarity",
            config=config,
            filter_fn=lambda record: bool(record.get("image_path") or record.get("clip_embedding_path")),
        )
        _record_analysis_pass(report, "image_similarity_surrogate", elapsed_seconds=perf_counter() - timer, stats=image_surrogate_stats)

        timer = perf_counter()
        image_stats = _analyze_embedding_overlap(
            records,
            report,
            kind="image",
            threshold=config.similarity.image_similarity_threshold,
            warning_type="image_similarity",
            config=config,
            embedding_cache=embedding_cache,
            filter_fn=lambda record: bool(record.get("image_path") or record.get("clip_embedding_path")),
        )
        _record_analysis_pass(report, "image_similarity_embeddings", elapsed_seconds=perf_counter() - timer, stats=image_stats)

        timer = perf_counter()
        cross_dataset_surrogate_stats = _analyze_phash_surrogate_overlap(
            records,
            report,
            kind="image",
            warning_type="cross_dataset_similarity",
            config=config,
            filter_fn=lambda record: bool(record.get("image_path") or record.get("clip_embedding_path")),
            require_cross_dataset=True,
        )
        _record_analysis_pass(
            report,
            "cross_dataset_similarity_surrogate",
            elapsed_seconds=perf_counter() - timer,
            stats=cross_dataset_surrogate_stats,
        )

        timer = perf_counter()
        cross_dataset_stats = _analyze_embedding_overlap(
            records,
            report,
            kind="image",
            threshold=config.similarity.image_similarity_threshold,
            warning_type="cross_dataset_similarity",
            config=config,
            embedding_cache=embedding_cache,
            filter_fn=lambda record: bool(record.get("image_path") or record.get("clip_embedding_path")),
            require_cross_dataset=True,
        )
        _record_analysis_pass(report, "cross_dataset_similarity_embeddings", elapsed_seconds=perf_counter() - timer, stats=cross_dataset_stats)

    finalized = _finalize_report(report)
    finalized["status"] = "WARNING" if finalized["total_warning_count"] > 0 else "PASS"
    return finalized
