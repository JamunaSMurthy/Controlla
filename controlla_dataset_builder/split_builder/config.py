"""Configuration models for the split builder."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json


@dataclass(slots=True)
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass(slots=True)
class SimilarityConfig:
    face_similarity_threshold: float = 0.98
    audio_similarity_threshold: float = 0.97
    image_similarity_threshold: float = 0.985
    text_similarity_threshold: float = 0.98
    image_phash_hamming_threshold: int = 2
    max_bucket_size: int = 256
    max_pair_comparisons_per_bucket: int = 4096
    proxy_identity_threshold: float = 0.992
    proxy_speaker_threshold: float = 0.992


@dataclass(slots=True)
class AssignmentPolicyConfig:
    strategy: str = "balanced"
    val_test_target_reserve_factor: float = 1.0

    def validate(self) -> None:
        if self.strategy not in {"balanced", "train_optimized"}:
            raise ValueError(f"Unsupported assignment strategy: {self.strategy}")
        if self.val_test_target_reserve_factor <= 0.0:
            raise ValueError("val_test_target_reserve_factor must be positive")


@dataclass(slots=True)
class GroupingPolicyConfig:
    max_group_size: int = 5000
    max_group_fraction: float = 0.35
    min_total_samples_for_fraction_rule: int = 1000
    quarantine_oversized_groups: bool = False
    enable_audio_hash_hard_grouping: bool = True
    audio_hash_max_group_size_for_hard_link: int = 32
    audio_hash_max_distinct_identities_for_hard_link: int = 8
    audio_hash_max_distinct_texts_for_hard_link: int = 32
    report_top_component_limit: int = 10


@dataclass(slots=True)
class RepairPolicyConfig:
    enable_repair: bool = True
    max_reassign_group_size: int = 48
    max_drop_group_size: int = 128
    enable_generic_soft_reassign: bool = True
    enable_generic_soft_conflict_drop: bool = True
    prefer_train_for_ambiguous_reassign: bool = False
    min_warning_edges_to_act: int = 1
    ambiguous_audio_quarantine_min_count: int = 4
    soft_conflict_drop_min_total_warnings: int = 8
    soft_conflict_drop_min_high_risk_types: int = 1
    ambiguity_ratio_threshold: float = 0.15
    max_post_repair_iterations: int = 7
    max_split_ratio_deviation: float = 0.08


@dataclass(slots=True)
class SplitBuilderConfig:
    input_path: str
    output_dir: str
    ratios: SplitRatios = field(default_factory=SplitRatios)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    assignment: AssignmentPolicyConfig = field(default_factory=AssignmentPolicyConfig)
    grouping: GroupingPolicyConfig = field(default_factory=GroupingPolicyConfig)
    repair: RepairPolicyConfig = field(default_factory=RepairPolicyConfig)
    seed: int = 0
    batch_size: int = 128
    enable_face_similarity: bool = True
    enable_audio_similarity: bool = True
    enable_image_similarity: bool = True
    enable_text_similarity: bool = True
    drop_ambiguous_samples: bool = False
    report_example_limit: int = 20

    def validate(self) -> None:
        self.ratios.validate()
        self.assignment.validate()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def load_config(
    *,
    input_path: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    config_path: str | None = None,
    seed: int = 0,
    batch_size: int = 128,
) -> SplitBuilderConfig:
    config = SplitBuilderConfig(
        input_path=input_path,
        output_dir=output_dir,
        ratios=SplitRatios(train=train_ratio, val=val_ratio, test=test_ratio),
        seed=seed,
        batch_size=batch_size,
    )
    if config_path:
        loaded = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if "similarity" in loaded:
            similarity = {**asdict(config.similarity), **loaded.pop("similarity")}
            config.similarity = SimilarityConfig(**similarity)
        if "assignment" in loaded:
            assignment = {**asdict(config.assignment), **loaded.pop("assignment")}
            config.assignment = AssignmentPolicyConfig(**assignment)
        if "grouping" in loaded:
            grouping = {**asdict(config.grouping), **loaded.pop("grouping")}
            config.grouping = GroupingPolicyConfig(**grouping)
        if "repair" in loaded:
            repair = {**asdict(config.repair), **loaded.pop("repair")}
            config.repair = RepairPolicyConfig(**repair)
        if "ratios" in loaded:
            ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio, **loaded.pop("ratios")}
            config.ratios = SplitRatios(**ratios)
        for key, value in loaded.items():
            if hasattr(config, key):
                setattr(config, key, value)
    config.validate()
    return config


def config_as_dict(config: SplitBuilderConfig) -> dict[str, Any]:
    return {
        "input_path": config.input_path,
        "output_dir": config.output_dir,
        "ratios": {
            "train": config.ratios.train,
            "val": config.ratios.val,
            "test": config.ratios.test,
        },
        "similarity": {
            "face_similarity_threshold": config.similarity.face_similarity_threshold,
            "audio_similarity_threshold": config.similarity.audio_similarity_threshold,
            "image_similarity_threshold": config.similarity.image_similarity_threshold,
            "text_similarity_threshold": config.similarity.text_similarity_threshold,
            "image_phash_hamming_threshold": config.similarity.image_phash_hamming_threshold,
            "max_bucket_size": config.similarity.max_bucket_size,
            "max_pair_comparisons_per_bucket": config.similarity.max_pair_comparisons_per_bucket,
            "proxy_identity_threshold": config.similarity.proxy_identity_threshold,
            "proxy_speaker_threshold": config.similarity.proxy_speaker_threshold,
        },
        "assignment": {
            "strategy": config.assignment.strategy,
            "val_test_target_reserve_factor": config.assignment.val_test_target_reserve_factor,
        },
        "grouping": {
            "max_group_size": config.grouping.max_group_size,
            "max_group_fraction": config.grouping.max_group_fraction,
            "min_total_samples_for_fraction_rule": config.grouping.min_total_samples_for_fraction_rule,
            "quarantine_oversized_groups": config.grouping.quarantine_oversized_groups,
            "enable_audio_hash_hard_grouping": config.grouping.enable_audio_hash_hard_grouping,
            "audio_hash_max_group_size_for_hard_link": config.grouping.audio_hash_max_group_size_for_hard_link,
            "audio_hash_max_distinct_identities_for_hard_link": config.grouping.audio_hash_max_distinct_identities_for_hard_link,
            "audio_hash_max_distinct_texts_for_hard_link": config.grouping.audio_hash_max_distinct_texts_for_hard_link,
            "report_top_component_limit": config.grouping.report_top_component_limit,
        },
        "repair": {
            "enable_repair": config.repair.enable_repair,
            "max_reassign_group_size": config.repair.max_reassign_group_size,
            "max_drop_group_size": config.repair.max_drop_group_size,
            "enable_generic_soft_reassign": config.repair.enable_generic_soft_reassign,
            "enable_generic_soft_conflict_drop": config.repair.enable_generic_soft_conflict_drop,
            "prefer_train_for_ambiguous_reassign": config.repair.prefer_train_for_ambiguous_reassign,
            "min_warning_edges_to_act": config.repair.min_warning_edges_to_act,
            "ambiguous_audio_quarantine_min_count": config.repair.ambiguous_audio_quarantine_min_count,
            "soft_conflict_drop_min_total_warnings": config.repair.soft_conflict_drop_min_total_warnings,
            "soft_conflict_drop_min_high_risk_types": config.repair.soft_conflict_drop_min_high_risk_types,
            "ambiguity_ratio_threshold": config.repair.ambiguity_ratio_threshold,
            "max_post_repair_iterations": config.repair.max_post_repair_iterations,
            "max_split_ratio_deviation": config.repair.max_split_ratio_deviation,
        },
        "seed": config.seed,
        "batch_size": config.batch_size,
        "enable_face_similarity": config.enable_face_similarity,
        "enable_audio_similarity": config.enable_audio_similarity,
        "enable_image_similarity": config.enable_image_similarity,
        "enable_text_similarity": config.enable_text_similarity,
        "drop_ambiguous_samples": config.drop_ambiguous_samples,
        "report_example_limit": config.report_example_limit,
    }