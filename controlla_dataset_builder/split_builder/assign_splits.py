"""Assign connected groups to train/val/test splits."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .config import SplitBuilderConfig


SPLITS = ("train", "val", "test")


def _target_counts(total_samples: int, config: SplitBuilderConfig) -> dict[str, float]:
    return {
        "train": total_samples * config.ratios.train,
        "val": total_samples * config.ratios.val,
        "test": total_samples * config.ratios.test,
    }


def _assignment_target_counts(total_samples: int, config: SplitBuilderConfig) -> dict[str, float]:
    targets = _target_counts(total_samples, config)
    if config.assignment.strategy != "train_optimized":
        return targets
    reserve_factor = config.assignment.val_test_target_reserve_factor
    adjusted_val = targets["val"] * reserve_factor
    adjusted_test = targets["test"] * reserve_factor
    adjusted_train = max(0.0, total_samples - adjusted_val - adjusted_test)
    return {
        "train": adjusted_train,
        "val": adjusted_val,
        "test": adjusted_test,
    }


def _target_fractions(target_samples: dict[str, float], total_samples: int) -> dict[str, float]:
    if total_samples <= 0:
        return {split: 0.0 for split in SPLITS}
    return {split: target_samples[split] / total_samples for split in SPLITS}


def _global_targets(groups: list[dict[str, Any]], target_fractions: dict[str, float]) -> dict[str, dict[str, float]]:
    emotion_total = Counter()
    dataset_total = Counter()
    modality_total = Counter()
    for group in groups:
        emotion_total.update(group["emotion_counts"])
        dataset_total.update(group["dataset_counts"])
        modality_total.update(group["modality_counts"])
    return {
        "emotion": {split: {key: value * target_fractions[split] for key, value in emotion_total.items()} for split in SPLITS},
        "dataset": {split: {key: value * target_fractions[split] for key, value in dataset_total.items()} for split in SPLITS},
        "modality": {split: {key: value * target_fractions[split] for key, value in modality_total.items()} for split in SPLITS},
    }


def _score_assignment(
    group: dict[str, Any],
    split_name: str,
    *,
    assigned_samples: dict[str, int],
    assigned_emotions: dict[str, Counter],
    assigned_datasets: dict[str, Counter],
    assigned_modalities: dict[str, Counter],
    target_samples: dict[str, float],
    target_balances: dict[str, dict[str, dict[str, float]]],
) -> float:
    projected_samples = assigned_samples[split_name] + group["size"]
    size_penalty = abs(projected_samples - target_samples[split_name]) / max(target_samples[split_name], 1.0)

    emotion_penalty = 0.0
    for emotion, count in group["emotion_counts"].items():
        projected = assigned_emotions[split_name][emotion] + count
        target = target_balances["emotion"][split_name].get(emotion, 0.0)
        emotion_penalty += abs(projected - target) / max(target + 1.0, 1.0)

    dataset_penalty = 0.0
    for source, count in group["dataset_counts"].items():
        projected = assigned_datasets[split_name][source] + count
        target = target_balances["dataset"][split_name].get(source, 0.0)
        dataset_penalty += abs(projected - target) / max(target + 1.0, 1.0)

    modality_penalty = 0.0
    for modality, count in group["modality_counts"].items():
        projected = assigned_modalities[split_name][modality] + count
        target = target_balances["modality"][split_name].get(modality, 0.0)
        modality_penalty += abs(projected - target) / max(target + 1.0, 1.0)

    diversity_bonus = 0.0
    if group["dataset_counts"] and not assigned_datasets[split_name]:
        diversity_bonus -= 0.1
    if group["emotion_counts"] and len(assigned_emotions[split_name]) < 3:
        diversity_bonus -= 0.05

    return 1.8 * size_penalty + 1.2 * emotion_penalty + 1.0 * dataset_penalty + 0.6 * modality_penalty + diversity_bonus


def _drop_reason(group: dict[str, Any], config: SplitBuilderConfig, total_samples: int) -> str | None:
    exceeds_size_limit = group["size"] > config.grouping.max_group_size
    exceeds_fraction_limit = (
        total_samples >= config.grouping.min_total_samples_for_fraction_rule
        and group["size"] / max(total_samples, 1) > config.grouping.max_group_fraction
    )
    if config.grouping.quarantine_oversized_groups and (exceeds_size_limit or exceeds_fraction_limit):
        return "oversized_hard_group"
    if config.drop_ambiguous_samples and len(group["existing_splits"]) > 2:
        return "ambiguous_existing_split_links"
    return None


def _apply_group_assignment(
    group: dict[str, Any],
    split_name: str,
    *,
    assigned_samples: dict[str, int],
    assigned_emotions: dict[str, Counter],
    assigned_datasets: dict[str, Counter],
    assigned_modalities: dict[str, Counter],
    grouped_assignments: list[dict[str, Any]],
) -> None:
    assigned_samples[split_name] += group["size"]
    assigned_emotions[split_name].update(group["emotion_counts"])
    assigned_datasets[split_name].update(group["dataset_counts"])
    assigned_modalities[split_name].update(group["modality_counts"])
    grouped_assignments.append({**group, "assigned_split": split_name})


def _seed_split_assignments(
    ordered_groups: list[dict[str, Any]],
    *,
    config: SplitBuilderConfig,
    total_samples: int,
    assigned_samples: dict[str, int],
    assigned_emotions: dict[str, Counter],
    assigned_datasets: dict[str, Counter],
    assigned_modalities: dict[str, Counter],
    grouped_assignments: list[dict[str, Any]],
    dropped_groups: list[dict[str, Any]],
    target_samples: dict[str, float],
) -> list[dict[str, Any]]:
    remaining_groups = []
    for group in ordered_groups:
        reason = _drop_reason(group, config, total_samples)
        if reason:
            dropped_groups.append({**group, "drop_reason": reason, "drop_stage": "initial_assignment"})
            continue
        remaining_groups.append(group)
    for split_name in ("val", "test", "train"):
        if not remaining_groups:
            break
        best_index = min(
            range(len(remaining_groups)),
            key=lambda index: (
                abs(remaining_groups[index]["size"] - target_samples[split_name]) / max(target_samples[split_name], 1.0),
                -len(remaining_groups[index]["dataset_counts"]),
                -len(remaining_groups[index]["emotion_counts"]),
                remaining_groups[index]["group_id"],
            ),
        )
        group = remaining_groups.pop(best_index)
        _apply_group_assignment(
            group,
            split_name,
            assigned_samples=assigned_samples,
            assigned_emotions=assigned_emotions,
            assigned_datasets=assigned_datasets,
            assigned_modalities=assigned_modalities,
            grouped_assignments=grouped_assignments,
        )
    return remaining_groups


def _filter_dropped_groups(
    ordered_groups: list[dict[str, Any]],
    *,
    config: SplitBuilderConfig,
    total_samples: int,
    dropped_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    remaining_groups = []
    for group in ordered_groups:
        reason = _drop_reason(group, config, total_samples)
        if reason:
            dropped_groups.append({**group, "drop_reason": reason, "drop_stage": "initial_assignment"})
            continue
        remaining_groups.append(group)
    return remaining_groups


def _assign_groups_train_optimized(
    ordered_groups: list[dict[str, Any]],
    *,
    config: SplitBuilderConfig,
    total_samples: int,
    assigned_samples: dict[str, int],
    assigned_emotions: dict[str, Counter],
    assigned_datasets: dict[str, Counter],
    assigned_modalities: dict[str, Counter],
    grouped_assignments: list[dict[str, Any]],
    dropped_groups: list[dict[str, Any]],
    target_samples: dict[str, float],
) -> None:
    remaining_groups = _filter_dropped_groups(
        ordered_groups,
        config=config,
        total_samples=total_samples,
        dropped_groups=dropped_groups,
    )
    for split_name in ("val", "test"):
        if not remaining_groups:
            return
        best_index = min(
            range(len(remaining_groups)),
            key=lambda index: (
                abs(remaining_groups[index]["size"] - target_samples[split_name]) / max(target_samples[split_name], 1.0),
                remaining_groups[index]["group_id"],
            ),
        )
        group = remaining_groups.pop(best_index)
        _apply_group_assignment(
            group,
            split_name,
            assigned_samples=assigned_samples,
            assigned_emotions=assigned_emotions,
            assigned_datasets=assigned_datasets,
            assigned_modalities=assigned_modalities,
            grouped_assignments=grouped_assignments,
        )

    for group in remaining_groups:
        deficits = {
            split_name: target_samples[split_name] - assigned_samples[split_name]
            for split_name in ("val", "test")
        }
        candidate_splits = [split_name for split_name in ("val", "test") if deficits[split_name] > 0]
        if candidate_splits:
            chosen_split = max(
                candidate_splits,
                key=lambda split_name: (
                    deficits[split_name],
                    -assigned_samples[split_name],
                    split_name,
                ),
            )
        else:
            chosen_split = "train"
        _apply_group_assignment(
            group,
            chosen_split,
            assigned_samples=assigned_samples,
            assigned_emotions=assigned_emotions,
            assigned_datasets=assigned_datasets,
            assigned_modalities=assigned_modalities,
            grouped_assignments=grouped_assignments,
        )


def assign_groups_to_splits(groups: list[dict[str, Any]], config: SplitBuilderConfig) -> dict[str, Any]:
    total_samples = sum(group["size"] for group in groups)
    target_samples = _assignment_target_counts(total_samples, config)
    target_balances = _global_targets(groups, _target_fractions(target_samples, total_samples))

    assigned_samples = {split: 0 for split in SPLITS}
    assigned_emotions = {split: Counter() for split in SPLITS}
    assigned_datasets = {split: Counter() for split in SPLITS}
    assigned_modalities = {split: Counter() for split in SPLITS}

    grouped_assignments: list[dict[str, Any]] = []
    dropped_groups: list[dict[str, Any]] = []

    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -group["size"],
            -len(group["existing_splits"]),
            -len(group["dataset_counts"]),
            group["group_id"],
        ),
    )

    if config.assignment.strategy == "train_optimized":
        _assign_groups_train_optimized(
            ordered_groups,
            config=config,
            total_samples=total_samples,
            assigned_samples=assigned_samples,
            assigned_emotions=assigned_emotions,
            assigned_datasets=assigned_datasets,
            assigned_modalities=assigned_modalities,
            grouped_assignments=grouped_assignments,
            dropped_groups=dropped_groups,
            target_samples=target_samples,
        )
        return {
            "assigned_groups": grouped_assignments,
            "dropped_groups": dropped_groups,
            "split_counts": assigned_samples,
            "emotion_distribution": {split: dict(counter) for split, counter in assigned_emotions.items()},
            "dataset_distribution": {split: dict(counter) for split, counter in assigned_datasets.items()},
            "modality_distribution": {split: dict(counter) for split, counter in assigned_modalities.items()},
        }

    ordered_groups = _seed_split_assignments(
        ordered_groups,
        config=config,
        total_samples=total_samples,
        assigned_samples=assigned_samples,
        assigned_emotions=assigned_emotions,
        assigned_datasets=assigned_datasets,
        assigned_modalities=assigned_modalities,
        grouped_assignments=grouped_assignments,
        dropped_groups=dropped_groups,
        target_samples=target_samples,
    )

    for group in ordered_groups:
        reason = _drop_reason(group, config, total_samples)
        if reason:
            dropped_groups.append({**group, "drop_reason": reason, "drop_stage": "initial_assignment"})
            continue

        split_scores = []
        for split_name in SPLITS:
            score = _score_assignment(
                group,
                split_name,
                assigned_samples=assigned_samples,
                assigned_emotions=assigned_emotions,
                assigned_datasets=assigned_datasets,
                assigned_modalities=assigned_modalities,
                target_samples=target_samples,
                target_balances=target_balances,
            )
            split_scores.append((score, split_name))
        _, chosen_split = min(split_scores, key=lambda item: (item[0], item[1]))

        _apply_group_assignment(
            group,
            chosen_split,
            assigned_samples=assigned_samples,
            assigned_emotions=assigned_emotions,
            assigned_datasets=assigned_datasets,
            assigned_modalities=assigned_modalities,
            grouped_assignments=grouped_assignments,
        )

    return {
        "assigned_groups": grouped_assignments,
        "dropped_groups": dropped_groups,
        "split_counts": assigned_samples,
        "emotion_distribution": {split: dict(counter) for split, counter in assigned_emotions.items()},
        "dataset_distribution": {split: dict(counter) for split, counter in assigned_datasets.items()},
        "modality_distribution": {split: dict(counter) for split, counter in assigned_modalities.items()},
    }