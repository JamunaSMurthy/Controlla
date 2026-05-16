"""Limited soft-overlap repair for initially assigned hard groups."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .config import SplitBuilderConfig


SPLITS = ("train", "val", "test")
HIGH_RISK_WARNING_TYPES = frozenset(
    {
        "exact_audio_duplicate",
        "face_similarity",
        "proxy_identity_cluster",
        "audio_similarity",
        "proxy_speaker_cluster",
        "image_similarity",
        "cross_dataset_similarity",
    }
)
TARGETED_COMPONENT_REPAIR_PRIORITY = (
    ("exact_audio_duplicate", "exact_audio_duplicate_reassign", "exact_audio_duplicate_drop", "drop"),
    ("cross_dataset_similarity", "cross_dataset_similarity_reassign", "cross_dataset_similarity_drop", "drop"),
    ("image_similarity", "image_similarity_reassign", "image_similarity_drop", "drop"),
)


def _target_counts(total_samples: int, config: SplitBuilderConfig) -> dict[str, float]:
    return {
        "train": total_samples * config.ratios.train,
        "val": total_samples * config.ratios.val,
        "test": total_samples * config.ratios.test,
    }


def _split_size_penalty(split_counts: dict[str, int], target_counts: dict[str, float]) -> float:
    penalty = 0.0
    for split_name in SPLITS:
        penalty += abs(split_counts.get(split_name, 0) - target_counts[split_name]) / max(target_counts[split_name], 1.0)
    return penalty


def _apply_action_to_group(group: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    updated = dict(group)
    if action["action"] == "reassign":
        updated["assigned_split"] = action["target_split"]
    return updated


def _split_preference_rank(split_name: str, config: SplitBuilderConfig) -> int:
    if config.assignment.strategy == "train_optimized":
        order = {"train": 0, "val": 1, "test": 2}
        return order.get(split_name, len(order))
    return {"test": 0, "train": 1, "val": 2}.get(split_name, 3)


def _ordered_group_ids_for_component(component: dict[str, Any]) -> list[str]:
    return [str(group_id) for group_id in component.get("group_ids", [])]


def _component_total_group_size(
    component: dict[str, Any],
    group_index: dict[str, dict[str, Any]],
    proposed_actions: dict[str, dict[str, Any]],
) -> int:
    return sum(
        int(group_index[group_id]["size"])
        for group_id in _ordered_group_ids_for_component(component)
        if group_id in group_index and group_id not in proposed_actions
    )


def _evaluate_component_target(
    component: dict[str, Any],
    *,
    warning_type: str,
    strategy: str,
    target_split: str,
    group_index: dict[str, dict[str, Any]],
    group_conflict_splits_by_type: dict[str, dict[str, dict[str, int]]],
    proposed_actions: dict[str, dict[str, Any]],
    split_counts: dict[str, int],
    target_counts: dict[str, float],
    config: SplitBuilderConfig,
    reassign_reason: str,
    drop_reason: str,
) -> dict[str, Any] | None:
    before_penalty = _split_size_penalty(split_counts, target_counts)
    candidate_counts = dict(split_counts)
    actions: list[dict[str, Any]] = []
    dropped_samples = 0
    moved_samples = 0
    changed_groups = 0
    resolved_conflicts = 0
    active_splits = {
        group_index[group_id]["assigned_split"]
        for group_id in _ordered_group_ids_for_component(component)
        if group_id in group_index and group_id not in proposed_actions
    }
    if len(active_splits) < 2:
        return None

    for group_id in _ordered_group_ids_for_component(component):
        if group_id in proposed_actions:
            continue
        group = group_index.get(group_id)
        if not group:
            continue
        current_split = group["assigned_split"]
        if current_split == target_split:
            continue
        if strategy == "drop":
            if group["size"] > config.repair.max_drop_group_size:
                return None
            action = {
                "action": "drop",
                "group_id": group_id,
                "from_split": current_split,
                "reason": drop_reason,
                "warning_count": int(component.get("num_groups", 0)),
            }
            candidate_counts[current_split] = candidate_counts.get(current_split, 0) - group["size"]
            dropped_samples += int(group["size"])
        else:
            resolved_conflicts += int(
                group_conflict_splits_by_type.get(group_id, {}).get(warning_type, {}).get(target_split, 0)
            )
            if group["size"] > config.repair.max_reassign_group_size:
                return None
            action = {
                "action": "reassign",
                "group_id": group_id,
                "from_split": current_split,
                "target_split": target_split,
                "reason": reassign_reason,
                "warning_count": int(component.get("num_groups", 0)),
            }
            candidate_counts[current_split] = candidate_counts.get(current_split, 0) - group["size"]
            candidate_counts[target_split] = candidate_counts.get(target_split, 0) + group["size"]
            moved_samples += int(group["size"])
        actions.append(action)
        changed_groups += 1

    if not actions:
        return None
    after_penalty = _split_size_penalty(candidate_counts, target_counts)
    if after_penalty > before_penalty + config.repair.max_split_ratio_deviation:
        return None
    if strategy == "drop":
        objective = (dropped_samples, after_penalty, changed_groups, _split_preference_rank(target_split, config), target_split)
    else:
        objective = (
            -resolved_conflicts,
            dropped_samples,
            after_penalty,
            moved_samples,
            changed_groups,
            _split_preference_rank(target_split, config),
            target_split,
        )
    return {
        "actions": actions,
        "split_counts": candidate_counts,
        "objective": objective,
    }


def _propose_component_repairs(
    *,
    soft_overlap_report: dict[str, Any],
    group_index: dict[str, dict[str, Any]],
    group_conflict_splits_by_type: dict[str, dict[str, dict[str, int]]],
    proposed_actions: dict[str, dict[str, Any]],
    split_counts: dict[str, int],
    target_counts: dict[str, float],
    config: SplitBuilderConfig,
) -> dict[str, int]:
    repair_components = soft_overlap_report.get("repair_components_by_type", {})
    for warning_type, reassign_reason, drop_reason, strategy in TARGETED_COMPONENT_REPAIR_PRIORITY:
        components = list(repair_components.get(warning_type, []))
        components.sort(
            key=lambda component: (
                -_component_total_group_size(component, group_index, proposed_actions),
                -int(component.get("num_groups", 0)),
                tuple(_ordered_group_ids_for_component(component)),
            )
        )
        for component in components:
            active_groups = [
                group_index[group_id]
                for group_id in _ordered_group_ids_for_component(component)
                if group_id in group_index and group_id not in proposed_actions
            ]
            if len({group["assigned_split"] for group in active_groups}) < 2:
                continue
            candidate_plans: list[dict[str, Any]] = []
            for target_split in sorted({group["assigned_split"] for group in active_groups}):
                plan = _evaluate_component_target(
                    component,
                    warning_type=warning_type,
                    strategy=strategy,
                    target_split=target_split,
                    group_index=group_index,
                    group_conflict_splits_by_type=group_conflict_splits_by_type,
                    proposed_actions=proposed_actions,
                    split_counts=split_counts,
                    target_counts=target_counts,
                    config=config,
                    reassign_reason=reassign_reason,
                    drop_reason=drop_reason,
                )
                if plan is not None:
                    candidate_plans.append(plan)
            if not candidate_plans:
                continue
            best_plan = min(candidate_plans, key=lambda plan: plan["objective"])
            for action in best_plan["actions"]:
                proposed_actions[action["group_id"]] = action
            split_counts = dict(best_plan["split_counts"])
    return split_counts


def _flatten_assigned_records(assigned_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in assigned_groups:
        for sample in group["samples"]:
            records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})
    return records


def _run_single_repair_iteration(
    *,
    assigned_groups: list[dict[str, Any]],
    dropped_groups: list[dict[str, Any]],
    soft_overlap_report: dict[str, Any],
    split_counts: dict[str, int],
    target_counts: dict[str, float],
    config: SplitBuilderConfig,
    iteration_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    group_conflict_splits = soft_overlap_report.get("group_conflict_split_counts", {})
    group_conflict_splits_by_type = soft_overlap_report.get("group_conflict_split_counts_by_type", {})
    group_total_conflicts = soft_overlap_report.get("group_total_warning_counts", {})
    group_warning_counts = soft_overlap_report.get("group_warning_counts", {})
    quarantine_candidate_groups = soft_overlap_report.get("quarantine_candidate_groups", {})
    proposed_actions: dict[str, dict[str, Any]] = {}
    group_index = {group["group_id"]: group for group in assigned_groups}
    ordered_groups = sorted(
        assigned_groups,
        key=lambda group: (-int(group_total_conflicts.get(group["group_id"], 0)), group["size"], group["group_id"]),
    )

    next_split_counts = dict(split_counts)
    for group in ordered_groups:
        group_id = group["group_id"]
        if group_id in proposed_actions:
            continue
        quarantine_score = int(quarantine_candidate_groups.get(group_id, 0))
        if (
            quarantine_score >= config.repair.ambiguous_audio_quarantine_min_count
            and group["size"] <= config.repair.max_drop_group_size
        ):
            current_split = group["assigned_split"]
            if config.repair.prefer_train_for_ambiguous_reassign and current_split in {"val", "test"}:
                proposed_actions[group_id] = {
                    "action": "reassign",
                    "group_id": group_id,
                    "from_split": current_split,
                    "target_split": "train",
                    "reason": "ambiguous_audio_reuse_train_reassign",
                    "warning_count": quarantine_score,
                }
                next_split_counts[current_split] = next_split_counts.get(current_split, 0) - group["size"]
                next_split_counts["train"] = next_split_counts.get("train", 0) + group["size"]
            else:
                proposed_actions[group_id] = {
                    "action": "drop",
                    "group_id": group_id,
                    "from_split": current_split,
                    "reason": "ambiguous_audio_reuse_quarantine",
                    "warning_count": quarantine_score,
                }
                next_split_counts[current_split] = next_split_counts.get(current_split, 0) - group["size"]

    next_split_counts = _propose_component_repairs(
        soft_overlap_report=soft_overlap_report,
        group_index=group_index,
        group_conflict_splits_by_type=group_conflict_splits_by_type,
        proposed_actions=proposed_actions,
        split_counts=next_split_counts,
        target_counts=target_counts,
        config=config,
    )

    for group in ordered_groups:
        group_id = group["group_id"]
        if group_id in proposed_actions:
            continue
        total_conflicts = int(group_total_conflicts.get(group_id, 0))
        if total_conflicts < config.repair.min_warning_edges_to_act:
            continue
        conflict_by_split = dict(group_conflict_splits.get(group_id, {}))
        warning_counts_by_type = dict(group_warning_counts.get(group_id, {}))
        high_risk_type_count = sum(1 for warning_type in warning_counts_by_type if warning_type in HIGH_RISK_WARNING_TYPES)
        current_split = group["assigned_split"]
        if not conflict_by_split:
            continue
        ranked_targets = sorted(conflict_by_split.items(), key=lambda item: (-item[1], item[0]))
        best_split, best_count = ranked_targets[0]
        second_count = ranked_targets[1][1] if len(ranked_targets) > 1 else 0
        if best_split == current_split:
            continue

        clear_winner = best_count > second_count * (1.0 + config.repair.ambiguity_ratio_threshold)
        if (
            config.repair.enable_generic_soft_reassign
            and clear_winner
            and group["size"] <= config.repair.max_reassign_group_size
        ):
            before_penalty = _split_size_penalty(next_split_counts, target_counts)
            candidate_counts = dict(next_split_counts)
            candidate_counts[current_split] = candidate_counts.get(current_split, 0) - group["size"]
            candidate_counts[best_split] = candidate_counts.get(best_split, 0) + group["size"]
            after_penalty = _split_size_penalty(candidate_counts, target_counts)
            if after_penalty <= before_penalty + config.repair.max_split_ratio_deviation:
                proposed_actions[group_id] = {
                    "action": "reassign",
                    "group_id": group_id,
                    "from_split": current_split,
                    "target_split": best_split,
                    "reason": "soft_overlap_conflict_reassign",
                    "warning_count": total_conflicts,
                }
                next_split_counts = candidate_counts
                continue

        if (
            config.repair.enable_generic_soft_conflict_drop
            and group["size"] <= config.repair.max_drop_group_size
            and total_conflicts >= config.repair.soft_conflict_drop_min_total_warnings
            and high_risk_type_count >= config.repair.soft_conflict_drop_min_high_risk_types
        ):
            proposed_actions[group_id] = {
                "action": "drop",
                "group_id": group_id,
                "from_split": current_split,
                "reason": "soft_overlap_conflict_drop",
                "warning_count": total_conflicts,
            }
            next_split_counts[current_split] = next_split_counts.get(current_split, 0) - group["size"]

    next_assigned_groups: list[dict[str, Any]] = []
    iteration_actions: list[dict[str, Any]] = []
    next_dropped_groups = list(dropped_groups)
    for group in assigned_groups:
        action = proposed_actions.get(group["group_id"])
        if not action:
            next_assigned_groups.append(group)
            continue
        iteration_actions.append(action)
        if action["action"] == "drop":
            next_dropped_groups.append(
                {**group, "drop_reason": action["reason"], "drop_stage": f"repair_iteration_{iteration_index}"}
            )
        else:
            next_assigned_groups.append(_apply_action_to_group(group, action))

    return next_assigned_groups, next_dropped_groups, next_split_counts, iteration_actions


def repair_split_assignments(
    assignment_result: dict[str, Any],
    soft_overlap_report: dict[str, Any],
    config: SplitBuilderConfig,
) -> dict[str, Any]:
    if not config.repair.enable_repair:
        return {
            "assigned_groups": assignment_result["assigned_groups"],
            "dropped_groups": assignment_result["dropped_groups"],
            "repair_actions": [],
            "repair_summary": {"num_reassigned_groups": 0, "num_dropped_groups": 0, "num_iterations": 0},
        }

    assigned_groups = [dict(group) for group in assignment_result["assigned_groups"]]
    dropped_groups = list(assignment_result["dropped_groups"])
    target_counts = _target_counts(sum(group["size"] for group in assigned_groups), config)
    split_counts = dict(assignment_result["split_counts"])
    actions: list[dict[str, Any]] = []
    current_soft_overlap_report = soft_overlap_report
    num_iterations = 0
    for iteration_index in range(1, config.repair.max_post_repair_iterations + 1):
        assigned_groups, dropped_groups, split_counts, iteration_actions = _run_single_repair_iteration(
            assigned_groups=assigned_groups,
            dropped_groups=dropped_groups,
            soft_overlap_report=current_soft_overlap_report,
            split_counts=split_counts,
            target_counts=target_counts,
            config=config,
            iteration_index=iteration_index,
        )
        if not iteration_actions:
            break
        actions.extend(iteration_actions)
        num_iterations = iteration_index
        if iteration_index >= config.repair.max_post_repair_iterations:
            break
        from .soft_overlap import analyze_soft_overlaps

        current_soft_overlap_report = analyze_soft_overlaps(_flatten_assigned_records(assigned_groups), config)

    action_reason_counts = Counter(action["reason"] for action in actions)
    repair_summary = {
        "num_reassigned_groups": sum(1 for action in actions if action["action"] == "reassign"),
        "num_dropped_groups": sum(1 for action in actions if action["action"] == "drop"),
        "num_iterations": num_iterations,
        "action_counts_by_reason": dict(action_reason_counts),
    }
    return {
        "assigned_groups": assigned_groups,
        "dropped_groups": dropped_groups,
        "repair_actions": actions,
        "repair_summary": repair_summary,
    }
