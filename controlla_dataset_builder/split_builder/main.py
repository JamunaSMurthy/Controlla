"""CLI entrypoint for leakage-safe group-based splitting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .assign_splits import assign_groups_to_splits
from .build_groups import build_groups
from .config import config_as_dict, load_config
from .load_data import load_metadata
from .repair_splits import repair_split_assignments
from .report import build_split_report, write_report
from .save_outputs import save_split_outputs, write_json
from .soft_overlap import analyze_soft_overlaps
from .validate_splits import validate_split_assignments


STAGE_NAMES = (
    "load_config+load_metadata",
    "build_groups",
    "assign_groups_to_splits",
    "analyze_soft_overlaps",
    "repair_split_assignments",
    "validate_split_assignments",
)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _top_counts(mapping: dict[str, int], *, limit: int = 5) -> dict[str, int]:
    return dict(sorted(mapping.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _slowest_timing(timings: dict[str, float]) -> dict[str, float]:
    if not timings:
        return {}
    name, value = max(timings.items(), key=lambda item: (item[1], item[0]))
    return {name: round(float(value), 6)}


def _validation_overlap_summary(validation_result: dict[str, Any]) -> dict[str, int]:
    return {
        "identity_overlap_count": int(validation_result.get("identity_overlap_count", 0)),
        "speaker_overlap_count": int(validation_result.get("speaker_overlap_count", 0)),
        "exact_image_duplicate_overlap": int(validation_result.get("exact_image_duplicate_overlap", 0)),
        "exact_audio_duplicate_overlap": int(validation_result.get("exact_audio_duplicate_overlap", 0)),
        "near_duplicate_overlap_count": int(validation_result.get("near_duplicate_overlap_count", 0)),
        "cross_dataset_overlap_count": int(validation_result.get("cross_dataset_overlap_count", 0)),
    }


def _build_stage_summary(stage_name: str, result: Any) -> dict[str, Any]:
    if stage_name == "load_config+load_metadata":
        config, samples = result
        return {
            "num_samples": len(samples),
            "batch_size": config.batch_size,
        }
    if stage_name == "build_groups":
        return {
            "num_groups": int(result["num_groups"]),
            "largest_group_size": int(result["hard_group_report"]["largest_hard_group_size"]),
        }
    if stage_name == "assign_groups_to_splits":
        return {
            "split_counts": dict(result["split_counts"]),
        }
    if stage_name == "analyze_soft_overlaps":
        return {
            "total_warnings": int(result["total_warning_count"]),
            "unique_groups_in_warnings": len(result["group_total_warning_counts"]),
            "top_warning_types": _top_counts(result["warning_counts_by_type"]),
            "quarantine_candidate_groups": len(result.get("quarantine_candidate_groups", {})),
            "slowest_pass": _slowest_timing(result.get("analysis_timings_seconds", {})),
        }
    if stage_name == "repair_split_assignments":
        repair_summary = result["repair_summary"]
        return {
            "groups_reassigned": int(repair_summary["num_reassigned_groups"]),
            "groups_dropped": int(repair_summary["num_dropped_groups"]),
            "iterations": int(repair_summary["num_iterations"]),
            "action_counts_by_reason": dict(repair_summary.get("action_counts_by_reason", {})),
        }
    if stage_name == "validate_split_assignments":
        return {
            "status": str(result["status"]),
            "non_empty_split_count": int(result["non_empty_split_count"]),
            "remaining_overlap_counts": _validation_overlap_summary(result),
        }
    return {}


def _print_stage_start(stage_name: str) -> str:
    started_at = _timestamp()
    print(f"[stage:{stage_name}] start={started_at}")
    return started_at


def _print_stage_end(stage_name: str, *, started_at: str, elapsed_seconds: float, summary: dict[str, Any]) -> None:
    ended_at = _timestamp()
    print(
        f"[stage:{stage_name}] end={ended_at} elapsed_seconds={elapsed_seconds:.3f} summary="
        f"{json.dumps(summary, sort_keys=True)}"
    )


def _run_stage(
    *,
    stage_name: str,
    cache_key: str,
    stage_cache: dict[str, Any],
    compute_fn,
    bottleneck_threshold_seconds: float | None,
    checkpoints: list[dict[str, Any]],
) -> tuple[Any, bool]:
    started_at = _print_stage_start(stage_name)
    timer = perf_counter()
    if cache_key not in stage_cache:
        stage_cache[cache_key] = compute_fn()
    result = stage_cache[cache_key]
    elapsed_seconds = perf_counter() - timer
    summary = _build_stage_summary(stage_name, result)
    _print_stage_end(stage_name, started_at=started_at, elapsed_seconds=elapsed_seconds, summary=summary)
    checkpoint = {
        "stage": stage_name,
        "start": started_at,
        "end": _timestamp(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "summary": summary,
    }
    checkpoints.append(checkpoint)
    if bottleneck_threshold_seconds is not None and elapsed_seconds >= bottleneck_threshold_seconds:
        print(
            f"[stage:{stage_name}] bottleneck_detected=true threshold_seconds={bottleneck_threshold_seconds:.3f} "
            f"diagnostics={json.dumps(summary, sort_keys=True)}"
        )
        return result, True
    return result, False


def run_stage_diagnostics(
    *,
    input_path: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    config_path: str | None,
    seed: int,
    batch_size: int,
    bottleneck_threshold_seconds: float | None,
) -> dict[str, Any]:
    stage_cache: dict[str, Any] = {}
    checkpoints: list[dict[str, Any]] = []

    def _load_stage() -> tuple[Any, list[dict[str, Any]]]:
        config = load_config(
            input_path=input_path,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            config_path=config_path,
            seed=seed,
            batch_size=batch_size,
        )
        return config, load_metadata(config.input_path)

    stage_1_result, should_stop = _run_stage(
        stage_name="load_config+load_metadata",
        cache_key="load",
        stage_cache=stage_cache,
        compute_fn=_load_stage,
        bottleneck_threshold_seconds=bottleneck_threshold_seconds,
        checkpoints=checkpoints,
    )
    config, samples = stage_1_result
    if should_stop:
        return {"stopped_at_stage": "load_config+load_metadata", "checkpoints": checkpoints}

    groups_result, should_stop = _run_stage(
        stage_name="build_groups",
        cache_key="groups",
        stage_cache=stage_cache,
        compute_fn=lambda: build_groups(samples, config),
        bottleneck_threshold_seconds=bottleneck_threshold_seconds,
        checkpoints=checkpoints,
    )
    if should_stop:
        return {"stopped_at_stage": "build_groups", "checkpoints": checkpoints}

    initial_assignment, should_stop = _run_stage(
        stage_name="assign_groups_to_splits",
        cache_key="assignment",
        stage_cache=stage_cache,
        compute_fn=lambda: assign_groups_to_splits(groups_result["groups"], config),
        bottleneck_threshold_seconds=bottleneck_threshold_seconds,
        checkpoints=checkpoints,
    )
    if should_stop:
        return {"stopped_at_stage": "assign_groups_to_splits", "checkpoints": checkpoints}

    stage_cache["initial_records"] = _flatten_records(initial_assignment["assigned_groups"])
    soft_overlap_report, should_stop = _run_stage(
        stage_name="analyze_soft_overlaps",
        cache_key="soft_overlap",
        stage_cache=stage_cache,
        compute_fn=lambda: analyze_soft_overlaps(stage_cache["initial_records"], config),
        bottleneck_threshold_seconds=bottleneck_threshold_seconds,
        checkpoints=checkpoints,
    )
    if should_stop:
        return {"stopped_at_stage": "analyze_soft_overlaps", "checkpoints": checkpoints}

    repair_result, should_stop = _run_stage(
        stage_name="repair_split_assignments",
        cache_key="repair",
        stage_cache=stage_cache,
        compute_fn=lambda: repair_split_assignments(initial_assignment, soft_overlap_report, config),
        bottleneck_threshold_seconds=bottleneck_threshold_seconds,
        checkpoints=checkpoints,
    )
    if should_stop:
        return {"stopped_at_stage": "repair_split_assignments", "checkpoints": checkpoints}

    stage_cache["final_assignment"] = _summarize_assignment(repair_result["assigned_groups"], repair_result["dropped_groups"])
    stage_cache["final_records"] = _flatten_records(stage_cache["final_assignment"]["assigned_groups"])
    validation_result, should_stop = _run_stage(
        stage_name="validate_split_assignments",
        cache_key="validation",
        stage_cache=stage_cache,
        compute_fn=lambda: validate_split_assignments(stage_cache["final_records"], config),
        bottleneck_threshold_seconds=bottleneck_threshold_seconds,
        checkpoints=checkpoints,
    )
    if should_stop:
        return {"stopped_at_stage": "validate_split_assignments", "checkpoints": checkpoints}

    print(
        "[stage:diagnostics] completed=true summary="
        f"{json.dumps({'stopped_at_stage': None, 'final_status': validation_result['status']}, sort_keys=True)}"
    )
    return {"stopped_at_stage": None, "checkpoints": checkpoints}


def _flatten_records(assigned_groups: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for group in assigned_groups:
        for sample in group["samples"]:
            records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})
    return records


def _summarize_assignment(assigned_groups: list[dict[str, object]], dropped_groups: list[dict[str, object]]) -> dict[str, object]:
    split_counts = {"train": 0, "val": 0, "test": 0}
    emotion_distribution = {"train": Counter(), "val": Counter(), "test": Counter()}
    dataset_distribution = {"train": Counter(), "val": Counter(), "test": Counter()}
    modality_distribution = {"train": Counter(), "val": Counter(), "test": Counter()}
    for group in assigned_groups:
        split_name = group["assigned_split"]
        split_counts[split_name] += group["size"]
        emotion_distribution[split_name].update(group["emotion_counts"])
        dataset_distribution[split_name].update(group["dataset_counts"])
        modality_distribution[split_name].update(group["modality_counts"])
    return {
        "assigned_groups": assigned_groups,
        "dropped_groups": dropped_groups,
        "split_counts": split_counts,
        "emotion_distribution": {split: dict(counts) for split, counts in emotion_distribution.items()},
        "dataset_distribution": {split: dict(counts) for split, counts in dataset_distribution.items()},
        "modality_distribution": {split: dict(counts) for split, counts in modality_distribution.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict leakage-safe train/val/test splits for multimodal metadata")
    parser.add_argument("--input", required=True, help="Path to metadata JSONL or CSV")
    parser.add_argument("--output_dir", required=True, help="Directory for split outputs")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--config", default=None, help="Optional JSON config override path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--diagnose-stages", action="store_true", help="Print stage timings and stop at the first bottleneck threshold")
    parser.add_argument(
        "--bottleneck-threshold-seconds",
        type=float,
        default=None,
        help="Optional elapsed-time threshold that stops diagnostic execution after a bottleneck stage",
    )
    args = parser.parse_args()

    if args.diagnose_stages:
        run_stage_diagnostics(
            input_path=args.input,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            config_path=args.config,
            seed=args.seed,
            batch_size=args.batch_size,
            bottleneck_threshold_seconds=args.bottleneck_threshold_seconds,
        )
        return

    config = load_config(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        config_path=args.config,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    samples = load_metadata(config.input_path)
    groups_result = build_groups(samples, config)
    initial_assignment = assign_groups_to_splits(groups_result["groups"], config)
    initial_records = _flatten_records(initial_assignment["assigned_groups"])
    initial_soft_overlap = analyze_soft_overlaps(initial_records, config)
    repair_result = repair_split_assignments(initial_assignment, initial_soft_overlap, config)
    final_assignment = _summarize_assignment(repair_result["assigned_groups"], repair_result["dropped_groups"])
    assigned_records = _flatten_records(final_assignment["assigned_groups"])
    final_soft_overlap = analyze_soft_overlaps(assigned_records, config)
    output_paths = save_split_outputs(config.output_dir, final_assignment["assigned_groups"], final_assignment["dropped_groups"])
    output_paths["hard_group_report"] = str(write_json(groups_result["hard_group_report"], Path(config.output_dir) / "hard_group_report.json"))
    output_paths["soft_overlap_report"] = str(write_json(final_soft_overlap, Path(config.output_dir) / "soft_overlap_report.json"))

    validation_result = validate_split_assignments(assigned_records, config)
    report = build_split_report(
        config=config_as_dict(config),
        num_input_samples=len(samples),
        num_groups=groups_result["num_groups"],
        hard_group_report=groups_result["hard_group_report"],
        assignment_result=final_assignment,
        soft_overlap_report=final_soft_overlap,
        repair_result=repair_result,
        validation_result=validation_result,
        output_paths=output_paths,
    )
    report_paths = write_report(report, config.output_dir)

    print(validation_result["status"])
    print(json.dumps(validation_result, indent=2, sort_keys=True))
    print(f"Split report JSON: {report_paths['json']}")
    print(f"Split report TXT: {report_paths['txt']}")


if __name__ == "__main__":
    main()