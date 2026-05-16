"""Human-readable and machine-readable reports for the split builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_split_report(
    *,
    config: dict[str, Any],
    num_input_samples: int,
    num_groups: int,
    hard_group_report: dict[str, Any],
    assignment_result: dict[str, Any],
    soft_overlap_report: dict[str, Any],
    repair_result: dict[str, Any],
    validation_result: dict[str, Any],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "config": config,
        "num_input_samples": num_input_samples,
        "num_groups": num_groups,
        "num_dropped_groups": len(assignment_result["dropped_groups"]),
        "hard_group_report": hard_group_report,
        "split_counts": assignment_result["split_counts"],
        "emotion_distribution": assignment_result["emotion_distribution"],
        "dataset_distribution": assignment_result["dataset_distribution"],
        "modality_distribution": assignment_result["modality_distribution"],
        "soft_overlap_report": soft_overlap_report,
        "repair_summary": repair_result["repair_summary"],
        "validation": validation_result,
        "output_paths": output_paths,
        "status": validation_result["status"],
    }


def render_text_report(report: dict[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "Controlla Leakage-Safe Split Report",
        "=",
        "",
        f"Status: {validation['status']}",
        f"Input samples: {report['num_input_samples']}",
        f"Hard groups: {report['num_groups']}",
        f"Largest hard group size: {report['hard_group_report']['largest_hard_group_size']}",
        f"Dropped groups: {report['num_dropped_groups']}",
        "",
        "Hard edge counts by rule:",
        json.dumps(report["hard_group_report"]["edge_counts_by_rule"], indent=2, sort_keys=True),
        "",
        "Split sizes:",
        json.dumps(report["split_counts"], indent=2, sort_keys=True),
        "",
        "Soft overlap warnings:",
        json.dumps(report["soft_overlap_report"]["warning_counts_by_type"], indent=2, sort_keys=True),
        "",
        "Repair summary:",
        json.dumps(report["repair_summary"], indent=2, sort_keys=True),
        "",
        "Validation metrics:",
        json.dumps(validation, indent=2, sort_keys=True),
        "",
        "Emotion distribution:",
        json.dumps(report["emotion_distribution"], indent=2, sort_keys=True),
        "",
        "Dataset source distribution:",
        json.dumps(report["dataset_distribution"], indent=2, sort_keys=True),
        "",
        "Output files:",
        json.dumps(report["output_paths"], indent=2, sort_keys=True),
    ]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target_dir = Path(output_dir)
    json_path = target_dir / "split_report.json"
    txt_path = target_dir / "split_report.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(render_text_report(report), encoding="utf-8")
    return {"json": str(json_path), "txt": str(txt_path)}