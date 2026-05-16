"""Report generation for dataset leakage checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity_leakage": report["identity_leakage"]["leakage_percent"],
        "speaker_leakage": report["speaker_leakage"]["leakage_percent"],
        "image_duplicates": report["duplicate_detection"]["image_duplicates"]["count"],
        "audio_duplicates": report["duplicate_detection"]["audio_duplicates"]["exact_duplicates"]["count"]
        + report["duplicate_detection"]["audio_duplicates"]["near_duplicates"]["count"],
        "text_duplicates": report["duplicate_detection"]["text_duplicates"]["exact_duplicates"]["count"]
        + report["duplicate_detection"]["text_duplicates"]["fuzzy_duplicates"]["count"],
        "cross_dataset_overlap": report["duplicate_detection"]["cross_dataset_overlap"]["count"],
    }


def determine_status(summary: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, str]:
    major = (
        summary["identity_leakage"] >= thresholds["major_percent"]
        or summary["speaker_leakage"] >= thresholds["major_percent"]
        or summary["image_duplicates"] >= thresholds["major_count"]
        or summary["audio_duplicates"] >= thresholds["major_count"]
        or summary["text_duplicates"] >= thresholds["major_count"]
        or summary["cross_dataset_overlap"] >= thresholds["major_count"]
    )
    if major:
        return "FAIL", "❌ FAIL"
    minor = (
        summary["identity_leakage"] >= thresholds["minor_percent"]
        or summary["speaker_leakage"] >= thresholds["minor_percent"]
        or summary["image_duplicates"] >= thresholds["minor_count"]
        or summary["audio_duplicates"] >= thresholds["minor_count"]
        or summary["text_duplicates"] >= thresholds["minor_count"]
        or summary["cross_dataset_overlap"] >= thresholds["minor_count"]
    )
    if minor:
        return "WARNING", "⚠ WARNING"
    return "PASS", "✔ PASS"


def render_text_report(report: dict[str, Any], summary: dict[str, Any], status_line: str) -> str:
    lines = [
        "Controlla Dataset Leakage Report",
        "=",
        "",
        f"Status: {status_line}",
        "",
        "Summary:",
        json.dumps(summary, indent=2, sort_keys=True),
        "",
        f"Identity leakage: {report['identity_leakage']['leakage_percent']}%",
        f"Speaker leakage: {report['speaker_leakage']['leakage_percent']}%",
        f"Image duplicate findings: {report['duplicate_detection']['image_duplicates']['count']}",
        f"Audio duplicate findings: {summary['audio_duplicates']}",
        f"Text duplicate findings: {summary['text_duplicates']}",
        f"Cross-dataset overlap findings: {summary['cross_dataset_overlap']}",
        "",
        "Example identity leaks:",
        json.dumps(report['identity_leakage']['explicit_identity_id']['examples'], indent=2),
        "",
        "Example speaker leaks:",
        json.dumps(report['speaker_leakage']['explicit_speaker_id']['examples'], indent=2),
        "",
        "Example image duplicates:",
        json.dumps(report['duplicate_detection']['image_duplicates']['near_duplicates']['examples'], indent=2),
        "",
        "Example cross-modal overlap:",
        json.dumps(report['cross_modal_leakage']['identity_emotion_overlap']['examples'], indent=2),
    ]
    return "\n".join(lines) + "\n"


def write_reports(
    output_dir: str | Path,
    report: dict[str, Any],
    summary: dict[str, Any],
    status: str,
    status_line: str,
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "leakage_report.json"
    txt_path = target_dir / "leakage_report.txt"
    full_payload = {"status": status, "status_line": status_line, "summary": summary, **report}
    json_path.write_text(json.dumps(full_payload, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(render_text_report(report, summary, status_line), encoding="utf-8")
    return {"json": str(json_path), "txt": str(txt_path)}