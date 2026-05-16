"""CLI entrypoint for multimodal dataset leakage analysis.

Usage:
    python main.py --data path/to/data.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .artifacts import export_leakage_examples
    from .duplicate_detection import run_duplicate_checks
    from .embedding_utils import audio_exact_hash, jsonl_records
    from .identity_leakage import run_identity_leakage_check
    from .remediation import apply_remediation_plan, build_remediation_plan, write_remediation_outputs
    from .report import determine_status, summarize_report, write_reports
    from .speaker_leakage import run_speaker_and_cross_modal_checks
except ImportError:
    from artifacts import export_leakage_examples
    from duplicate_detection import run_duplicate_checks
    from embedding_utils import audio_exact_hash, jsonl_records
    from identity_leakage import run_identity_leakage_check
    from remediation import apply_remediation_plan, build_remediation_plan, write_remediation_outputs
    from report import determine_status, summarize_report, write_reports
    from speaker_leakage import run_speaker_and_cross_modal_checks


DEFAULT_CONFIG: dict[str, Any] = {
    "batch_size": 128,
    "image_embedding_dim": 64,
    "audio_embedding_dim": 64,
    "text_embedding_dim": 64,
    "identity_similarity_threshold": 0.93,
    "speaker_similarity_threshold": 0.94,
    "image_similarity_threshold": 0.98,
    "audio_similarity_threshold": 0.985,
    "text_similarity_threshold": 0.96,
    "text_fuzzy_ratio_threshold": 0.97,
    "image_phash_hamming_threshold": 4,
    "image_phash_prefix_length": 6,
    "max_bucket_size": 256,
    "max_pair_comparisons_per_bucket": 4096,
    "enable_identity": True,
    "enable_speaker": True,
    "enable_duplicates": True,
    "enable_cross_modal": True,
    "enable_cross_dataset": True,
    "severity_thresholds": {
        "minor_percent": 0.1,
        "major_percent": 1.0,
        "minor_count": 1,
        "major_count": 10,
    },
}


def _merge_config(config_path: str | None, batch_size: int | None, enabled_checks: list[str] | None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path:
        loaded = json.loads(Path(config_path).read_text(encoding="utf-8"))
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    if batch_size is not None:
        config["batch_size"] = batch_size
    if enabled_checks is not None:
        enabled = set(enabled_checks)
        config["enable_identity"] = "identity" in enabled
        config["enable_speaker"] = "speaker" in enabled
        config["enable_duplicates"] = "duplicates" in enabled
        config["enable_cross_modal"] = "cross_modal" in enabled
        config["enable_cross_dataset"] = "cross_dataset" in enabled or config["enable_duplicates"]
    return config


def _derive_dataset_source(raw: dict[str, Any], field: str) -> str | None:
    explicit = raw.get(field)
    if explicit:
        return str(explicit)
    nested = raw.get("dataset_sources") or {}
    if field == "dataset_source":
        candidates = [nested.get("image_source"), nested.get("audio_source"), nested.get("text_source")]
        available = [candidate for candidate in candidates if candidate]
        if not available:
            return None
        if len(set(available)) == 1:
            return str(available[0])
        return "+".join(sorted({str(item) for item in available}))
    return nested.get(field.replace("dataset_", "")) or nested.get(field)


def normalize_sample(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    normalized["dataset_source"] = _derive_dataset_source(raw, "dataset_source")
    nested = raw.get("dataset_sources") or {}
    normalized["image_dataset_source"] = raw.get("image_dataset_source") or nested.get("image_source") or normalized["dataset_source"]
    normalized["audio_dataset_source"] = raw.get("audio_dataset_source") or nested.get("audio_source") or normalized["dataset_source"]
    normalized["text_dataset_source"] = raw.get("text_dataset_source") or nested.get("text_source") or normalized["dataset_source"]
    normalized["text"] = raw.get("text") or raw.get("raw_text") or ""
    normalized["audio_exact_hash"] = audio_exact_hash(raw.get("audio_path")) if raw.get("audio_path") else None
    return normalized


def load_samples(data_path: str | Path) -> list[dict[str, Any]]:
    return [normalize_sample(record) for record in jsonl_records(data_path)]


def run_all_checks(samples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    identity_leakage = run_identity_leakage_check(samples, config) if config["enable_identity"] else {"leakage_percent": 0.0, "explicit_identity_id": {"examples": []}, "fallback_embedding_clusters": {"examples": []}}
    speaker_results = run_speaker_and_cross_modal_checks(samples, config) if config["enable_speaker"] or config["enable_cross_modal"] else {
        "speaker_leakage": {"leakage_percent": 0.0, "explicit_speaker_id": {"examples": []}, "fallback_embedding_clusters": {"examples": []}},
        "cross_modal_leakage": {},
    }
    duplicate_results = run_duplicate_checks(samples, config) if config["enable_duplicates"] else {
        "image_duplicates": {"count": 0, "near_duplicates": {"examples": []}},
        "audio_duplicates": {"exact_duplicates": {"count": 0}, "near_duplicates": {"count": 0}},
        "text_duplicates": {"exact_duplicates": {"count": 0}, "fuzzy_duplicates": {"count": 0}},
        "cross_dataset_overlap": {"count": 0},
    }
    report = {
        "num_samples": len(samples),
        "identity_leakage": identity_leakage,
        "speaker_leakage": speaker_results["speaker_leakage"],
        "duplicate_detection": duplicate_results,
        "cross_modal_leakage": speaker_results["cross_modal_leakage"],
    }
    summary = summarize_report(report)
    status, status_line = determine_status(summary, config["severity_thresholds"])
    return {
        "report": report,
        "summary": summary,
        "status": status,
        "status_line": status_line,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect train/val/test leakage in multimodal JSONL datasets")
    parser.add_argument("--data", required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output-dir", default=None, help="Directory for leakage_report.json and leakage_report.txt")
    parser.add_argument("--config", default=None, help="Optional JSON config file")
    parser.add_argument("--batch-size", type=int, default=None, help="Override embedding batch size")
    parser.add_argument("--write-remediation-plan", action="store_true", help="Write a proposed split-fix remediation plan")
    parser.add_argument("--apply-remediation", action="store_true", help="Write a remediated JSONL with proposed split fixes applied")
    parser.add_argument("--export-example-artifacts", action="store_true", help="Export example leaked image/audio/text pairs for inspection")
    parser.add_argument("--max-example-artifacts", type=int, default=10, help="Maximum examples per leakage type to export")
    parser.add_argument(
        "--checks",
        nargs="*",
        choices=["identity", "speaker", "duplicates", "cross_modal", "cross_dataset"],
        default=None,
        help="Run only the selected checks",
    )
    args = parser.parse_args()

    config = _merge_config(args.config, args.batch_size, args.checks)
    samples = load_samples(args.data)
    results = run_all_checks(samples, config)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data).resolve().parent / "leakage_reports"
    report_paths = write_reports(output_dir, results["report"], results["summary"], results["status"], results["status_line"])

    if args.write_remediation_plan or args.apply_remediation:
        remediation_plan = build_remediation_plan(results["report"], samples)
        remediated_samples = apply_remediation_plan(samples, remediation_plan) if args.apply_remediation else None
        remediation_paths = write_remediation_outputs(output_dir, remediation_plan, remediated_samples)
        print(f"Remediation plan: {remediation_paths['plan_json']}")
        if "remediated_jsonl" in remediation_paths:
            print(f"Remediated dataset: {remediation_paths['remediated_jsonl']}")

    if args.export_example_artifacts:
        artifact_summary = export_leakage_examples(
            samples,
            results["report"],
            output_dir / "example_artifacts",
            max_examples_per_type=args.max_example_artifacts,
        )
        print(f"Artifact manifest: {artifact_summary['manifest']}")

    print(results["status_line"])
    print(json.dumps(results["summary"], indent=2, sort_keys=True))
    print(f"JSON report: {report_paths['json']}")
    print(f"Text report: {report_paths['txt']}")


if __name__ == "__main__":
    main()