"""Run ablation evaluations for Controlla experiments.

This runner evaluates existing prediction manifests for ablation variants.
It does not train variants and does not fabricate predictions. If a prediction
manifest is missing, a metrics file with status="missing_prediction_manifest"
is written unless --skip-missing is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from controlla.experiments.metrics.paper_metrics import (
    compute_core_paper_metrics,
    prepare_prediction_frame,
)
from controlla.utils import ensure_directory, load_config, resolve_project_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Controlla ablation evaluations")
    parser.add_argument("--config", type=str, default="experiments/configs/eval_ablation.yaml")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--prediction-root", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def _resolve_manifest(config: dict[str, Any], project_root: Path, override: str | None) -> Path:
    candidate = override or config.get("manifest_path")
    if candidate is None:
        raise ValueError("Manifest path is required. Pass --manifest-path or set manifest_path in config.")

    path = resolve_project_path(project_root, candidate)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Manifest not found: {candidate}")

    return path


def _prediction_candidates(
    prediction_root: Path,
    variant_name: str,
    split_name: str,
    seed: int,
) -> list[Path]:
    return [
        prediction_root / variant_name / split_name / f"seed_{seed}" / "predictions.csv",
        prediction_root / variant_name / f"seed_{seed}" / "predictions.csv",
        prediction_root / split_name / variant_name / f"seed_{seed}" / "predictions.csv",
    ]


def _resolve_prediction(
    prediction_root: Path,
    variant_name: str,
    split_name: str,
    seed: int,
    fallback_manifest: Path,
    allow_fallback: bool,
) -> tuple[Path | None, str]:
    for candidate in _prediction_candidates(prediction_root, variant_name, split_name, seed):
        if candidate.exists():
            return candidate, "prediction_manifest"

    if allow_fallback:
        return fallback_manifest, "source_manifest_fallback"

    return None, "missing_prediction_manifest"


def _variant_groups(config: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups = []

    for key in ["component_variants", "modality_variants", "traversal_variants"]:
        variants = config.get(key, [])
        if variants:
            groups.append((key.replace("_variants", ""), variants))

    legacy_variants = config.get("variants", [])
    if legacy_variants:
        groups.append(("legacy", legacy_variants))

    return groups


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["project_root"])

    manifest_path = _resolve_manifest(config, project_root, args.manifest_path)

    prediction_root = resolve_project_path(
        project_root,
        args.prediction_root or config.get("prediction_root", "experiments/outputs/predictions/ablations"),
    )
    results_root = resolve_project_path(
        project_root,
        args.results_root or config.get("results_root", "experiments/outputs/results/ablations"),
    )

    if prediction_root is None or results_root is None:
        raise ValueError("Could not resolve prediction_root/results_root")

    ensure_directory(prediction_root)
    ensure_directory(results_root)

    split_name = args.split_name or config.get("split_name", "test")
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    max_samples = args.max_samples
    allow_fallback_names = {"controlla", "controlla_full", "image_text_audio"}

    for group_name, variants in _variant_groups(config):
        for variant in variants:
            variant_name = str(variant["name"])

            for seed in seeds:
                prediction_manifest, status = _resolve_prediction(
                    prediction_root=prediction_root,
                    variant_name=variant_name,
                    split_name=split_name,
                    seed=seed,
                    fallback_manifest=manifest_path,
                    allow_fallback=variant_name in allow_fallback_names,
                )

                if prediction_manifest is None and args.skip_missing:
                    continue

                metrics: dict[str, Any] = {
                    "section": f"ablations/{group_name}",
                    "variant": variant_name,
                    "method": variant_name,
                    "split": split_name,
                    "seed": seed,
                    "status": status,
                    "prediction_manifest": str(prediction_manifest) if prediction_manifest is not None else "",
                    "settings": variant,
                }

                if prediction_manifest is not None:
                    frame = prepare_prediction_frame(
                        prediction_manifest,
                        split=split_name,
                        max_samples=max_samples,
                    )
                    metrics.update(compute_core_paper_metrics(frame, seed=seed))
                    metrics["num_samples"] = len(frame)

                output_path = (
                    ensure_directory(results_root / group_name / variant_name / f"seed_{seed}")
                    / "metrics.json"
                )
                write_json(metrics, output_path)
                print(f"wrote ablation metrics to {output_path}")


if __name__ == "__main__":
    main()