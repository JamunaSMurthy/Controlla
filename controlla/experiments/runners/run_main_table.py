"""Run the main evaluation table for Controlla experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from controlla.utils import ensure_directory, load_config, resolve_experiment_manifest_path, write_json

from controlla.experiments.metrics import (
    compute_controllability,
    compute_cross_modal_consistency,
    compute_disentanglement,
    compute_identity_preservation,
    compute_latency_overhead,
)


def _prediction_manifest(results_root: Path, method: str, seed: int, fallback_manifest: Path) -> Path:
    candidate = results_root / method / f"seed_{seed}" / "predictions.csv"
    return candidate if candidate.exists() else fallback_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Controlla main evaluation table")
    parser.add_argument("--config", type=str, default="experiments/configs/eval_main.yaml")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project_root = Path(config["project_root"])
    manifest_path = resolve_experiment_manifest_path(
        config,
        manifest_path=args.manifest_path,
        split_dir=args.split_dir,
        split_name=args.split_name,
    )
    results_root = ensure_directory(project_root / config["results_root"])

    for method in config["methods"]:
        for seed in config["seeds"]:
            prediction_manifest = _prediction_manifest(results_root, method, int(seed), manifest_path)
            metrics = {}
            metrics.update(compute_controllability(str(prediction_manifest)))
            metrics.update(compute_identity_preservation(str(prediction_manifest)))
            metrics.update(compute_cross_modal_consistency(str(prediction_manifest)))
            metrics.update(compute_disentanglement(str(prediction_manifest), seed=int(seed)))
            metrics.update(compute_latency_overhead(str(prediction_manifest)))
            metrics.update(
                {
                    "method": method,
                    "seed": int(seed),
                    "prediction_manifest": str(prediction_manifest),
                    "used_fallback_manifest": prediction_manifest == manifest_path,
                }
            )
            output_path = ensure_directory(results_root / method / f"seed_{seed}") / "metrics.json"
            write_json(metrics, output_path)
            print(f"wrote main metrics to {output_path}")


if __name__ == "__main__":
    main()