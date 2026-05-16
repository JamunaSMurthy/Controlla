"""Run ablation evaluations for Controlla experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from controlla.utils import ensure_directory, load_config, resolve_experiment_manifest_path, write_json

from controlla.experiments.metrics import compute_controllability, compute_disentanglement, compute_identity_preservation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Controlla ablation evaluations")
    parser.add_argument("--config", type=str, default="experiments/configs/eval_ablation.yaml")
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
    for variant in config["variants"]:
        for seed in config["seeds"]:
            metrics = {}
            metrics.update(compute_controllability(str(manifest_path)))
            metrics.update(compute_identity_preservation(str(manifest_path)))
            metrics.update(compute_disentanglement(str(manifest_path), seed=int(seed)))
            metrics.update({"variant": variant["name"], "seed": int(seed), "settings": variant})
            output_path = ensure_directory(results_root / variant["name"] / f"seed_{seed}") / "metrics.json"
            write_json(metrics, output_path)
            print(f"wrote ablation metrics to {output_path}")


if __name__ == "__main__":
    main()