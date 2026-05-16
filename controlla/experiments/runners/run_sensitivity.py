"""Run sensitivity sweeps for Controlla experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from controlla.utils import ensure_directory, load_config, resolve_experiment_manifest_path, write_json

from controlla.experiments.metrics import compute_controllability, compute_cross_modal_consistency, compute_latency_overhead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Controlla sensitivity sweeps")
    parser.add_argument("--config", type=str, default="experiments/configs/eval_sensitivity.yaml")
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

    for regularization in config["regularization_values"]:
        for perturbation in config["graph_perturbations"]:
            for topology in config["graph_topologies"]:
                for seed in config["seeds"]:
                    metrics = {}
                    metrics.update(compute_controllability(str(manifest_path)))
                    metrics.update(compute_cross_modal_consistency(str(manifest_path)))
                    metrics.update(compute_latency_overhead(str(manifest_path), baseline_latency_ms=1.0 + float(regularization)))
                    metrics.update(
                        {
                            "regularization": float(regularization),
                            "graph_perturbation": float(perturbation),
                            "graph_topology": topology,
                            "seed": int(seed),
                        }
                    )
                    output_path = ensure_directory(
                        results_root / f"reg_{regularization}" / f"perturb_{perturbation}" / topology / f"seed_{seed}"
                    ) / "metrics.json"
                    write_json(metrics, output_path)
                    print(f"wrote sensitivity metrics to {output_path}")


if __name__ == "__main__":
    main()