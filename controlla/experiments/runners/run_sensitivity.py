"""Run sensitivity evaluations for Controlla experiments.

This runner evaluates existing prediction manifests for sensitivity variants:
OT regularization, graph perturbation, and graph topology.

It does not generate predictions itself.
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
    parser = argparse.ArgumentParser(description="Run Controlla sensitivity evaluations")
    parser.add_argument("--config", type=str, default="experiments/configs/eval_sensitivity.yaml")
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
    topology_name: str,
    regularization: float,
    perturbation: float,
    split_name: str,
    seed: int,
) -> list[Path]:
    reg = f"reg_{regularization}"
    perturb = f"perturb_{perturbation}"

    return [
        prediction_root / reg / perturb / topology_name / split_name / f"seed_{seed}" / "predictions.csv",
        prediction_root / reg / perturb / topology_name / f"seed_{seed}" / "predictions.csv",
        prediction_root / topology_name / reg / perturb / f"seed_{seed}" / "predictions.csv",
    ]


def _resolve_prediction(
    prediction_root: Path,
    topology_name: str,
    regularization: float,
    perturbation: float,
    split_name: str,
    seed: int,
    fallback_manifest: Path,
    allow_fallback: bool,
) -> tuple[Path | None, str]:
    for candidate in _prediction_candidates(
        prediction_root=prediction_root,
        topology_name=topology_name,
        regularization=regularization,
        perturbation=perturbation,
        split_name=split_name,
        seed=seed,
    ):
        if candidate.exists():
            return candidate, "prediction_manifest"

    if allow_fallback:
        return fallback_manifest, "source_manifest_fallback"

    return None, "missing_prediction_manifest"


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["project_root"])

    manifest_path = _resolve_manifest(config, project_root, args.manifest_path)

    prediction_root = resolve_project_path(
        project_root,
        args.prediction_root or config.get("prediction_root", "experiments/outputs/predictions/sensitivity"),
    )
    results_root = resolve_project_path(
        project_root,
        args.results_root or config.get("results_root", "experiments/outputs/results/sensitivity"),
    )

    if prediction_root is None or results_root is None:
        raise ValueError("Could not resolve prediction_root/results_root")

    ensure_directory(prediction_root)
    ensure_directory(results_root)

    split_name = args.split_name or config.get("split_name", "test")
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    max_samples = args.max_samples

    regularization_values = config.get("ot_regularization_values", config.get("regularization_values", [0.05]))
    perturbations = config.get("graph_perturbations", [0.0])
    graph_topologies = config.get("graph_topologies", [])

    for regularization in regularization_values:
        for perturbation in perturbations:
            for topology in graph_topologies:
                topology_name = str(topology["name"] if isinstance(topology, dict) else topology)

                for seed in seeds:
                    prediction_manifest, status = _resolve_prediction(
                        prediction_root=prediction_root,
                        topology_name=topology_name,
                        regularization=float(regularization),
                        perturbation=float(perturbation),
                        split_name=split_name,
                        seed=seed,
                        fallback_manifest=manifest_path,
                        allow_fallback=topology_name in {"knn_10"},
                    )

                    if prediction_manifest is None and args.skip_missing:
                        continue

                    metrics: dict[str, Any] = {
                        "section": "sensitivity",
                        "method": "controlla",
                        "variant": topology_name,
                        "split": split_name,
                        "seed": seed,
                        "status": status,
                        "ot_regularization": float(regularization),
                        "regularization": float(regularization),
                        "graph_perturbation": float(perturbation),
                        "graph_topology": topology_name,
                        "prediction_manifest": str(prediction_manifest) if prediction_manifest is not None else "",
                    }

                    if isinstance(topology, dict):
                        metrics["topology_settings"] = topology

                    if prediction_manifest is not None:
                        frame = prepare_prediction_frame(
                            prediction_manifest,
                            split=split_name,
                            max_samples=max_samples,
                        )
                        metrics.update(compute_core_paper_metrics(frame, seed=seed))
                        metrics["num_samples"] = len(frame)

                    output_path = (
                        ensure_directory(
                            results_root
                            / f"reg_{regularization}"
                            / f"perturb_{perturbation}"
                            / topology_name
                            / f"seed_{seed}"
                        )
                        / "metrics.json"
                    )
                    write_json(metrics, output_path)
                    print(f"wrote sensitivity metrics to {output_path}")


if __name__ == "__main__":
    main()