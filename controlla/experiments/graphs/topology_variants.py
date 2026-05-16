"""Construct graph topology variants and perturbations for sensitivity studies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controlla.utils import ensure_directory, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate graph topology variants")
    parser.add_argument("--graph-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--variant", type=str, default="knn_5", choices=["fully_connected", "knn_5", "knn_10", "sparse"])
    parser.add_argument("--perturbation", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _apply_variant(adjacency: np.ndarray, variant: str) -> np.ndarray:
    result = adjacency.copy()
    if variant == "fully_connected":
        result[result > 0] = 1.0
        result[result == 0] = 0.25
        np.fill_diagonal(result, 1.0)
        return result
    degree = 5 if variant == "knn_5" else 10 if variant == "knn_10" else 2
    thresholded = np.zeros_like(result)
    for index in range(result.shape[0]):
        neighbors = np.argsort(-result[index])
        kept = 0
        for neighbor in neighbors:
            if neighbor == index:
                continue
            thresholded[index, neighbor] = result[index, neighbor]
            kept += 1
            if kept >= degree:
                break
    thresholded = np.maximum(thresholded, thresholded.T)
    np.fill_diagonal(thresholded, 1.0)
    return thresholded


def _perturb(adjacency: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction <= 0.0:
        return adjacency
    rng = np.random.default_rng(seed)
    result = adjacency.copy()
    upper = np.transpose(np.nonzero(np.triu(result > 0, 1)))
    if len(upper) == 0:
        return result
    num_changes = max(1, int(len(upper) * fraction))
    drop_indices = rng.choice(len(upper), size=min(num_changes, len(upper)), replace=False)
    for edge_index in drop_indices:
        source, target = upper[edge_index]
        result[source, target] = 0.0
        result[target, source] = 0.0
    return result


def main() -> None:
    args = parse_args()
    adjacency = np.load(args.graph_path)
    variant = _apply_variant(adjacency, args.variant)
    perturbed = _perturb(variant, args.perturbation, args.seed)
    output_dir = ensure_directory(args.output_dir)
    output_path = Path(output_dir) / f"{Path(args.graph_path).stem}_{args.variant}_p{args.perturbation:.2f}.npy"
    np.save(output_path, perturbed.astype(np.float32))
    write_json(
        {
            "base_graph": str(Path(args.graph_path).resolve()),
            "variant": args.variant,
            "perturbation": args.perturbation,
            "seed": args.seed,
        },
        Path(output_dir) / f"{output_path.stem}.json",
    )
    print(f"saved topology variant to {output_path}")


if __name__ == "__main__":
    main()