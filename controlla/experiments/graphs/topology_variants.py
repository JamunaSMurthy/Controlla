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
    parser.add_argument(
        "--variant",
        type=str,
        default="knn_10",
        choices=[
            "fully_connected",
            "knn_5",
            "knn_10",
            "knn_20",
            "sparse",
            "random_graph",
        ],
    )
    parser.add_argument("--perturbation", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _knn_from_adjacency(adjacency: np.ndarray, degree: int) -> np.ndarray:
    result = np.zeros_like(adjacency, dtype=np.float32)

    for index in range(adjacency.shape[0]):
        scores = adjacency[index].copy()
        scores[index] = -np.inf
        neighbors = np.argsort(-scores)[:degree]

        for neighbor in neighbors:
            if np.isfinite(scores[neighbor]) and scores[neighbor] > 0:
                result[index, neighbor] = adjacency[index, neighbor]

    result = np.maximum(result, result.T)
    np.fill_diagonal(result, 0.0)

    return result.astype(np.float32)


def _random_graph_like(adjacency: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    num_nodes = adjacency.shape[0]
    num_edges = int(np.triu(adjacency > 0, 1).sum())

    result = np.zeros_like(adjacency, dtype=np.float32)

    if num_edges == 0:
        return result

    possible_edges = [
        (i, j)
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
    ]

    chosen = rng.choice(
        len(possible_edges),
        size=min(num_edges, len(possible_edges)),
        replace=False,
    )

    positive_weights = adjacency[np.triu(adjacency > 0, 1)]
    if positive_weights.size == 0:
        positive_weights = np.asarray([1.0], dtype=np.float32)

    sampled_weights = rng.choice(
        positive_weights,
        size=len(chosen),
        replace=True,
    )

    for edge_idx, weight in zip(chosen, sampled_weights):
        source, target = possible_edges[int(edge_idx)]
        result[source, target] = float(weight)
        result[target, source] = float(weight)

    np.fill_diagonal(result, 0.0)

    return result.astype(np.float32)


def apply_topology_variant(
    adjacency: np.ndarray,
    variant: str,
    seed: int = 0,
) -> np.ndarray:
    """Apply a graph topology variant."""

    adjacency = np.asarray(adjacency, dtype=np.float32)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must have shape [N, N], got {adjacency.shape}")

    result = adjacency.copy()
    np.fill_diagonal(result, 0.0)

    if variant == "fully_connected":
        positive = result[result > 0]
        fill_weight = float(np.median(positive)) if positive.size else 0.25
        result = np.full_like(result, fill_weight, dtype=np.float32)
        np.fill_diagonal(result, 0.0)
        return result

    if variant == "knn_5":
        return _knn_from_adjacency(result, degree=5)

    if variant == "knn_10":
        return _knn_from_adjacency(result, degree=10)

    if variant == "knn_20":
        return _knn_from_adjacency(result, degree=20)

    if variant == "sparse":
        return _knn_from_adjacency(result, degree=2)

    if variant == "random_graph":
        return _random_graph_like(result, seed=seed)

    raise ValueError(f"Unsupported topology variant: {variant}")


def perturb_adjacency(
    adjacency: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Randomly remove a fraction of graph edges."""

    if fraction <= 0.0:
        return adjacency.astype(np.float32)

    if fraction > 1.0:
        raise ValueError(f"perturbation fraction must be <= 1.0, got {fraction}")

    rng = np.random.default_rng(seed)
    result = adjacency.copy().astype(np.float32)

    upper_edges = np.transpose(np.nonzero(np.triu(result > 0, 1)))

    if len(upper_edges) == 0:
        return result

    num_changes = max(1, int(len(upper_edges) * fraction))
    drop_indices = rng.choice(
        len(upper_edges),
        size=min(num_changes, len(upper_edges)),
        replace=False,
    )

    for edge_index in drop_indices:
        source, target = upper_edges[edge_index]
        result[source, target] = 0.0
        result[target, source] = 0.0

    return result.astype(np.float32)


def main() -> None:
    args = parse_args()

    adjacency = np.load(args.graph_path)

    variant = apply_topology_variant(
        adjacency=adjacency,
        variant=args.variant,
        seed=args.seed,
    )

    perturbed = perturb_adjacency(
        adjacency=variant,
        fraction=args.perturbation,
        seed=args.seed,
    )

    output_dir = ensure_directory(args.output_dir)

    output_path = (
        Path(output_dir)
        / f"{Path(args.graph_path).stem}_{args.variant}_p{args.perturbation:.2f}.npy"
    )

    np.save(output_path, perturbed.astype(np.float32))

    write_json(
        {
            "base_graph": str(Path(args.graph_path).resolve()),
            "variant": args.variant,
            "perturbation": float(args.perturbation),
            "seed": int(args.seed),
            "num_nodes": int(perturbed.shape[0]),
            "num_edges": int(np.triu(perturbed > 0, 1).sum()),
        },
        Path(output_dir) / f"{output_path.stem}.json",
    )

    print(f"saved topology variant to {output_path}")


if __name__ == "__main__":
    main()