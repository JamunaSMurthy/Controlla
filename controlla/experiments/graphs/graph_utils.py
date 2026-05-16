"""Shared graph-construction helpers for Controlla experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from controlla.utils import ensure_directory, load_config, load_manifest_frame, resolve_experiment_manifest_path, write_json


def load_manifest(config_path: str, manifest_path: str | None = None) -> tuple[dict, pd.DataFrame, Path]:
    """Load a prepared experiment manifest."""
    config = load_config(config_path)
    resolved_manifest = resolve_experiment_manifest_path(config, manifest_path=manifest_path)
    manifest = load_manifest_frame(resolved_manifest)
    return config, manifest, resolved_manifest.resolve()


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return a cosine similarity matrix for normalized or unnormalized embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms
    return normalized @ normalized.T


def knn_adjacency(similarity: np.ndarray, k: int) -> np.ndarray:
    """Build a symmetric k-nearest-neighbor adjacency matrix."""
    num_nodes = similarity.shape[0]
    adjacency = np.zeros_like(similarity, dtype=np.float32)
    for index in range(num_nodes):
        neighbors = np.argsort(-similarity[index])
        added = 0
        for neighbor in neighbors:
            if neighbor == index:
                continue
            adjacency[index, neighbor] = float(similarity[index, neighbor])
            added += 1
            if added >= k:
                break
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def edge_list_from_adjacency(adjacency: np.ndarray, sample_ids: list[str]) -> pd.DataFrame:
    """Convert a dense adjacency matrix into an edge list."""
    rows = []
    for source in range(adjacency.shape[0]):
        for target in range(source + 1, adjacency.shape[1]):
            weight = float(adjacency[source, target])
            if weight <= 0:
                continue
            rows.append({"source": sample_ids[source], "target": sample_ids[target], "weight": weight})
    return pd.DataFrame.from_records(rows)


def save_graph(output_dir: Path, graph_name: str, adjacency: np.ndarray, manifest: pd.DataFrame, metadata: dict | None = None) -> None:
    """Persist adjacency, edge list, and metadata for a graph artifact."""
    ensure_directory(output_dir)
    np.save(output_dir / f"{graph_name}_adjacency.npy", adjacency.astype(np.float32))
    edge_list = edge_list_from_adjacency(adjacency, manifest["sample_id"].astype(str).tolist())
    edge_list.to_csv(output_dir / f"{graph_name}_edges.csv", index=False)
    if metadata is not None:
        write_json(metadata, output_dir / f"{graph_name}_metadata.json")


def infer_graph_output_dir(config: dict, explicit_output_dir: str | None = None) -> Path:
    """Resolve the graph output directory from config or CLI override."""
    project_root = Path(config["project_root"])
    output_dir = Path(explicit_output_dir) if explicit_output_dir else project_root / "experiments/outputs/graphs"
    return ensure_directory(output_dir)