"""Shared graph-construction helpers for Controlla experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from controlla.utils import ensure_directory, write_json


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    return payload


def resolve_manifest_path(
    config: dict[str, Any],
    manifest_path: str | None = None,
) -> Path:
    """Resolve manifest path from CLI override or config."""

    if manifest_path is not None:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path.resolve()

    candidates = [
        config.get("manifest_path"),
        config.get("manifest_output"),
    ]

    datasets = config.get("datasets", {})
    if isinstance(datasets, dict):
        affecthuman = datasets.get("affecthuman43k", {})
        if isinstance(affecthuman, dict):
            candidates.append(affecthuman.get("manifest"))

    for candidate in candidates:
        if candidate:
            path = Path(str(candidate))
            if path.exists():
                return path.resolve()

    raise FileNotFoundError(
        "Could not resolve manifest path. Pass --manifest-path or set "
        "manifest_path / manifest_output / datasets.affecthuman43k.manifest in config."
    )


def load_manifest(
    config_path: str,
    manifest_path: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    """Load experiment graph config and manifest."""

    config = load_yaml(config_path)
    resolved_manifest = resolve_manifest_path(config, manifest_path=manifest_path)

    if resolved_manifest.suffix.lower() == ".parquet":
        manifest = pd.read_parquet(resolved_manifest)
    else:
        manifest = pd.read_csv(resolved_manifest)

    if len(manifest) == 0:
        raise ValueError(f"Manifest is empty: {resolved_manifest}")

    return config, manifest, resolved_manifest


def get_column(
    manifest: pd.DataFrame,
    preferred: str,
    alternatives: list[str] | None = None,
    required: bool = True,
) -> str | None:
    """Resolve a column name with fallbacks."""

    alternatives = alternatives or []
    candidates = [preferred, *alternatives]

    for candidate in candidates:
        if candidate in manifest.columns:
            return candidate

    if required:
        available = ", ".join(manifest.columns)
        raise KeyError(
            f"None of the expected columns {candidates} were found. "
            f"Available columns: {available}"
        )

    return None


def safe_string_series(
    manifest: pd.DataFrame,
    column: str | None,
    default_prefix: str,
) -> list[str]:
    """Return a safe string list from a dataframe column."""

    if column is None:
        return [f"{default_prefix}::{index}" for index in range(len(manifest))]

    values = manifest[column].fillna("").astype(str).tolist()

    return [
        value if value.strip() else f"{default_prefix}::{index}"
        for index, value in enumerate(values)
    ]


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return a cosine similarity matrix."""

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [N, D], got {embeddings.shape}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms

    similarity = normalized @ normalized.T
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)

    return similarity.astype(np.float32)


def knn_adjacency(
    similarity: np.ndarray,
    k: int,
    self_loops: bool = False,
    nonnegative: bool = True,
) -> np.ndarray:
    """Build a symmetric k-nearest-neighbor adjacency matrix."""

    similarity = np.asarray(similarity, dtype=np.float32)

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"similarity must have shape [N, N], got {similarity.shape}")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    num_nodes = similarity.shape[0]
    k = min(k, max(num_nodes - 1, 1))

    adjacency = np.zeros_like(similarity, dtype=np.float32)
    score = similarity.copy()

    if nonnegative:
        score = (score + 1.0) * 0.5

    np.fill_diagonal(score, -np.inf)

    for index in range(num_nodes):
        neighbors = np.argsort(-score[index])[:k]
        for neighbor in neighbors:
            weight = float(score[index, neighbor])
            if np.isfinite(weight) and weight > 0:
                adjacency[index, neighbor] = weight

    adjacency = np.maximum(adjacency, adjacency.T)

    if self_loops:
        np.fill_diagonal(adjacency, 1.0)
    else:
        np.fill_diagonal(adjacency, 0.0)

    return adjacency.astype(np.float32)


def adjacency_to_distance(
    adjacency: np.ndarray,
    disconnected_value: float = 1e6,
) -> np.ndarray:
    """Convert weighted adjacency to normalized shortest-path distance."""

    adjacency = np.asarray(adjacency, dtype=np.float32)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must have shape [N, N], got {adjacency.shape}")

    num_nodes = adjacency.shape[0]

    distance = np.where(
        adjacency > 0,
        1.0 / (adjacency + 1e-8),
        disconnected_value,
    ).astype(np.float32)

    np.fill_diagonal(distance, 0.0)

    for k in range(num_nodes):
        distance = np.minimum(
            distance,
            distance[:, [k]] + distance[[k], :],
        )

    finite = distance < disconnected_value

    if finite.any():
        max_finite = max(float(distance[finite].max()), 1.0)
        distance = np.where(finite, distance / max_finite, 1.0)
    else:
        distance = np.ones_like(distance, dtype=np.float32)
        np.fill_diagonal(distance, 0.0)

    return distance.astype(np.float32)


def edge_list_from_adjacency(
    adjacency: np.ndarray,
    sample_ids: list[str],
) -> pd.DataFrame:
    """Convert dense adjacency into an edge list."""

    rows = []

    for source in range(adjacency.shape[0]):
        for target in range(source + 1, adjacency.shape[1]):
            weight = float(adjacency[source, target])

            if weight <= 0:
                continue

            rows.append(
                {
                    "source": sample_ids[source],
                    "target": sample_ids[target],
                    "weight": weight,
                }
            )

    return pd.DataFrame.from_records(rows)


def infer_sample_ids(manifest: pd.DataFrame) -> list[str]:
    """Infer stable sample IDs from manifest."""

    for column in ["sample_id", "id", "uid"]:
        if column in manifest.columns:
            values = manifest[column].fillna("").astype(str).tolist()
            return [
                value if value.strip() else f"sample_{index:08d}"
                for index, value in enumerate(values)
            ]

    return [f"sample_{index:08d}" for index in range(len(manifest))]


def save_graph(
    output_dir: Path,
    graph_name: str,
    adjacency: np.ndarray,
    manifest: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist adjacency, distance matrix, edge list, and metadata."""

    output_dir = ensure_directory(output_dir)

    adjacency = np.asarray(adjacency, dtype=np.float32)
    distance = adjacency_to_distance(adjacency)

    np.save(output_dir / f"{graph_name}_adjacency.npy", adjacency)
    np.save(output_dir / f"{graph_name}_distance.npy", distance)

    sample_ids = infer_sample_ids(manifest)
    edge_list = edge_list_from_adjacency(adjacency, sample_ids)
    edge_list.to_csv(output_dir / f"{graph_name}_edges.csv", index=False)

    payload = {
        "graph_name": graph_name,
        "num_nodes": int(adjacency.shape[0]),
        "num_edges": int((adjacency > 0).sum() // 2),
        "has_self_loops": bool(np.any(np.diag(adjacency) > 0)),
    }

    if metadata is not None:
        payload.update(metadata)

    write_json(payload, output_dir / f"{graph_name}_metadata.json")


def infer_graph_output_dir(
    config: dict[str, Any],
    explicit_output_dir: str | None = None,
) -> Path:
    """Resolve graph output directory from config or CLI override."""

    if explicit_output_dir is not None:
        return ensure_directory(Path(explicit_output_dir))

    candidates = [
        config.get("graph_output_dir"),
        config.get("output_dir"),
        config.get("processed_root"),
    ]

    for candidate in candidates:
        if candidate:
            return ensure_directory(Path(str(candidate)) / "graphs")

    return ensure_directory(Path("experiments/outputs/graphs"))