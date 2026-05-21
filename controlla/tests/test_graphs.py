"""Tests for Controlla graph utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from controlla.experiments.graphs.graph_utils import (
    adjacency_to_distance,
    cosine_similarity_matrix,
    knn_adjacency,
    save_graph,
)
from controlla.experiments.graphs.topology_variants import (
    apply_topology_variant,
    perturb_adjacency,
)


def test_knn_adjacency_and_distance() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    similarity = cosine_similarity_matrix(embeddings)
    adjacency = knn_adjacency(similarity, k=1)

    assert adjacency.shape == (4, 4)
    assert np.allclose(adjacency, adjacency.T)
    assert np.allclose(np.diag(adjacency), 0.0)

    distance = adjacency_to_distance(adjacency)

    assert distance.shape == (4, 4)
    assert np.allclose(np.diag(distance), 0.0)


def test_topology_variants() -> None:
    adjacency = np.asarray(
        [
            [0.0, 0.9, 0.2],
            [0.9, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ],
        dtype=np.float32,
    )

    fully_connected = apply_topology_variant(adjacency, "fully_connected")
    knn_5 = apply_topology_variant(adjacency, "knn_5")
    random_graph = apply_topology_variant(adjacency, "random_graph", seed=0)
    perturbed = perturb_adjacency(adjacency, fraction=0.5, seed=0)

    assert fully_connected.shape == adjacency.shape
    assert knn_5.shape == adjacency.shape
    assert random_graph.shape == adjacency.shape
    assert perturbed.shape == adjacency.shape


def test_save_graph_outputs(tmp_path: Path) -> None:
    adjacency = np.asarray(
        [
            [0.0, 0.9],
            [0.9, 0.0],
        ],
        dtype=np.float32,
    )

    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_path": ["a.png", "b.png"],
        }
    )

    save_graph(
        output_dir=tmp_path,
        graph_name="test_graph",
        adjacency=adjacency,
        manifest=manifest,
        metadata={"k": 1},
    )

    assert (tmp_path / "test_graph_adjacency.npy").exists()
    assert (tmp_path / "test_graph_distance.npy").exists()
    assert (tmp_path / "test_graph_edges.csv").exists()
    assert (tmp_path / "test_graph_metadata.json").exists()