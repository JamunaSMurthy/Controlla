"""Graph prior utilities for Controlla.

This module builds graph priors used by Controlla:

- emotion / attribute graph G_e
- identity graph G_i
- shortest-path distance matrices D_e and D_i

These graph priors are used by latent graph alignment and traversal modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class GraphPrior:
    """Container for graph prior data.

    Attributes:
        node_features: Node features with shape [N, D].
        adjacency: Weighted adjacency matrix with shape [N, N].
        distance: Shortest-path distance matrix with shape [N, N].
        node_labels: Optional node labels.
    """

    node_features: torch.Tensor
    adjacency: torch.Tensor
    distance: torch.Tensor
    node_labels: list[str] | None = None


def _validate_node_features(node_features: torch.Tensor) -> None:
    if node_features.ndim != 2:
        raise ValueError(
            f"node_features must have shape [N, D], got {tuple(node_features.shape)}"
        )

    if node_features.shape[0] <= 0:
        raise ValueError("node_features must contain at least one node")


def build_knn_adjacency(
    node_features: torch.Tensor,
    k: int = 10,
    metric: Literal["cosine", "euclidean"] = "cosine",
    symmetrize: bool = True,
    self_loops: bool = False,
) -> torch.Tensor:
    """Build weighted kNN adjacency from node features."""

    _validate_node_features(node_features)

    num_nodes = node_features.shape[0]

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    k = min(k, max(num_nodes - 1, 1))

    if metric == "cosine":
        normalized = F.normalize(node_features, dim=-1, eps=1e-6)
        similarity = normalized @ normalized.transpose(0, 1)
        score = similarity
    elif metric == "euclidean":
        distance = torch.cdist(node_features, node_features, p=2)
        score = -distance
        similarity = 1.0 / (1.0 + distance)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if num_nodes > 1:
        score = score.clone()
        score.fill_diagonal_(-float("inf"))

    _, indices = torch.topk(score, k=k, dim=-1)

    adjacency = torch.zeros(
        (num_nodes, num_nodes),
        dtype=node_features.dtype,
        device=node_features.device,
    )

    row_indices = torch.arange(num_nodes, device=node_features.device).unsqueeze(-1)
    adjacency[row_indices, indices] = similarity[row_indices, indices].clamp_min(0.0)

    if symmetrize:
        adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))

    if self_loops:
        adjacency.fill_diagonal_(1.0)
    else:
        adjacency.fill_diagonal_(0.0)

    return adjacency.clamp(0.0, 1.0)


def adjacency_to_distance(
    adjacency: torch.Tensor,
    eps: float = 1e-6,
    disconnected_value: float = 1e6,
) -> torch.Tensor:
    """Convert weighted adjacency to all-pairs shortest-path distance."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"adjacency must have shape [N, N], got {tuple(adjacency.shape)}"
        )

    num_nodes = adjacency.shape[0]

    edge_lengths = torch.where(
        adjacency > 0,
        1.0 / (adjacency + eps),
        torch.full_like(adjacency, disconnected_value),
    )

    edge_lengths.fill_diagonal_(0.0)

    distance = edge_lengths.clone()

    for k in range(num_nodes):
        distance = torch.minimum(
            distance,
            distance[:, k].unsqueeze(1) + distance[k, :].unsqueeze(0),
        )

    finite_mask = distance < disconnected_value
    if finite_mask.any():
        max_finite = distance[finite_mask].max().clamp_min(1.0)
        distance = torch.where(
            finite_mask,
            distance / max_finite,
            torch.ones_like(distance),
        )
    else:
        distance = torch.ones_like(distance)
        distance.fill_diagonal_(0.0)

    return distance


def build_graph_prior(
    node_features: torch.Tensor,
    k: int = 10,
    metric: Literal["cosine", "euclidean"] = "cosine",
    node_labels: list[str] | None = None,
) -> GraphPrior:
    """Build a graph prior from node features."""

    _validate_node_features(node_features)

    if node_labels is not None and len(node_labels) != node_features.shape[0]:
        raise ValueError(
            f"node_labels length {len(node_labels)} does not match "
            f"num_nodes {node_features.shape[0]}"
        )

    adjacency = build_knn_adjacency(
        node_features=node_features,
        k=k,
        metric=metric,
        symmetrize=True,
        self_loops=False,
    )

    distance = adjacency_to_distance(adjacency)

    return GraphPrior(
        node_features=node_features,
        adjacency=adjacency,
        distance=distance,
        node_labels=node_labels,
    )


def build_emotion_graph_prior(
    emotion_features: torch.Tensor,
    emotion_labels: list[str] | None = None,
    k: int = 10,
) -> GraphPrior:
    """Build the affective / emotion graph prior G_e."""

    return build_graph_prior(
        node_features=emotion_features,
        k=k,
        metric="cosine",
        node_labels=emotion_labels,
    )


def build_identity_graph_prior(
    identity_features: torch.Tensor,
    identity_labels: list[str] | None = None,
    k: int = 10,
) -> GraphPrior:
    """Build the reference-identity graph prior G_i."""

    return build_graph_prior(
        node_features=identity_features,
        k=k,
        metric="cosine",
        node_labels=identity_labels,
    )


def save_graph_prior(path: str, graph: GraphPrior) -> None:
    """Save a graph prior to disk."""

    torch.save(
        {
            "node_features": graph.node_features.detach().cpu(),
            "adjacency": graph.adjacency.detach().cpu(),
            "distance": graph.distance.detach().cpu(),
            "node_labels": graph.node_labels,
        },
        path,
    )


def load_graph_prior(path: str, device: torch.device | str | None = None) -> GraphPrior:
    """Load a graph prior from disk."""

    payload = torch.load(path, map_location=device or "cpu")

    return GraphPrior(
        node_features=payload["node_features"],
        adjacency=payload["adjacency"],
        distance=payload["distance"],
        node_labels=payload.get("node_labels"),
    )