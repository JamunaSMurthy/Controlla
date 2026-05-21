"""Graph-consistent traversal utilities for Controlla.

This module supports graph-guided attribute traversal while keeping z_id fixed.
It also provides simple traversal alternatives for ablation:
linear, spline-like, random path, and graph shortest path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class TraversalOutput:
    """Output from latent traversal.

    Attributes:
        z_attr_path: Tensor with shape [T, Z] or [B, T, Z].
        node_path: List of graph node indices.
    """

    z_attr_path: torch.Tensor
    node_path: list[int]


def shortest_path(
    distance: torch.Tensor,
    source: int,
    target: int,
) -> list[int]:
    """Compute a shortest path using Floyd-Warshall predecessor reconstruction."""

    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"distance must have shape [N, N], got {tuple(distance.shape)}")

    num_nodes = distance.shape[0]

    if not 0 <= source < num_nodes:
        raise ValueError(f"source index out of range: {source}")
    if not 0 <= target < num_nodes:
        raise ValueError(f"target index out of range: {target}")

    dist = distance.clone()
    next_node = torch.full(
        (num_nodes, num_nodes),
        -1,
        dtype=torch.long,
        device=distance.device,
    )

    for i in range(num_nodes):
        for j in range(num_nodes):
            if torch.isfinite(dist[i, j]):
                next_node[i, j] = j

    for k in range(num_nodes):
        candidate = dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0)
        improve = candidate < dist
        dist = torch.where(improve, candidate, dist)
        next_node = torch.where(
            improve,
            next_node[:, k].unsqueeze(1).expand_as(next_node),
            next_node,
        )

    if next_node[source, target].item() < 0:
        return [source, target]

    path = [source]
    current = source

    while current != target:
        current = int(next_node[current, target].item())
        if current < 0:
            return [source, target]
        path.append(current)
        if len(path) > num_nodes + 1:
            return [source, target]

    return path


def interpolate_linear(
    start: torch.Tensor,
    end: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Linear interpolation between two latent vectors."""

    if steps <= 1:
        raise ValueError(f"steps must be > 1, got {steps}")

    weights = torch.linspace(
        0.0,
        1.0,
        steps,
        device=start.device,
        dtype=start.dtype,
    )

    return (1.0 - weights).unsqueeze(-1) * start + weights.unsqueeze(-1) * end


def graph_consistent_traversal(
    z_attr_start: torch.Tensor,
    graph_node_latents: torch.Tensor,
    graph_distance: torch.Tensor,
    source_node: int,
    target_node: int,
    steps_per_edge: int = 4,
) -> TraversalOutput:
    """Traverse z_attr along a shortest path in graph-node latent space."""

    if z_attr_start.ndim != 1:
        raise ValueError(
            f"z_attr_start must have shape [Z], got {tuple(z_attr_start.shape)}"
        )

    if graph_node_latents.ndim != 2:
        raise ValueError(
            f"graph_node_latents must have shape [N, Z], got {tuple(graph_node_latents.shape)}"
        )

    if graph_node_latents.shape[-1] != z_attr_start.shape[-1]:
        raise ValueError(
            f"Latent dim mismatch: z_attr_start has {z_attr_start.shape[-1]}, "
            f"graph_node_latents has {graph_node_latents.shape[-1]}"
        )

    if steps_per_edge <= 0:
        raise ValueError(f"steps_per_edge must be positive, got {steps_per_edge}")

    node_path = shortest_path(graph_distance, source_node, target_node)

    segments: list[torch.Tensor] = []

    current = z_attr_start

    for path_index, node_idx in enumerate(node_path[1:]):
        target = graph_node_latents[node_idx]
        segment = interpolate_linear(current, target, steps_per_edge + 1)

        if path_index > 0:
            segment = segment[1:]

        segments.append(segment)
        current = target

    if not segments:
        z_path = z_attr_start.unsqueeze(0)
    else:
        z_path = torch.cat(segments, dim=0)

    return TraversalOutput(z_attr_path=z_path, node_path=node_path)


def random_path_traversal(
    z_attr_start: torch.Tensor,
    graph_node_latents: torch.Tensor,
    steps: int,
) -> TraversalOutput:
    """Random latent-node traversal for ablation."""

    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    num_nodes = graph_node_latents.shape[0]
    random_nodes = torch.randint(
        low=0,
        high=num_nodes,
        size=(steps,),
        device=graph_node_latents.device,
    )

    latents = [z_attr_start]

    current = z_attr_start

    for node in random_nodes:
        target = graph_node_latents[int(node.item())]
        latents.append(target)
        current = target

    del current

    return TraversalOutput(
        z_attr_path=torch.stack(latents, dim=0),
        node_path=[int(x.item()) for x in random_nodes],
    )


def build_traversal(
    z_attr_start: torch.Tensor,
    z_attr_target: torch.Tensor,
    graph_node_latents: torch.Tensor | None = None,
    graph_distance: torch.Tensor | None = None,
    source_node: int | None = None,
    target_node: int | None = None,
    mode: Literal["linear", "graph", "random"] = "graph",
    steps: int = 8,
) -> TraversalOutput:
    """Build a latent traversal path."""

    if mode == "linear":
        return TraversalOutput(
            z_attr_path=interpolate_linear(z_attr_start, z_attr_target, steps),
            node_path=[],
        )

    if mode == "graph":
        if graph_node_latents is None or graph_distance is None:
            raise ValueError("graph_node_latents and graph_distance are required for graph traversal")
        if source_node is None or target_node is None:
            raise ValueError("source_node and target_node are required for graph traversal")

        return graph_consistent_traversal(
            z_attr_start=z_attr_start,
            graph_node_latents=graph_node_latents,
            graph_distance=graph_distance,
            source_node=source_node,
            target_node=target_node,
            steps_per_edge=max(1, steps),
        )

    if mode == "random":
        if graph_node_latents is None:
            raise ValueError("graph_node_latents is required for random traversal")

        return random_path_traversal(
            z_attr_start=z_attr_start,
            graph_node_latents=graph_node_latents,
            steps=steps,
        )

    raise ValueError(f"Unsupported traversal mode: {mode}")


def geodesic_consistency(
    z_path: torch.Tensor,
    node_path: list[int],
    graph_distance: torch.Tensor,
) -> torch.Tensor:
    """Compute a GC-style diagnostic for one traversal path.

    Lower is better.
    """

    if z_path.ndim != 2:
        raise ValueError(f"z_path must have shape [T, Z], got {tuple(z_path.shape)}")

    if len(node_path) < 2 or z_path.shape[0] < 2:
        return z_path.new_zeros(())

    latent_steps = torch.norm(z_path[1:] - z_path[:-1], dim=-1)
    latent_progress = latent_steps / latent_steps.sum().clamp_min(1e-8)

    graph_steps = []
    for a, b in zip(node_path[:-1], node_path[1:]):
        graph_steps.append(graph_distance[a, b])

    graph_steps_tensor = torch.stack(graph_steps).to(device=z_path.device, dtype=z_path.dtype)
    graph_progress = graph_steps_tensor / graph_steps_tensor.sum().clamp_min(1e-8)

    min_len = min(latent_progress.shape[0], graph_progress.shape[0])

    return torch.abs(latent_progress[:min_len] - graph_progress[:min_len]).mean()