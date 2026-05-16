"""Optimal transport alignment for Controlla.

This implementation upgrades the earlier placeholder to an entropically
regularized GW or FGW-style solver over a small fixed modality graph. The solver
is intentionally lightweight but is a real iterative OT routine rather than a
pointwise similarity proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class OTAlignmentOutput:
    """Outputs from the OT solver.

    Attributes:
        loss: Scalar tensor.
        transport: Tensor with shape `[B, N, N]`.
        latent_similarity: Tensor with shape `[B, N, N]`.
        graph_similarity: Tensor with shape `[B, N, N]`.
    """

    loss: torch.Tensor
    transport: torch.Tensor
    latent_similarity: torch.Tensor
    graph_similarity: torch.Tensor


class OTAlignment(nn.Module):
    """Entropic GW or FGW alignment between modality graphs and latent nodes."""

    def __init__(
        self,
        hidden_dim: int,
        enabled: bool = True,
        mode: str = "fgw",
        regularization: float = 0.05,
        iterations: int = 20,
        sinkhorn_iterations: int = 50,
        feature_alpha: float = 0.5,
        max_nodes: int = 5,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.mode = mode
        self.regularization = regularization
        self.iterations = iterations
        self.sinkhorn_iterations = sinkhorn_iterations
        self.feature_alpha = feature_alpha
        self.node_projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.graph_feature_encoder = nn.Sequential(
            nn.Linear(max_nodes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _sinkhorn(self, source_mass: torch.Tensor, target_mass: torch.Tensor, cost: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        masked_cost = cost.masked_fill(support_mask.eq(0.0), 1e4)
        kernel = torch.exp(-masked_cost / self.regularization) * support_mask + 1e-8
        u = torch.ones_like(source_mass)
        v = torch.ones_like(target_mass)
        for _ in range(self.sinkhorn_iterations):
            u = source_mass / (kernel @ v + 1e-8)
            v = target_mass / (kernel.transpose(0, 1) @ u + 1e-8)
        return (u.unsqueeze(1) * kernel) * v.unsqueeze(0) * support_mask

    def _gw_structural_cost(self, source_structure: torch.Tensor, target_structure: torch.Tensor, transport: torch.Tensor, source_mass: torch.Tensor, target_mass: torch.Tensor) -> torch.Tensor:
        const_source = (source_structure.pow(2) @ source_mass).unsqueeze(1)
        const_target = (target_structure.pow(2) @ target_mass).unsqueeze(0)
        coupling_term = source_structure @ transport @ target_structure.transpose(0, 1)
        return const_source + const_target - 2.0 * coupling_term

    def forward(
        self,
        node_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> OTAlignmentOutput:
        """Compute a graph-latent consistency penalty.

        Inputs:
            node_embeddings: `[B, N, D]`
            adjacency: `[B, N, N]`
            node_mask: `[B, N]`
        """
        if not self.enabled:
            zero = adjacency.new_zeros(())
            transport = adjacency.new_zeros(adjacency.shape)
            return OTAlignmentOutput(loss=zero, transport=transport, latent_similarity=transport, graph_similarity=adjacency)

        projected_nodes = F.normalize(self.node_projector(node_embeddings), dim=-1)
        pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        latent_similarity = torch.matmul(projected_nodes, projected_nodes.transpose(1, 2)) * pair_mask
        graph_similarity = adjacency * pair_mask

        transport_matrices = []
        losses = []
        for sample_index in range(node_embeddings.shape[0]):
            sample_mask = node_mask[sample_index]
            pair_support = torch.outer(sample_mask, sample_mask)
            source_mass = sample_mask / sample_mask.sum().clamp_min(1.0)
            target_mass = source_mass

            graph_rows = graph_similarity[sample_index]
            graph_features = F.normalize(self.graph_feature_encoder(graph_rows), dim=-1)
            source_structure = torch.cdist(graph_rows, graph_rows, p=2)
            target_structure = torch.cdist(projected_nodes[sample_index], projected_nodes[sample_index], p=2)
            feature_cost = torch.cdist(graph_features, projected_nodes[sample_index], p=2).pow(2)

            transport = torch.outer(source_mass, target_mass) * pair_support
            for _ in range(self.iterations):
                structural_cost = self._gw_structural_cost(
                    source_structure=source_structure,
                    target_structure=target_structure,
                    transport=transport,
                    source_mass=source_mass,
                    target_mass=target_mass,
                )
                if self.mode == "gw":
                    fused_cost = structural_cost
                else:
                    fused_cost = self.feature_alpha * feature_cost + (1.0 - self.feature_alpha) * structural_cost
                transport = self._sinkhorn(source_mass, target_mass, fused_cost, pair_support)

            final_structural_cost = self._gw_structural_cost(
                source_structure=source_structure,
                target_structure=target_structure,
                transport=transport,
                source_mass=source_mass,
                target_mass=target_mass,
            )
            if self.mode == "gw":
                final_cost = final_structural_cost
            else:
                final_cost = self.feature_alpha * feature_cost + (1.0 - self.feature_alpha) * final_structural_cost

            entropy = -(transport.clamp_min(1e-8) * transport.clamp_min(1e-8).log() * pair_support).sum()
            sample_loss = (final_cost * transport).sum() + self.regularization * entropy
            transport_matrices.append(transport)
            losses.append(sample_loss)

        transport = torch.stack(transport_matrices, dim=0)
        total_loss = torch.stack(losses).mean()

        return OTAlignmentOutput(
            loss=total_loss,
            transport=transport,
            latent_similarity=latent_similarity,
            graph_similarity=graph_similarity,
        )