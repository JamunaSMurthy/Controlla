"""Optimal transport alignment for Controlla.

This module implements a lightweight entropic GW/FGW-style solver over the
input-level modality graph produced by GraphFusion. It is useful as a
modality-graph consistency regularizer.

For the full Controlla paper setting, dataset-level graph priors should also be
used:
- attribute/affective graph Ge aligned with z_attr
- identity graph Gi aligned with z_id
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class OTAlignmentOutput:
    """Outputs from the OT solver.

    Attributes:
        loss: Scalar tensor.
        transport: Tensor with shape [B, N, N].
        latent_similarity: Tensor with shape [B, N, N].
        graph_similarity: Tensor with shape [B, N, N].
    """

    loss: torch.Tensor
    transport: torch.Tensor
    latent_similarity: torch.Tensor
    graph_similarity: torch.Tensor


class OTAlignment(nn.Module):
    """Entropic GW/FGW alignment between modality graph and latent-node graph."""

    def __init__(
        self,
        hidden_dim: int,
        enabled: bool = True,
        mode: Literal["gw", "fgw"] = "fgw",
        regularization: float = 0.05,
        iterations: int = 20,
        sinkhorn_iterations: int = 50,
        feature_alpha: float = 0.5,
        max_nodes: int = 5,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if mode not in {"gw", "fgw"}:
            raise ValueError(f"mode must be 'gw' or 'fgw', got {mode}")
        if regularization <= 0:
            raise ValueError(f"regularization must be positive, got {regularization}")
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        if sinkhorn_iterations <= 0:
            raise ValueError(
                f"sinkhorn_iterations must be positive, got {sinkhorn_iterations}"
            )
        if not 0.0 <= feature_alpha <= 1.0:
            raise ValueError(f"feature_alpha must be in [0, 1], got {feature_alpha}")
        if max_nodes <= 0:
            raise ValueError(f"max_nodes must be positive, got {max_nodes}")

        self.hidden_dim = hidden_dim
        self.enabled = enabled
        self.mode = mode
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.feature_alpha = float(feature_alpha)
        self.max_nodes = int(max_nodes)

        self.node_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.graph_feature_encoder = nn.Sequential(
            nn.LayerNorm(max_nodes),
            nn.Linear(max_nodes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def _validate_inputs(
        self,
        node_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> None:
        if node_embeddings.ndim != 3:
            raise ValueError(
                f"node_embeddings must have shape [B, N, D], got {tuple(node_embeddings.shape)}"
            )
        if adjacency.ndim != 3:
            raise ValueError(
                f"adjacency must have shape [B, N, N], got {tuple(adjacency.shape)}"
            )
        if node_mask.ndim != 2:
            raise ValueError(
                f"node_mask must have shape [B, N], got {tuple(node_mask.shape)}"
            )

        batch_size, num_nodes, hidden_dim = node_embeddings.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected node embedding dim {self.hidden_dim}, got {hidden_dim}"
            )
        if num_nodes != self.max_nodes:
            raise ValueError(
                f"This lightweight OT module expects N=max_nodes={self.max_nodes}, got N={num_nodes}"
            )
        if adjacency.shape != (batch_size, num_nodes, num_nodes):
            raise ValueError(
                f"adjacency must have shape {(batch_size, num_nodes, num_nodes)}, "
                f"got {tuple(adjacency.shape)}"
            )
        if node_mask.shape != (batch_size, num_nodes):
            raise ValueError(
                f"node_mask must have shape {(batch_size, num_nodes)}, got {tuple(node_mask.shape)}"
            )

    def _sinkhorn(
        self,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
        cost: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_cost = cost.masked_fill(support_mask.eq(0.0), 1e4)

        kernel = torch.exp(-masked_cost / self.regularization) * support_mask
        kernel = kernel + 1e-8 * support_mask

        u = torch.ones_like(source_mass)
        v = torch.ones_like(target_mass)

        for _ in range(self.sinkhorn_iterations):
            u = source_mass / (kernel @ v + 1e-8)
            v = target_mass / (kernel.transpose(0, 1) @ u + 1e-8)

        transport = (u.unsqueeze(1) * kernel) * v.unsqueeze(0)
        return transport * support_mask

    @staticmethod
    def _gw_structural_cost(
        source_structure: torch.Tensor,
        target_structure: torch.Tensor,
        transport: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
    ) -> torch.Tensor:
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
        """Compute a modality graph-latent consistency penalty.

        Args:
            node_embeddings: Tensor with shape [B, N, D].
            adjacency: Tensor with shape [B, N, N].
            node_mask: Tensor with shape [B, N].

        Returns:
            OTAlignmentOutput.
        """

        self._validate_inputs(node_embeddings, adjacency, node_mask)

        node_mask = node_mask.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
        adjacency = adjacency.to(device=node_embeddings.device, dtype=node_embeddings.dtype)

        pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        graph_similarity = adjacency * pair_mask

        projected_nodes = F.normalize(
            self.node_projector(node_embeddings),
            dim=-1,
            eps=1e-6,
        )
        projected_nodes = projected_nodes * node_mask.unsqueeze(-1)

        latent_similarity = torch.matmul(
            projected_nodes,
            projected_nodes.transpose(1, 2),
        ) * pair_mask

        if not self.enabled:
            zero = adjacency.new_zeros(())
            transport = adjacency.new_zeros(adjacency.shape)
            return OTAlignmentOutput(
                loss=zero,
                transport=transport,
                latent_similarity=latent_similarity,
                graph_similarity=graph_similarity,
            )

        transport_matrices: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []

        batch_size = node_embeddings.shape[0]

        for sample_index in range(batch_size):
            sample_mask = node_mask[sample_index]

            active_mass = sample_mask.sum()
            if active_mass <= 0:
                transport = torch.zeros_like(adjacency[sample_index])
                sample_loss = adjacency.new_zeros(())
                transport_matrices.append(transport)
                losses.append(sample_loss)
                continue

            pair_support = torch.outer(sample_mask, sample_mask)
            source_mass = sample_mask / active_mass.clamp_min(1.0)
            target_mass = source_mass

            graph_rows = graph_similarity[sample_index]
            latent_nodes = projected_nodes[sample_index]

            graph_features = F.normalize(
                self.graph_feature_encoder(graph_rows),
                dim=-1,
                eps=1e-6,
            )

            source_structure = torch.cdist(graph_rows, graph_rows, p=2)
            target_structure = torch.cdist(latent_nodes, latent_nodes, p=2)

            feature_cost = torch.cdist(graph_features, latent_nodes, p=2).pow(2)

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
                    fused_cost = (
                        self.feature_alpha * feature_cost
                        + (1.0 - self.feature_alpha) * structural_cost
                    )

                transport = self._sinkhorn(
                    source_mass=source_mass,
                    target_mass=target_mass,
                    cost=fused_cost,
                    support_mask=pair_support,
                )

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
                final_cost = (
                    self.feature_alpha * feature_cost
                    + (1.0 - self.feature_alpha) * final_structural_cost
                )

            sample_loss = (final_cost * transport).sum()

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