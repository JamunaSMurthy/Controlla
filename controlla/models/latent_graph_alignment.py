"""Latent graph alignment for Controlla.

This module implements the paper-level graph prior alignment:

- FGW-style alignment between z_attr and the emotion graph G_e.
- GW-style alignment between z_id and the identity graph G_i.

This is separate from the lightweight modality-graph OT alignment used inside
the input fusion stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LatentGraphAlignmentOutput:
    """Outputs from latent graph alignment.

    Attributes:
        loss: Total alignment loss.
        attribute_loss: FGW-style attribute graph loss.
        identity_loss: GW-style identity graph loss.
        attribute_transport: Attribute transport matrix [B, M].
        identity_transport: Identity transport matrix [B, K].
    """

    loss: torch.Tensor
    attribute_loss: torch.Tensor
    identity_loss: torch.Tensor
    attribute_transport: torch.Tensor
    identity_transport: torch.Tensor


class LatentGraphAlignment(nn.Module):
    """Align z_attr and z_id with external graph priors."""

    def __init__(
        self,
        z_dim: int,
        attr_graph_feature_dim: int,
        id_graph_feature_dim: int,
        mode: Literal["gw", "fgw"] = "fgw",
        regularization: float = 0.05,
        sinkhorn_iterations: int = 50,
        feature_alpha: float = 0.5,
        attr_weight: float = 1.0,
        id_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        if attr_graph_feature_dim <= 0:
            raise ValueError(
                f"attr_graph_feature_dim must be positive, got {attr_graph_feature_dim}"
            )
        if id_graph_feature_dim <= 0:
            raise ValueError(
                f"id_graph_feature_dim must be positive, got {id_graph_feature_dim}"
            )
        if mode not in {"gw", "fgw"}:
            raise ValueError(f"mode must be 'gw' or 'fgw', got {mode}")
        if regularization <= 0:
            raise ValueError(f"regularization must be positive, got {regularization}")
        if sinkhorn_iterations <= 0:
            raise ValueError(
                f"sinkhorn_iterations must be positive, got {sinkhorn_iterations}"
            )
        if not 0.0 <= feature_alpha <= 1.0:
            raise ValueError(f"feature_alpha must be in [0, 1], got {feature_alpha}")

        self.z_dim = z_dim
        self.mode = mode
        self.regularization = float(regularization)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.feature_alpha = float(feature_alpha)
        self.attr_weight = float(attr_weight)
        self.id_weight = float(id_weight)

        self.attr_node_projector = nn.Sequential(
            nn.LayerNorm(attr_graph_feature_dim),
            nn.Linear(attr_graph_feature_dim, z_dim),
            nn.GELU(),
            nn.Linear(z_dim, z_dim),
            nn.LayerNorm(z_dim),
        )

        self.id_node_projector = nn.Sequential(
            nn.LayerNorm(id_graph_feature_dim),
            nn.Linear(id_graph_feature_dim, z_dim),
            nn.GELU(),
            nn.Linear(z_dim, z_dim),
            nn.LayerNorm(z_dim),
        )

    def _sinkhorn(
        self,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        kernel = torch.exp(-cost / self.regularization).clamp_min(1e-8)

        u = torch.ones_like(source_mass)
        v = torch.ones_like(target_mass)

        for _ in range(self.sinkhorn_iterations):
            u = source_mass / (kernel @ v + 1e-8)
            v = target_mass / (kernel.transpose(0, 1) @ u + 1e-8)

        return (u.unsqueeze(1) * kernel) * v.unsqueeze(0)

    @staticmethod
    def _gw_cost(
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

    def _align_batch_to_graph(
        self,
        z: torch.Tensor,
        graph_node_features: torch.Tensor,
        graph_distance: torch.Tensor,
        projector: nn.Module,
        use_feature_term: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align batch latents to graph nodes.

        z: [B, Z]
        graph_node_features: [M, F]
        graph_distance: [M, M]
        """

        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, Z], got {tuple(z.shape)}")

        if z.shape[-1] != self.z_dim:
            raise ValueError(f"Expected z dim {self.z_dim}, got {z.shape[-1]}")

        if graph_node_features.ndim != 2:
            raise ValueError(
                "graph_node_features must have shape [M, F], got "
                f"{tuple(graph_node_features.shape)}"
            )

        if graph_distance.ndim != 2 or graph_distance.shape[0] != graph_distance.shape[1]:
            raise ValueError(
                f"graph_distance must have shape [M, M], got {tuple(graph_distance.shape)}"
            )

        batch_size = z.shape[0]
        num_nodes = graph_node_features.shape[0]

        if graph_distance.shape[0] != num_nodes:
            raise ValueError(
                f"graph_distance size {graph_distance.shape[0]} does not match "
                f"num graph nodes {num_nodes}"
            )

        z_norm = F.normalize(z, dim=-1, eps=1e-6)
        graph_nodes = F.normalize(projector(graph_node_features), dim=-1, eps=1e-6)

        source_mass = torch.full(
            (batch_size,),
            1.0 / max(batch_size, 1),
            device=z.device,
            dtype=z.dtype,
        )

        target_mass = torch.full(
            (num_nodes,),
            1.0 / max(num_nodes, 1),
            device=z.device,
            dtype=z.dtype,
        )

        latent_distance = torch.cdist(z_norm, z_norm, p=2)
        graph_distance = graph_distance.to(device=z.device, dtype=z.dtype)

        feature_cost = torch.cdist(z_norm, graph_nodes, p=2).pow(2)

        transport = torch.outer(source_mass, target_mass)

        structural_cost = self._gw_cost(
            source_structure=latent_distance,
            target_structure=graph_distance,
            transport=transport,
            source_mass=source_mass,
            target_mass=target_mass,
        )

        if use_feature_term:
            cost = self.feature_alpha * feature_cost + (1.0 - self.feature_alpha) * structural_cost
        else:
            cost = structural_cost

        transport = self._sinkhorn(source_mass, target_mass, cost)

        final_structural_cost = self._gw_cost(
            source_structure=latent_distance,
            target_structure=graph_distance,
            transport=transport,
            source_mass=source_mass,
            target_mass=target_mass,
        )

        if use_feature_term:
            final_cost = (
                self.feature_alpha * feature_cost
                + (1.0 - self.feature_alpha) * final_structural_cost
            )
        else:
            final_cost = final_structural_cost

        loss = (final_cost * transport).sum()

        return loss, transport

    def forward(
        self,
        z_attr: torch.Tensor,
        z_id: torch.Tensor,
        emotion_node_features: torch.Tensor,
        emotion_distance: torch.Tensor,
        identity_node_features: torch.Tensor,
        identity_distance: torch.Tensor,
    ) -> LatentGraphAlignmentOutput:
        """Compute attribute FGW and identity GW alignment."""

        attr_loss, attr_transport = self._align_batch_to_graph(
            z=z_attr,
            graph_node_features=emotion_node_features,
            graph_distance=emotion_distance,
            projector=self.attr_node_projector,
            use_feature_term=(self.mode == "fgw"),
        )

        id_loss, id_transport = self._align_batch_to_graph(
            z=z_id,
            graph_node_features=identity_node_features,
            graph_distance=identity_distance,
            projector=self.id_node_projector,
            use_feature_term=False,
        )

        total = self.attr_weight * attr_loss + self.id_weight * id_loss

        return LatentGraphAlignmentOutput(
            loss=total,
            attribute_loss=attr_loss,
            identity_loss=id_loss,
            attribute_transport=attr_transport,
            identity_transport=id_transport,
        )