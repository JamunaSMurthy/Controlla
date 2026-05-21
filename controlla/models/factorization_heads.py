"""Factorization heads for Controlla.

This module implements the learned identity and attribute factor heads:

    z_id   = h_id(z)
    z_attr = h_attr(z)

The heads are separated from GraphFusion so the implementation matches the
paper-level Controlla design more cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FactorizationOutput:
    """Outputs from factorization heads.

    Attributes:
        z_id: Identity factor with shape [B, Z].
        z_attr: Attribute factor with shape [B, Z].
    """

    z_id: torch.Tensor
    z_attr: torch.Tensor


class FactorizationHeads(nn.Module):
    """Learn identity and attribute factors from fused/context embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        input_dim = hidden_dim * 2

        self.z_id_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )

        self.z_attr_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(
        self,
        fused_embedding: torch.Tensor,
        identity_context: torch.Tensor,
        attribute_context: torch.Tensor,
    ) -> FactorizationOutput:
        if fused_embedding.ndim != 2:
            raise ValueError(
                f"fused_embedding must have shape [B, D], got {tuple(fused_embedding.shape)}"
            )

        if identity_context.ndim != 2:
            raise ValueError(
                f"identity_context must have shape [B, D], got {tuple(identity_context.shape)}"
            )

        if attribute_context.ndim != 2:
            raise ValueError(
                f"attribute_context must have shape [B, D], got {tuple(attribute_context.shape)}"
            )

        if fused_embedding.shape != identity_context.shape:
            raise ValueError(
                "fused_embedding and identity_context must have the same shape, got "
                f"{tuple(fused_embedding.shape)} and {tuple(identity_context.shape)}"
            )

        if fused_embedding.shape != attribute_context.shape:
            raise ValueError(
                "fused_embedding and attribute_context must have the same shape, got "
                f"{tuple(fused_embedding.shape)} and {tuple(attribute_context.shape)}"
            )

        id_input = torch.cat([fused_embedding, identity_context], dim=-1)
        attr_input = torch.cat([fused_embedding, attribute_context], dim=-1)

        z_id = self.z_id_head(id_input)
        z_attr = self.z_attr_head(attr_input)

        return FactorizationOutput(z_id=z_id, z_attr=z_attr)