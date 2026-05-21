"""Graph consistency loss wrapper for Controlla.

This loss wraps the OT alignment penalty used to encourage consistency between
learned latent/node geometry and graph structure. The actual GW/FGW computation
is performed by the OTAlignment module.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphConsistencyLoss(nn.Module):
    """Weighted wrapper around the OT graph-alignment penalty."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()

        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")

        self.weight = float(weight)

    def forward(self, ot_loss: torch.Tensor | None) -> torch.Tensor:
        """Return weighted OT graph-consistency loss.

        Args:
            ot_loss: Scalar tensor returned by OTAlignment.

        Returns:
            Scalar graph-consistency loss.
        """

        if ot_loss is None:
            raise ValueError("ot_loss cannot be None")

        if ot_loss.ndim != 0:
            raise ValueError(f"ot_loss must be a scalar tensor, got shape {tuple(ot_loss.shape)}")

        return self.weight * ot_loss