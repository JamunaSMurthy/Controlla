"""Graph consistency loss wrapper for Controlla."""

from __future__ import annotations

import torch
from torch import nn


class GraphConsistencyLoss(nn.Module):
    """Return the OT alignment penalty as the graph consistency loss."""

    def forward(self, ot_loss: torch.Tensor) -> torch.Tensor:
        return ot_loss