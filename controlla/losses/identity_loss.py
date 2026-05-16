"""Identity preservation loss for Controlla."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IdentityPreservationLoss(nn.Module):
    """Encourage generated images to match the reference identity embedding."""

    def forward(
        self,
        generated_identity: torch.Tensor,
        target_identity: torch.Tensor,
        has_reference: torch.Tensor,
    ) -> torch.Tensor:
        mask = has_reference.unsqueeze(-1)
        cosine = F.cosine_similarity(generated_identity * mask, target_identity * mask, dim=-1)
        return ((1.0 - cosine) * has_reference).sum() / has_reference.sum().clamp_min(1.0)