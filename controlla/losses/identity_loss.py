"""Identity preservation loss for Controlla."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IdentityPreservationLoss(nn.Module):
    """Encourage generated images to match the reference identity embedding.

    Inputs:
        generated_identity: [B, D]
        target_identity: [B, D]
        has_reference: [B], where 1 means reference identity is available

    Output:
        Scalar identity-preservation loss.
    """

    def __init__(self, loss_type: str = "cosine") -> None:
        super().__init__()

        if loss_type not in {"cosine", "mse"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.loss_type = loss_type

    def forward(
        self,
        generated_identity: torch.Tensor,
        target_identity: torch.Tensor,
        has_reference: torch.Tensor,
    ) -> torch.Tensor:
        if generated_identity.ndim != 2:
            raise ValueError(
                f"generated_identity must have shape [B, D], got {tuple(generated_identity.shape)}"
            )

        if target_identity.ndim != 2:
            raise ValueError(
                f"target_identity must have shape [B, D], got {tuple(target_identity.shape)}"
            )

        if generated_identity.shape != target_identity.shape:
            raise ValueError(
                "generated_identity and target_identity must have the same shape, "
                f"got {tuple(generated_identity.shape)} and {tuple(target_identity.shape)}"
            )

        if has_reference.ndim != 1:
            raise ValueError(
                f"has_reference must have shape [B], got {tuple(has_reference.shape)}"
            )

        if has_reference.shape[0] != generated_identity.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"generated_identity has B={generated_identity.shape[0]}, "
                f"has_reference has B={has_reference.shape[0]}"
            )

        mask = has_reference.to(
            device=generated_identity.device,
            dtype=generated_identity.dtype,
        )

        if self.loss_type == "cosine":
            generated_norm = F.normalize(generated_identity, dim=-1, eps=1e-6)
            target_norm = F.normalize(target_identity, dim=-1, eps=1e-6)
            per_sample_loss = 1.0 - (generated_norm * target_norm).sum(dim=-1)

        else:
            per_sample_loss = F.mse_loss(
                generated_identity,
                target_identity,
                reduction="none",
            ).mean(dim=-1)

        return (per_sample_loss * mask).sum() / mask.sum().clamp_min(1.0)