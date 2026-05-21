"""Emotion consistency loss for Controlla.

Aligns the learned attribute factor z_attr with the target emotion representation.
This is an auxiliary endpoint/semantic consistency loss, not the graph-geometry loss.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class EmotionConsistencyLoss(nn.Module):
    """Align attribute latents with target emotion embeddings.

    Args:
        input_dim: Dimension of z_attr.
        target_dim: Dimension of target_emotion.
        loss_type: "mse", "cosine", or "smooth_l1".

    Inputs:
        z_attr: Tensor with shape [B, input_dim].
        target_emotion: Tensor with shape [B, target_dim].
        mask: Optional tensor with shape [B], where 1 means valid sample.

    Output:
        Scalar loss.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        loss_type: Literal["mse", "cosine", "smooth_l1"] = "cosine",
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if target_dim <= 0:
            raise ValueError(f"target_dim must be positive, got {target_dim}")
        if loss_type not in {"mse", "cosine", "smooth_l1"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.loss_type = loss_type

        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
            nn.LayerNorm(target_dim),
        )

    def forward(
        self,
        z_attr: torch.Tensor,
        target_emotion: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_attr.ndim != 2:
            raise ValueError(f"z_attr must have shape [B, D], got {tuple(z_attr.shape)}")

        if target_emotion.ndim != 2:
            raise ValueError(
                f"target_emotion must have shape [B, D], got {tuple(target_emotion.shape)}"
            )

        if z_attr.shape[0] != target_emotion.shape[0]:
            raise ValueError(
                f"Batch size mismatch: z_attr has B={z_attr.shape[0]}, "
                f"target_emotion has B={target_emotion.shape[0]}"
            )

        if z_attr.shape[1] != self.input_dim:
            raise ValueError(f"Expected z_attr dim {self.input_dim}, got {z_attr.shape[1]}")

        if target_emotion.shape[1] != self.target_dim:
            raise ValueError(
                f"Expected target_emotion dim {self.target_dim}, got {target_emotion.shape[1]}"
            )

        projected = self.projection(z_attr)

        if self.loss_type == "mse":
            per_sample_loss = F.mse_loss(projected, target_emotion, reduction="none").mean(dim=-1)

        elif self.loss_type == "smooth_l1":
            per_sample_loss = F.smooth_l1_loss(
                projected,
                target_emotion,
                reduction="none",
            ).mean(dim=-1)

        else:
            projected_norm = F.normalize(projected, dim=-1, eps=1e-6)
            target_norm = F.normalize(target_emotion, dim=-1, eps=1e-6)
            per_sample_loss = 1.0 - (projected_norm * target_norm).sum(dim=-1)

        if mask is not None:
            if mask.ndim != 1:
                raise ValueError(f"mask must have shape [B], got {tuple(mask.shape)}")
            if mask.shape[0] != z_attr.shape[0]:
                raise ValueError(
                    f"mask batch size {mask.shape[0]} does not match z_attr batch size {z_attr.shape[0]}"
                )

            mask = mask.to(device=z_attr.device, dtype=per_sample_loss.dtype)
            return (per_sample_loss * mask).sum() / mask.sum().clamp_min(1.0)

        return per_sample_loss.mean()