"""Emotion consistency loss for Controlla."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EmotionConsistencyLoss(nn.Module):
    """Align attribute latents with the target emotion embedding."""

    def __init__(self, input_dim: int, target_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, target_dim)

    def forward(self, z_attr: torch.Tensor, target_emotion: torch.Tensor) -> torch.Tensor:
        projected = self.projection(z_attr)
        return F.mse_loss(projected, target_emotion)