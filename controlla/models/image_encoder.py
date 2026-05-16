"""Image encoder used for reference-image and target-image features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ImageEncoder(nn.Module):
    """Lightweight CNN image encoder.

    Input shape: `[B, 3, H, W]`
    Output shape: `[B, D]`
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images).flatten(start_dim=1)
        return F.normalize(self.projection(features), dim=-1)