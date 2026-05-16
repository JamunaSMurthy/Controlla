"""Identity encoder for reference-image preservation."""

from __future__ import annotations

import torch
from torch import nn

from .image_encoder import ImageEncoder


class IdentityEncoder(nn.Module):
    """Reference-image identity encoder.

    Input image shape: `[B, 3, H, W]`
    Output shape: `[B, D]`
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)
        self.projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, reference_images: torch.Tensor, has_reference: torch.Tensor) -> torch.Tensor:
        embedding = self.projection(self.image_encoder(reference_images))
        return embedding * has_reference.unsqueeze(-1)