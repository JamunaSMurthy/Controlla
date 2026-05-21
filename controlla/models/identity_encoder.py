"""Identity encoder for reference-image preservation.

This module maps the visual reference image x_ref into the shared Controlla
hidden space. If the wrapped ImageEncoder uses identity-aware face features
such as ArcFace, this acts as an identity encoder. If it uses generic visual
features, this acts as a reference-image identity adapter.
"""

from __future__ import annotations

import torch
from torch import nn

from .image_encoder import ImageEncoder


class IdentityEncoder(nn.Module):
    """Reference-image identity encoder / adapter.

    Inputs:
        reference_images: Tensor with shape [B, 3, H, W].
        has_reference: Tensor with shape [B], bool/float mask.

    Output:
        Tensor with shape [B, D].
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.hidden_dim = hidden_dim
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        reference_images: torch.Tensor,
        has_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode reference images into identity/reference embeddings."""

        if reference_images.ndim != 4:
            raise ValueError(
                "reference_images must have shape [B, 3, H, W], "
                f"got {tuple(reference_images.shape)}"
            )

        if reference_images.shape[1] != 3:
            raise ValueError(
                f"reference_images must have 3 channels, got {reference_images.shape[1]}"
            )

        embedding = self.image_encoder(reference_images)

        if embedding.ndim != 2:
            raise ValueError(
                f"ImageEncoder must return shape [B, D], got {tuple(embedding.shape)}"
            )

        if embedding.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected ImageEncoder dim {self.hidden_dim}, got {embedding.shape[-1]}"
            )

        embedding = self.projection(embedding)

        if has_reference is None:
            return embedding

        if has_reference.ndim != 1:
            raise ValueError(
                f"has_reference must have shape [B], got {tuple(has_reference.shape)}"
            )

        if has_reference.shape[0] != reference_images.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"reference_images has B={reference_images.shape[0]}, "
                f"has_reference has B={has_reference.shape[0]}"
            )

        mask = has_reference.to(device=embedding.device, dtype=embedding.dtype).unsqueeze(-1)
        return embedding * mask