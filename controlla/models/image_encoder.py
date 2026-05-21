"""Image encoder used for target-image and reference-image features.

Backends:
- cnn: lightweight trainable CNN for smoke tests and fast local runs.
- clip: pretrained CLIP vision encoder for paper-aligned experiments.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class LightweightCNNImageEncoder(nn.Module):
    """Lightweight CNN image encoder.

    Input shape:
        images: [B, 3, H, W]

    Output shape:
        embedding: [B, D]
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")

        if images.shape[1] != 3:
            raise ValueError(f"images must have 3 channels, got {images.shape[1]}")

        features = self.encoder(images).flatten(start_dim=1)
        embedding = self.projection(features)
        return F.normalize(embedding, dim=-1, eps=1e-6)


class CLIPImageEncoder(nn.Module):
    """Pretrained CLIP image encoder with projection into Controlla hidden space.

    This backend expects images already normalized according to the CLIP
    preprocessing convention used in the training pipeline.
    """

    def __init__(
        self,
        hidden_dim: int,
        model_name: str = "openai/clip-vit-base-patch32",
        local_files_only: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        try:
            from transformers import CLIPVisionModel
        except ImportError as error:
            raise ImportError(
                "CLIP image encoder requested but transformers is not installed. "
                "Install with: pip install transformers"
            ) from error

        self.hidden_dim = hidden_dim
        self.vision_model = CLIPVisionModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        clip_dim = int(self.vision_model.config.hidden_size)

        if freeze:
            self.vision_model.requires_grad_(False)

        self.projection = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")

        if images.shape[1] != 3:
            raise ValueError(f"images must have 3 channels, got {images.shape[1]}")

        outputs = self.vision_model(pixel_values=images)
        pooled = outputs.pooler_output
        embedding = self.projection(pooled)
        return F.normalize(embedding, dim=-1, eps=1e-6)


class ImageEncoder(nn.Module):
    """Image encoder wrapper.

    Args:
        hidden_dim: Output embedding dimension.
        backend: "cnn" for lightweight tests, "clip" for paper-aligned experiments.
        model_name: CLIP model name/path when backend="clip".
        local_files_only: Whether to load CLIP only from local files.
        freeze: Whether to freeze the pretrained CLIP vision model.
    """

    def __init__(
        self,
        hidden_dim: int,
        backend: Literal["cnn", "clip"] = "cnn",
        model_name: str = "openai/clip-vit-base-patch32",
        local_files_only: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        self.backend_name = backend

        if backend == "cnn":
            self.encoder = LightweightCNNImageEncoder(hidden_dim=hidden_dim)
        elif backend == "clip":
            self.encoder = CLIPImageEncoder(
                hidden_dim=hidden_dim,
                model_name=model_name,
                local_files_only=local_files_only,
                freeze=freeze,
            )
        else:
            raise ValueError(f"Unsupported image encoder backend: {backend}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)