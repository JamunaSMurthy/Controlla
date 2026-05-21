"""Contrastive alignment loss for paired multimodal embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss over paired embeddings.

    Assumes source[i] and target[i] are positive pairs.
    All other samples in the batch are treated as negatives.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.temperature = float(temperature)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.ndim != 2:
            raise ValueError(f"source must have shape [B, D], got {tuple(source.shape)}")

        if target.ndim != 2:
            raise ValueError(f"target must have shape [B, D], got {tuple(target.shape)}")

        if source.shape != target.shape:
            raise ValueError(
                f"source and target must have the same shape, got "
                f"{tuple(source.shape)} and {tuple(target.shape)}"
            )

        batch_size = source.shape[0]

        if batch_size <= 1:
            raise ValueError("ContrastiveLoss requires batch_size > 1")

        source = F.normalize(source, dim=-1, eps=1e-6)
        target = F.normalize(target, dim=-1, eps=1e-6)

        logits = torch.matmul(source, target.transpose(0, 1))
        logits = logits / self.temperature

        labels = torch.arange(batch_size, device=source.device)

        loss_source_to_target = F.cross_entropy(logits, labels)
        loss_target_to_source = F.cross_entropy(logits.transpose(0, 1), labels)

        return 0.5 * (loss_source_to_target + loss_target_to_source)