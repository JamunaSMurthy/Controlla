"""Contrastive alignment loss for text and fused multimodal latents."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss over paired embeddings."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = F.normalize(source, dim=-1)
        target = F.normalize(target, dim=-1)
        logits = torch.matmul(source, target.transpose(0, 1)) / self.temperature
        labels = torch.arange(source.shape[0], device=source.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_a + loss_b)