"""Audio feature encoder for precomputed emotion-aware descriptors."""

from __future__ import annotations

import torch
from torch import nn


class AudioEncoder(nn.Module):
    """MLP encoder for audio features.

    Input shape: `[B, A]`
    Output shape: `[B, D]`
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, audio_features: torch.Tensor, has_audio: torch.Tensor) -> torch.Tensor:
        embedding = self.network(audio_features)
        return embedding * has_audio.unsqueeze(-1)