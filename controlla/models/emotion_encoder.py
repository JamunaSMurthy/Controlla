"""Emotion encoder combining explicit labels and optional audio cues."""

from __future__ import annotations

import torch
from torch import nn


class EmotionEncoder(nn.Module):
    """Fuse label and audio embeddings into an attribute representation.

    Inputs:
        emotion_labels: `[B]`
        audio_embedding: `[B, D]`
        has_audio: `[B]`

    Output:
        Tensor with shape `[B, D]`
    """

    def __init__(self, num_emotions: int, hidden_dim: int) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(num_emotions, hidden_dim)
        self.audio_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(
        self,
        emotion_labels: torch.Tensor,
        audio_embedding: torch.Tensor,
        has_audio: torch.Tensor,
    ) -> torch.Tensor:
        label_embedding = self.label_embedding(emotion_labels)
        audio_term = self.audio_projection(audio_embedding) * has_audio.unsqueeze(-1)
        return self.output(label_embedding + audio_term)