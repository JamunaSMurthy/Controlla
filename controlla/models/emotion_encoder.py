"""Emotion encoder combining explicit emotion labels and optional audio cues.

This module maps the normalized target emotion label and optional audio-derived
affective embedding into the shared Controlla hidden space.

It is best understood as an attribute-control encoder:
- emotion_labels provide the discrete target affect class.
- audio_embedding provides optional prosodic / affective evidence.
"""

from __future__ import annotations

import torch
from torch import nn


class EmotionEncoder(nn.Module):
    """Fuse emotion labels and optional audio embeddings into an attribute representation.

    Args:
        num_emotions: Number of normalized emotion classes.
        hidden_dim: Shared Controlla hidden dimension.
        dropout: Dropout used in the output projection.

    Inputs:
        emotion_labels: Tensor with shape [B], dtype long.
        audio_embedding: Tensor with shape [B, D].
        has_audio: Tensor with shape [B], bool or float mask.

    Output:
        Tensor with shape [B, D].
    """

    def __init__(
        self,
        num_emotions: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_emotions <= 0:
            raise ValueError(f"num_emotions must be positive, got {num_emotions}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim

        self.label_embedding = nn.Embedding(num_emotions, hidden_dim)

        self.audio_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learned gate controls how strongly audio modifies the label embedding.
        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        emotion_labels: torch.Tensor,
        audio_embedding: torch.Tensor,
        has_audio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode emotion labels with optional audio cues."""

        if emotion_labels.ndim != 1:
            raise ValueError(
                f"emotion_labels must have shape [B], got {tuple(emotion_labels.shape)}"
            )

        if audio_embedding.ndim != 2:
            raise ValueError(
                f"audio_embedding must have shape [B, D], got {tuple(audio_embedding.shape)}"
            )

        if audio_embedding.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected audio_embedding dim {self.hidden_dim}, got {audio_embedding.shape[-1]}"
            )

        if emotion_labels.shape[0] != audio_embedding.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"emotion_labels has B={emotion_labels.shape[0]}, "
                f"audio_embedding has B={audio_embedding.shape[0]}"
            )

        emotion_labels = emotion_labels.to(device=audio_embedding.device, dtype=torch.long)

        label_embedding = self.label_embedding(emotion_labels)
        audio_term = self.audio_projection(audio_embedding)

        if has_audio is not None:
            if has_audio.ndim != 1:
                raise ValueError(
                    f"has_audio must have shape [B], got {tuple(has_audio.shape)}"
                )

            if has_audio.shape[0] != audio_embedding.shape[0]:
                raise ValueError(
                    "Batch size mismatch: "
                    f"audio_embedding has B={audio_embedding.shape[0]}, "
                    f"has_audio has B={has_audio.shape[0]}"
                )

            audio_mask = has_audio.to(
                device=audio_embedding.device,
                dtype=audio_embedding.dtype,
            ).unsqueeze(-1)
            audio_term = audio_term * audio_mask

        gate = self.audio_gate(torch.cat([label_embedding, audio_term], dim=-1))
        fused = label_embedding + gate * audio_term

        return self.output(fused)