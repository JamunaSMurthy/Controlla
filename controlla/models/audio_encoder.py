"""Audio feature encoder for precomputed emotion-aware descriptors.

This module expects audio features that have already been extracted from
a pretrained audio model, such as wav2vec2-style embeddings, prosody features,
or emotion-aware audio descriptors.

Input:
    audio_features: [B, A]
    has_audio: [B] boolean/float mask, where 1 means audio is available.

Output:
    audio_embedding: [B, D]
"""

from __future__ import annotations

import torch
from torch import nn


class AudioEncoder(nn.Module):
    """MLP adapter for precomputed audio features.

    This is not a raw waveform encoder. It maps precomputed audio descriptors
    into the shared Controlla hidden space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        has_audio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode precomputed audio features.

        Args:
            audio_features: Tensor with shape [B, input_dim].
            has_audio: Optional tensor with shape [B]. Values are interpreted as
                a mask: 1/True means audio exists, 0/False means missing audio.

        Returns:
            Tensor with shape [B, hidden_dim].
        """

        if audio_features.ndim != 2:
            raise ValueError(
                f"audio_features must have shape [B, A], got {tuple(audio_features.shape)}"
            )

        if audio_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected audio feature dim {self.input_dim}, got {audio_features.shape[-1]}"
            )

        embedding = self.network(audio_features)

        if has_audio is None:
            return embedding

        if has_audio.ndim != 1:
            raise ValueError(f"has_audio must have shape [B], got {tuple(has_audio.shape)}")

        if has_audio.shape[0] != audio_features.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"audio_features has B={audio_features.shape[0]}, "
                f"has_audio has B={has_audio.shape[0]}"
            )

        mask = has_audio.to(device=embedding.device, dtype=embedding.dtype).unsqueeze(-1)
        return embedding * mask