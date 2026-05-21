"""Graph-based multimodal fusion for Controlla.

This module represents each modality as a typed node and performs lightweight
structure-aware fusion with transformer message passing. It produces:
- fused multimodal embedding
- identity factor z_id
- attribute factor z_attr
- node embeddings and an adjacency proxy for graph/OT diagnostics

This is the input-level modality fusion graph. Dataset-level emotion/identity
graph priors are handled by latent graph alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .factorization_heads import FactorizationHeads


NODE_NAMES = ("text", "identity", "emotion", "image", "audio")
TEXT_IDX = NODE_NAMES.index("text")
IDENTITY_IDX = NODE_NAMES.index("identity")
EMOTION_IDX = NODE_NAMES.index("emotion")
IMAGE_IDX = NODE_NAMES.index("image")
AUDIO_IDX = NODE_NAMES.index("audio")


@dataclass
class GraphFusionOutput:
    """Outputs from graph fusion.

    Attributes:
        fused_embedding: Tensor with shape [B, D].
        z_id: Tensor with shape [B, Z].
        z_attr: Tensor with shape [B, Z].
        node_embeddings: Tensor with shape [B, N, D].
        adjacency: Tensor with shape [B, N, N].
        node_mask: Tensor with shape [B, N].
    """

    fused_embedding: torch.Tensor
    z_id: torch.Tensor
    z_attr: torch.Tensor
    node_embeddings: torch.Tensor
    adjacency: torch.Tensor
    node_mask: torch.Tensor


class GraphFusion(nn.Module):
    """Fuse typed modality embeddings into factorized controllable latents."""

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        num_layers: int,
        dropout: float,
        num_heads: int = 4,
        nonnegative_adjacency: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_nodes = len(NODE_NAMES)
        self.nonnegative_adjacency = nonnegative_adjacency

        self.type_embedding = nn.Embedding(self.num_nodes, hidden_dim)

        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder: nn.Module = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        else:
            self.encoder = nn.Identity()

        self.fused_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.factorization_heads = FactorizationHeads(
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            dropout=dropout,
        )

    def _validate_inputs(
        self,
        embeddings: list[torch.Tensor],
        node_mask: torch.Tensor,
    ) -> None:
        if len(embeddings) != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} modality embeddings, got {len(embeddings)}"
            )

        batch_size = embeddings[0].shape[0]

        for idx, embedding in enumerate(embeddings):
            if embedding.ndim != 2:
                raise ValueError(
                    f"{NODE_NAMES[idx]} embedding must have shape [B, D], "
                    f"got {tuple(embedding.shape)}"
                )
            if embedding.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch for {NODE_NAMES[idx]} embedding: "
                    f"expected B={batch_size}, got B={embedding.shape[0]}"
                )
            if embedding.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"{NODE_NAMES[idx]} embedding dim must be {self.hidden_dim}, "
                    f"got {embedding.shape[1]}"
                )

        if node_mask.ndim != 2:
            raise ValueError(f"node_mask must have shape [B, N], got {tuple(node_mask.shape)}")
        if node_mask.shape != (batch_size, self.num_nodes):
            raise ValueError(
                f"node_mask must have shape {(batch_size, self.num_nodes)}, "
                f"got {tuple(node_mask.shape)}"
            )

    def forward(
        self,
        text_embedding: torch.Tensor,
        identity_embedding: torch.Tensor,
        emotion_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
        node_mask: torch.Tensor,
        no_graph: bool = False,
    ) -> GraphFusionOutput:
        """Build and fuse a typed modality graph."""

        embeddings = [
            text_embedding,
            identity_embedding,
            emotion_embedding,
            image_embedding,
            audio_embedding,
        ]

        self._validate_inputs(embeddings, node_mask)

        node_mask = node_mask.to(device=text_embedding.device, dtype=text_embedding.dtype)

        nodes = torch.stack(embeddings, dim=1)
        nodes = nodes * node_mask.unsqueeze(-1)

        type_ids = torch.arange(self.num_nodes, device=nodes.device)
        type_emb = self.type_embedding(type_ids).unsqueeze(0)
        nodes = nodes + type_emb * node_mask.unsqueeze(-1)

        if no_graph or isinstance(self.encoder, nn.Identity):
            encoded_nodes = nodes
        else:
            encoded_nodes = self.encoder(
                nodes,
                src_key_padding_mask=node_mask.eq(0.0),
            )

        encoded_nodes = encoded_nodes * node_mask.unsqueeze(-1)

        normalized_nodes = F.normalize(encoded_nodes, dim=-1, eps=1e-6)
        adjacency = torch.matmul(normalized_nodes, normalized_nodes.transpose(1, 2))

        if self.nonnegative_adjacency:
            adjacency = (adjacency + 1.0) * 0.5

        pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        adjacency = adjacency * pair_mask

        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = encoded_nodes.sum(dim=1) / denom
        fused_embedding = self.fused_projection(pooled)

        identity_node = encoded_nodes[:, IDENTITY_IDX, :]
        emotion_node = encoded_nodes[:, EMOTION_IDX, :]

        identity_available = node_mask[:, IDENTITY_IDX].unsqueeze(-1)
        emotion_available = node_mask[:, EMOTION_IDX].unsqueeze(-1)

        identity_context = (
            identity_available * identity_node
            + (1.0 - identity_available) * fused_embedding
        )

        attribute_context = (
            emotion_available * emotion_node
            + (1.0 - emotion_available) * fused_embedding
        )

        factors = self.factorization_heads(
            fused_embedding=fused_embedding,
            identity_context=identity_context,
            attribute_context=attribute_context,
        )

        return GraphFusionOutput(
            fused_embedding=fused_embedding,
            z_id=factors.z_id,
            z_attr=factors.z_attr,
            node_embeddings=encoded_nodes,
            adjacency=adjacency,
            node_mask=node_mask,
        )