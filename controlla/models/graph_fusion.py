"""Graph-based multimodal fusion for Controlla.

This module is inspired conceptually by multimodal fusion and structure-aware
conditioning patterns rather than a direct graph library dependency. The graph is
represented as typed modality nodes with learned message passing via transformer
blocks and a cosine-similarity adjacency proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


NODE_NAMES = ("text", "identity", "emotion", "image", "audio")


@dataclass
class GraphFusionOutput:
    """Outputs from graph fusion.

    Attributes:
        fused_embedding: `[B, D]`
        z_id: `[B, Z]`
        z_attr: `[B, Z]`
        node_embeddings: `[B, N, D]`
        adjacency: `[B, N, N]`
        node_mask: `[B, N]`
    """

    fused_embedding: torch.Tensor
    z_id: torch.Tensor
    z_attr: torch.Tensor
    node_embeddings: torch.Tensor
    adjacency: torch.Tensor
    node_mask: torch.Tensor


class GraphFusion(nn.Module):
    """Fuse modality embeddings into a shared controllable latent space."""

    def __init__(self, hidden_dim: int, z_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(len(NODE_NAMES), hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fused_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.z_id_head = nn.Linear(hidden_dim * 2, z_dim)
        self.z_attr_head = nn.Linear(hidden_dim * 2, z_dim)

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
        """Build and fuse a typed modality graph.

        All modality embeddings have shape `[B, D]`. The `node_mask` has shape `[B, 5]`
        and contains 1.0 for active nodes and 0.0 for inactive nodes.
        """
        nodes = torch.stack(
            [text_embedding, identity_embedding, emotion_embedding, image_embedding, audio_embedding],
            dim=1,
        )
        type_ids = torch.arange(len(NODE_NAMES), device=nodes.device)
        nodes = nodes + self.type_embedding(type_ids).unsqueeze(0)

        if no_graph:
            encoded_nodes = nodes
        else:
            encoded_nodes = self.encoder(nodes, src_key_padding_mask=node_mask.eq(0.0))

        normalized_nodes = F.normalize(encoded_nodes, dim=-1)
        adjacency = torch.matmul(normalized_nodes, normalized_nodes.transpose(1, 2))
        pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        adjacency = adjacency * pair_mask

        pooled = (encoded_nodes * node_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused_embedding = self.fused_projection(pooled)

        identity_node = encoded_nodes[:, NODE_NAMES.index("identity")]
        emotion_node = encoded_nodes[:, NODE_NAMES.index("emotion")]
        z_id = self.z_id_head(torch.cat([fused_embedding, identity_node], dim=-1))
        z_attr = self.z_attr_head(torch.cat([fused_embedding, emotion_node], dim=-1))
        return GraphFusionOutput(
            fused_embedding=fused_embedding,
            z_id=z_id,
            z_attr=z_attr,
            node_embeddings=encoded_nodes,
            adjacency=adjacency,
            node_mask=node_mask,
        )