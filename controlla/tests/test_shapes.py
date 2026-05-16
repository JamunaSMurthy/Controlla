"""Tensor shape tests for Controlla modules."""

from __future__ import annotations

import torch

from models.graph_fusion import GraphFusion
from models.ot_alignment import OTAlignment


def test_graph_fusion_shapes() -> None:
    batch_size = 2
    hidden_dim = 32
    z_dim = 16
    fusion = GraphFusion(hidden_dim=hidden_dim, z_dim=z_dim, num_layers=1, dropout=0.0)
    node_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
    output = fusion(
        text_embedding=torch.randn(batch_size, hidden_dim),
        identity_embedding=torch.randn(batch_size, hidden_dim),
        emotion_embedding=torch.randn(batch_size, hidden_dim),
        image_embedding=torch.randn(batch_size, hidden_dim),
        audio_embedding=torch.randn(batch_size, hidden_dim),
        node_mask=node_mask,
    )
    assert output.fused_embedding.shape == (batch_size, hidden_dim)
    assert output.z_id.shape == (batch_size, z_dim)
    assert output.z_attr.shape == (batch_size, z_dim)
    assert output.node_embeddings.shape == (batch_size, 5, hidden_dim)
    assert output.adjacency.shape == (batch_size, 5, 5)


def test_ot_alignment_shapes() -> None:
    ot_alignment = OTAlignment(hidden_dim=32)
    node_embeddings = torch.randn(2, 5, 32)
    adjacency = torch.randn(2, 5, 5)
    node_mask = torch.ones(2, 5)
    output = ot_alignment(node_embeddings, adjacency, node_mask)
    assert output.transport.shape == (2, 5, 5)
    assert output.latent_similarity.shape == (2, 5, 5)