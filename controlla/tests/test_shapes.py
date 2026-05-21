"""Tensor shape tests for Controlla modules."""

from __future__ import annotations

import torch

from controlla.models.factorization_heads import FactorizationHeads
from controlla.models.graph_fusion import GraphFusion
from controlla.models.latent_graph_alignment import LatentGraphAlignment
from controlla.models.ot_alignment import OTAlignment


def test_graph_fusion_shapes() -> None:
    batch_size = 2
    hidden_dim = 32
    z_dim = 16

    fusion = GraphFusion(
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        num_layers=1,
        dropout=0.0,
        num_heads=4,
    )

    node_mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
        dtype=torch.float32,
    )

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


def test_factorization_heads_shapes() -> None:
    heads = FactorizationHeads(hidden_dim=32, z_dim=16)

    output = heads(
        fused_embedding=torch.randn(3, 32),
        identity_context=torch.randn(3, 32),
        attribute_context=torch.randn(3, 32),
    )

    assert output.z_id.shape == (3, 16)
    assert output.z_attr.shape == (3, 16)


def test_ot_alignment_shapes() -> None:
    ot_alignment = OTAlignment(hidden_dim=32)

    node_embeddings = torch.randn(2, 5, 32)
    adjacency = torch.rand(2, 5, 5)
    node_mask = torch.ones(2, 5)

    output = ot_alignment(node_embeddings, adjacency, node_mask)

    assert output.loss.ndim == 0
    assert output.transport.shape == (2, 5, 5)
    assert output.latent_similarity.shape == (2, 5, 5)
    assert output.graph_similarity.shape == (2, 5, 5)


def test_latent_graph_alignment_shapes() -> None:
    alignment = LatentGraphAlignment(
        z_dim=16,
        attr_graph_feature_dim=8,
        id_graph_feature_dim=8,
    )

    z_attr = torch.randn(4, 16)
    z_id = torch.randn(4, 16)

    emotion_node_features = torch.randn(6, 8)
    identity_node_features = torch.randn(5, 8)

    emotion_distance = torch.rand(6, 6)
    emotion_distance = 0.5 * (emotion_distance + emotion_distance.T)
    emotion_distance.fill_diagonal_(0.0)

    identity_distance = torch.rand(5, 5)
    identity_distance = 0.5 * (identity_distance + identity_distance.T)
    identity_distance.fill_diagonal_(0.0)

    output = alignment(
        z_attr=z_attr,
        z_id=z_id,
        emotion_node_features=emotion_node_features,
        emotion_distance=emotion_distance,
        identity_node_features=identity_node_features,
        identity_distance=identity_distance,
    )

    assert output.loss.ndim == 0
    assert output.attribute_loss.ndim == 0
    assert output.identity_loss.ndim == 0
    assert output.attribute_transport.shape == (4, 6)
    assert output.identity_transport.shape == (4, 5)