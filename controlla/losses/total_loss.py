"""Weighted loss aggregation for Controlla."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .contrastive_loss import ContrastiveLoss
from .emotion_loss import EmotionConsistencyLoss
from .graph_loss import GraphConsistencyLoss
from .identity_loss import IdentityPreservationLoss


class ControllaLoss(nn.Module):
    """Combine all research losses with configurable weights."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config["model"]
        self.weights = config["loss_weights"]
        self.identity_loss = IdentityPreservationLoss()
        self.emotion_loss = EmotionConsistencyLoss(
            input_dim=int(model_config["z_dim"]),
            target_dim=int(model_config["hidden_dim"]),
        )
        self.contrastive_loss = ContrastiveLoss()
        self.graph_loss = GraphConsistencyLoss()

    def forward(self, components: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        identity = self.identity_loss(
            generated_identity=components["generated_identity"],
            target_identity=components["target_identity"],
            has_reference=components["has_reference"],
        )
        emotion = self.emotion_loss(components["z_attr"], components["target_emotion"])
        contrastive = self.contrastive_loss(components["text_embedding"], components["fused_embedding"])
        graph = self.graph_loss(components["graph_loss"])
        diffusion = components["diffusion_loss"]

        total = (
            self.weights["diffusion"] * diffusion
            + self.weights["identity"] * identity
            + self.weights["emotion"] * emotion
            + self.weights["graph"] * graph
            + self.weights["contrastive"] * contrastive
        )
        metrics = {
            "loss_total": float(total.detach().item()),
            "loss_diffusion": float(diffusion.detach().item()),
            "loss_identity": float(identity.detach().item()),
            "loss_emotion": float(emotion.detach().item()),
            "loss_graph": float(graph.detach().item()),
            "loss_contrastive": float(contrastive.detach().item()),
        }
        return total, metrics