"""Weighted loss aggregation for Controlla."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .contrastive_loss import ContrastiveLoss
from .emotion_loss import EmotionConsistencyLoss
from .graph_loss import GraphConsistencyLoss
from .identity_loss import IdentityPreservationLoss
from .regularization_loss import LipschitzSmoothnessLoss, OrthogonalityLoss


class ControllaLoss(nn.Module):
    """Combine Controlla training losses with configurable weights.

    Expected component keys:
        diffusion_loss: scalar tensor
        graph_loss: scalar tensor from OTAlignment
        z_attr: [B, z_dim]
        z_id: [B, z_dim]
        target_emotion: [B, hidden_dim]
        text_embedding: [B, hidden_dim]
        fused_embedding: [B, hidden_dim]

    Optional component keys:
        generated_identity: [B, D]
        target_identity: [B, D]
        has_reference: [B]
        emotion_mask: [B]
        latent_a, latent_b, output_a, output_b for Lipschitz loss
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        model_config = config["model"]
        self.weights = dict(config.get("loss_weights", {}))

        self.identity_loss = IdentityPreservationLoss(
            loss_type=str(model_config.get("identity_loss_type", "cosine")),
        )

        self.emotion_loss = EmotionConsistencyLoss(
            input_dim=int(model_config["z_dim"]),
            target_dim=int(model_config["hidden_dim"]),
            loss_type=str(model_config.get("emotion_loss_type", "cosine")),
        )

        self.contrastive_loss = ContrastiveLoss(
            temperature=float(model_config.get("contrastive_temperature", 0.07)),
        )

        self.graph_loss = GraphConsistencyLoss(weight=1.0)

        self.orthogonality_loss = OrthogonalityLoss()

        self.lipschitz_loss = LipschitzSmoothnessLoss(
            lipschitz_constant=float(model_config.get("lipschitz_constant", 1.0)),
        )

    def _weight(self, name: str) -> float:
        return float(self.weights.get(name, 0.0))

    @staticmethod
    def _zero_like(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    @staticmethod
    def _metric_value(value: torch.Tensor) -> float:
        return float(value.detach().float().item())

    def forward(
        self,
        components: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if "diffusion_loss" not in components:
            raise KeyError("components must contain 'diffusion_loss'")

        diffusion = components["diffusion_loss"]
        total = self._weight("diffusion") * diffusion

        zero = self._zero_like(diffusion)

        identity = zero
        emotion = zero
        contrastive = zero
        graph = zero
        orthogonality = zero
        lipschitz = zero

        if self._weight("identity") > 0:
            required = {"generated_identity", "target_identity", "has_reference"}
            if required.issubset(components):
                identity = self.identity_loss(
                    generated_identity=components["generated_identity"],
                    target_identity=components["target_identity"],
                    has_reference=components["has_reference"],
                )
                total = total + self._weight("identity") * identity

        if self._weight("emotion") > 0:
            required = {"z_attr", "target_emotion"}
            if required.issubset(components):
                emotion = self.emotion_loss(
                    z_attr=components["z_attr"],
                    target_emotion=components["target_emotion"],
                    mask=components.get("emotion_mask"),
                )
                total = total + self._weight("emotion") * emotion

        if self._weight("contrastive") > 0:
            required = {"text_embedding", "fused_embedding"}
            if required.issubset(components):
                contrastive = self.contrastive_loss(
                    components["text_embedding"],
                    components["fused_embedding"],
                )
                total = total + self._weight("contrastive") * contrastive

        if self._weight("graph") > 0:
            if "graph_loss" in components:
                graph = self.graph_loss(components["graph_loss"])
                total = total + self._weight("graph") * graph

        if self._weight("orthogonality") > 0:
            required = {"z_attr", "z_id"}
            if required.issubset(components):
                orthogonality = self.orthogonality_loss(
                    components["z_attr"],
                    components["z_id"],
                )
                total = total + self._weight("orthogonality") * orthogonality

        if self._weight("lipschitz") > 0:
            required = {"latent_a", "latent_b", "output_a", "output_b"}
            if required.issubset(components):
                lipschitz = self.lipschitz_loss(
                    latent_a=components["latent_a"],
                    latent_b=components["latent_b"],
                    output_a=components["output_a"],
                    output_b=components["output_b"],
                )
                total = total + self._weight("lipschitz") * lipschitz

        metrics = {
            "loss_total": self._metric_value(total),
            "loss_diffusion": self._metric_value(diffusion),
            "loss_identity": self._metric_value(identity),
            "loss_emotion": self._metric_value(emotion),
            "loss_graph": self._metric_value(graph),
            "loss_contrastive": self._metric_value(contrastive),
            "loss_orthogonality": self._metric_value(orthogonality),
            "loss_lipschitz": self._metric_value(lipschitz),
        }

        return total, metrics