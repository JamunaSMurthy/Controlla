"""Adapter and system assembly for Controlla.

The adapter is the Controlla-native equivalent of a lightweight control pathway:
it turns fused multimodal latents into conditioning tokens consumed by the
diffusion backbone without modifying external repositories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .audio_encoder import AudioEncoder
from .diffusion_backbone import build_diffusion_backbone
from .emotion_encoder import EmotionEncoder
from .graph_fusion import GraphFusion
from .identity_encoder import IdentityEncoder
from .image_encoder import ImageEncoder
from .ot_alignment import OTAlignment
from .text_encoder import TextEncoder


@dataclass
class AdapterOutput:
    """Adapter outputs injected into the diffusion backbone.

    Attributes:
        condition_tokens: Tensor with shape [B, K, C].
        global_condition: Tensor with shape [B, D].
    """

    condition_tokens: torch.Tensor
    global_condition: torch.Tensor


class ControllaAdapter(nn.Module):
    """Project fused Controlla latents into cross-attention tokens."""

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        adapter_tokens: int,
        cross_attention_dim: int,
        include_text_summary_token: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        if adapter_tokens <= 0:
            raise ValueError(f"adapter_tokens must be positive, got {adapter_tokens}")
        if cross_attention_dim <= 0:
            raise ValueError(f"cross_attention_dim must be positive, got {cross_attention_dim}")

        self.adapter_tokens = adapter_tokens
        self.cross_attention_dim = cross_attention_dim
        self.include_text_summary_token = include_text_summary_token

        input_dim = hidden_dim + z_dim * 2

        self.global_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.token_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, adapter_tokens * cross_attention_dim),
        )

        self.text_token_projection = nn.Linear(hidden_dim, cross_attention_dim)

    def forward(
        self,
        fused_embedding: torch.Tensor,
        z_id: torch.Tensor,
        z_attr: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> AdapterOutput:
        if fused_embedding.ndim != 2:
            raise ValueError(f"fused_embedding must have shape [B, D], got {tuple(fused_embedding.shape)}")
        if z_id.ndim != 2:
            raise ValueError(f"z_id must have shape [B, Z], got {tuple(z_id.shape)}")
        if z_attr.ndim != 2:
            raise ValueError(f"z_attr must have shape [B, Z], got {tuple(z_attr.shape)}")
        if text_tokens.ndim != 3:
            raise ValueError(f"text_tokens must have shape [B, T, D], got {tuple(text_tokens.shape)}")

        batch_size = fused_embedding.shape[0]

        if z_id.shape[0] != batch_size or z_attr.shape[0] != batch_size or text_tokens.shape[0] != batch_size:
            raise ValueError("Batch size mismatch in ControllaAdapter inputs")

        combined = torch.cat([fused_embedding, z_id, z_attr], dim=-1)

        global_condition = self.global_projection(combined)

        learned_tokens = self.token_projection(combined)
        learned_tokens = learned_tokens.view(
            batch_size,
            self.adapter_tokens,
            self.cross_attention_dim,
        )

        text_summary = self.text_token_projection(text_tokens.mean(dim=1)).unsqueeze(1)

        if self.include_text_summary_token:
            condition_tokens = torch.cat([text_summary, learned_tokens], dim=1)
        else:
            condition_tokens = learned_tokens + text_summary

        return AdapterOutput(
            condition_tokens=condition_tokens,
            global_condition=global_condition,
        )


class ControllaModel(nn.Module):
    """End-to-end Controlla model."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.config = config
        model_config = config["model"]
        self.ablations = config["ablations"]

        hidden_dim = int(model_config["hidden_dim"])
        z_dim = int(model_config["z_dim"])

        self.text_encoder = TextEncoder(
            hidden_dim=int(model_config["text_dim"]),
            max_tokens=int(model_config["max_text_tokens"]),
            backend=str(model_config.get("text_encoder_backend", "simple")),
            pretrained_name=str(model_config.get("clip_text_model", "openai/clip-vit-base-patch32")),
            local_files_only=bool(model_config.get("clip_local_only", True)),
            freeze=bool(model_config.get("freeze_text_encoder", True)),
        )

        self.image_encoder = ImageEncoder(
            hidden_dim=int(model_config["image_dim"]),
            backend=str(model_config.get("image_encoder_backend", "cnn")),
            model_name=str(model_config.get("clip_vision_model", "openai/clip-vit-base-patch32")),
            local_files_only=bool(model_config.get("clip_local_only", True)),
            freeze=bool(model_config.get("freeze_image_encoder", True)),
        )

        self.identity_encoder = IdentityEncoder(hidden_dim=hidden_dim)

        self.audio_encoder = AudioEncoder(
            input_dim=int(model_config["audio_dim"]),
            hidden_dim=hidden_dim,
        )

        self.emotion_encoder = EmotionEncoder(
            num_emotions=int(model_config["num_emotions"]),
            hidden_dim=hidden_dim,
        )

        self.graph_fusion = GraphFusion(
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            num_layers=int(model_config["graph_layers"]),
            dropout=float(model_config["graph_dropout"]),
            num_heads=int(model_config.get("graph_heads", 4)),
        )

        ot_mode = str(model_config.get("ot_solver", "fgw"))

        self.ot_alignment = OTAlignment(
            hidden_dim=hidden_dim,
            enabled=not bool(self.ablations.get("no_ot", False)),
            mode=ot_mode,
            regularization=float(model_config.get("ot_regularization", 0.05)),
            iterations=int(model_config.get("ot_iterations", 20)),
            sinkhorn_iterations=int(model_config.get("ot_sinkhorn_iterations", 50)),
        )

        self.adapter = ControllaAdapter(
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            adapter_tokens=int(model_config["adapter_tokens"]),
            cross_attention_dim=int(model_config["cross_attention_dim"]),
            include_text_summary_token=bool(model_config.get("include_text_summary_token", True)),
        )

        self.diffusion_backbone = build_diffusion_backbone(config)

        self.text_projection = nn.Linear(int(model_config["text_dim"]), hidden_dim)
        self.image_projection = nn.Linear(int(model_config["image_dim"]), hidden_dim)

    def _get_bool_ablation(self, key: str) -> bool:
        return bool(self.ablations.get(key, False))

    def _build_node_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["has_reference"].shape[0]
        device = batch["has_reference"].device

        text_mask = torch.ones(batch_size, device=device)
        identity_mask = batch["has_reference"].float().clone()
        emotion_mask = torch.ones(batch_size, device=device)
        image_mask = torch.ones(batch_size, device=device)
        audio_mask = batch["has_audio"].float().clone()

        if self._get_bool_ablation("no_identity_branch"):
            identity_mask.zero_()

        if self._get_bool_ablation("no_audio"):
            audio_mask.zero_()

        if self._get_bool_ablation("text_only"):
            identity_mask.zero_()
            emotion_mask.zero_()
            image_mask.zero_()
            audio_mask.zero_()

        if self._get_bool_ablation("image_text"):
            emotion_mask.zero_()
            audio_mask.zero_()

        if self._get_bool_ablation("text_audio"):
            identity_mask.zero_()
            image_mask.zero_()

        return torch.stack(
            [text_mask, identity_mask, emotion_mask, image_mask, audio_mask],
            dim=1,
        )

    def encode_modalities(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Encode all available modalities and return intermediate states."""

        text_output = self.text_encoder(batch["prompt"])
        text_embedding = self.text_projection(text_output.pooled_embedding)

        image_embedding = self.image_projection(self.image_encoder(batch["image"]))

        identity_embedding = self.identity_encoder(
            batch["reference_image"],
            batch["has_reference"],
        )

        audio_embedding = self.audio_encoder(
            batch["audio_features"],
            batch["has_audio"],
        )

        emotion_embedding = self.emotion_encoder(
            batch["emotion_label"],
            audio_embedding,
            batch["has_audio"],
        )

        node_mask = self._build_node_mask(batch)

        graph_output = self.graph_fusion(
            text_embedding=text_embedding,
            identity_embedding=identity_embedding,
            emotion_embedding=emotion_embedding,
            image_embedding=image_embedding,
            audio_embedding=audio_embedding,
            node_mask=node_mask,
            no_graph=self._get_bool_ablation("no_graph"),
        )

        ot_output = self.ot_alignment(
            graph_output.node_embeddings,
            graph_output.adjacency,
            graph_output.node_mask,
        )

        adapter_output = self.adapter(
            fused_embedding=graph_output.fused_embedding,
            z_id=graph_output.z_id,
            z_attr=graph_output.z_attr,
            text_tokens=text_output.token_embeddings,
        )

        return {
            "text_output": text_output,
            "text_embedding": text_embedding,
            "image_embedding": image_embedding,
            "identity_embedding": identity_embedding,
            "audio_embedding": audio_embedding,
            "emotion_embedding": emotion_embedding,
            "graph_output": graph_output,
            "ot_output": ot_output,
            "adapter_output": adapter_output,
        }

    def forward(
        self,
        batch: dict[str, Any],
        generate: bool = False,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        states = self.encode_modalities(batch)

        adapter_output = states["adapter_output"]
        graph_output = states["graph_output"]

        if generate:
            backbone_output = self.diffusion_backbone.generate(
                batch_size=len(batch["prompt"]),
                condition_tokens=adapter_output.condition_tokens,
                global_condition=adapter_output.global_condition,
                z_id=graph_output.z_id,
                z_attr=graph_output.z_attr,
                device=adapter_output.condition_tokens.device,
                num_inference_steps=num_inference_steps
                or int(self.config["infer"]["num_inference_steps"]),
                generator=generator,
            )
        else:
            backbone_output = self.diffusion_backbone(
                target_images=batch["image"],
                condition_tokens=adapter_output.condition_tokens,
                global_condition=adapter_output.global_condition,
                z_id=graph_output.z_id,
                z_attr=graph_output.z_attr,
            )

        states["backbone_output"] = backbone_output
        states["generated_images"] = backbone_output.images

        return states