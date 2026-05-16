"""Single training step for Controlla."""

from __future__ import annotations

from typing import Any

import torch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch to the target device."""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_train_step(model: torch.nn.Module, loss_fn: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    """Run forward propagation and compute the weighted Controlla objective."""
    batch = move_batch_to_device(batch, device)
    states = model(batch, generate=False)

    generated_image_embedding = model.image_projection(model.image_encoder(states["generated_images"]))
    loss_components = {
        "generated_identity": generated_image_embedding,
        "target_identity": states["identity_embedding"],
        "has_reference": batch["has_reference"],
        "z_attr": states["graph_output"].z_attr,
        "target_emotion": states["emotion_embedding"],
        "text_embedding": model.text_projection(states["text_output"].pooled_embedding),
        "fused_embedding": states["graph_output"].fused_embedding,
        "graph_loss": states["ot_output"].loss,
        "diffusion_loss": states["backbone_output"].diffusion_loss,
    }
    loss, metrics = loss_fn(loss_components)
    return loss, metrics, states