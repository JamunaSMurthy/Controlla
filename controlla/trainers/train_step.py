"""Single training step for Controlla."""

from __future__ import annotations

from typing import Any

import torch


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move tensor values in a batch to the target device."""

    moved: dict[str, Any] = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value

    return moved


def _encode_generated_identity_proxy(
    model: torch.nn.Module,
    generated_images: torch.Tensor,
) -> torch.Tensor:
    """Encode generated image identity/reference proxy.

    Paper note:
        For released training code, this is an internal identity proxy using the
        model image pathway. Paper-level ID/ID-Sim evaluation should still use
        the frozen evaluation identity encoder such as ArcFace.
    """

    image_features = model.image_encoder(generated_images)
    return model.image_projection(image_features)


def compute_train_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    """Run forward propagation and compute the weighted Controlla objective."""

    batch = move_batch_to_device(batch, device)
    states = model(batch, generate=False)

    graph_output = states["graph_output"]

    generated_identity = _encode_generated_identity_proxy(
        model,
        states["generated_images"],
    )

    text_embedding = model.text_projection(
        states["text_output"].pooled_embedding,
    )

    loss_components = {
        "generated_identity": generated_identity,
        "target_identity": states["identity_embedding"],
        "has_reference": batch["has_reference"],

        "z_attr": graph_output.z_attr,
        "z_id": graph_output.z_id,
        "target_emotion": states["emotion_embedding"],
        "emotion_mask": torch.ones_like(batch["has_reference"]),

        "text_embedding": text_embedding,
        "fused_embedding": graph_output.fused_embedding,

        "graph_loss": states["ot_output"].loss,
        "diffusion_loss": states["backbone_output"].diffusion_loss,
    }

    # Optional latent/output pairs for Lipschitz smoothness if provided by
    # future traversal training code.
    for optional_key in ["latent_a", "latent_b", "output_a", "output_b"]:
        if optional_key in states:
            loss_components[optional_key] = states[optional_key]

    loss, metrics = loss_fn(loss_components)

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite training loss detected: {loss.item()}")

    return loss, metrics, states