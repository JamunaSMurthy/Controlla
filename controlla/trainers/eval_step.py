"""Evaluation and inference helpers for Controlla."""

from __future__ import annotations

from typing import Any

import torch

from controlla.trainers.train_step import move_batch_to_device


@torch.no_grad()
def compute_eval_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    num_inference_steps: int,
) -> dict[str, Any]:
    """Run inference for a single batch."""

    model.eval()
    batch = move_batch_to_device(batch, device)

    return model(
        batch,
        generate=True,
        num_inference_steps=num_inference_steps,
    )


@torch.no_grad()
def compute_validation_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    """Run validation forward pass and compute loss."""

    from controlla.trainers.train_step import compute_train_step

    model.eval()

    return compute_train_step(
        model=model,
        loss_fn=loss_fn,
        batch=batch,
        device=device,
    )