"""Evaluation and inference helpers for Controlla."""

from __future__ import annotations

from typing import Any

import torch

from .train_step import move_batch_to_device


@torch.no_grad()
def compute_eval_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    num_inference_steps: int,
) -> dict[str, Any]:
    """Run inference for a single batch."""
    batch = move_batch_to_device(batch, device)
    return model(batch, generate=True, num_inference_steps=num_inference_steps)