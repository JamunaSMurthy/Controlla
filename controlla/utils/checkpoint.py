"""Checkpoint helpers for Controlla training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    payload: dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    """Save a PyTorch checkpoint safely."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a PyTorch checkpoint."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return torch.load(checkpoint_path, map_location=map_location)


def extract_model_state_dict(checkpoint: dict[str, Any] | Any) -> dict[str, torch.Tensor]:
    """Extract model state dict from common checkpoint structures."""

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

    return checkpoint


def load_model_state(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load checkpoint weights into a model and return missing/unexpected keys."""

    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    state_dict = extract_model_state_dict(checkpoint)

    result = model.load_state_dict(state_dict, strict=strict)

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    return missing, unexpected