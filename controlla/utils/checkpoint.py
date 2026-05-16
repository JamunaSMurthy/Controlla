"""Checkpoint helpers for saving and restoring prototype runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], checkpoint_path: str | Path) -> None:
    """Persist a training checkpoint to disk."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path, _use_new_zipfile_serialization=False)


def load_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dictionary from disk."""
    return torch.load(Path(checkpoint_path), map_location=map_location)