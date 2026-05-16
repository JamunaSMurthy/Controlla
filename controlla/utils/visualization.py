"""Visualization helpers for saving generated images and debugging grids."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a single image tensor in `[-1, 1]` or `[0, 1]` into a PIL image."""
    tensor = image_tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(array)


def save_image_tensor(image_tensor: torch.Tensor, output_path: str | Path) -> None:
    """Save a single image tensor to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image_tensor).save(output_path)


def save_image_grid(images: torch.Tensor, output_path: str | Path, nrow: int = 4) -> None:
    """Save a batch of image tensors as a grid for qualitative debugging."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid((images.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0, nrow=nrow)
    tensor_to_pil(grid).save(output_path)