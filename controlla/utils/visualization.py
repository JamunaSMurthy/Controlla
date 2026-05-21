"""Visualization helpers for saving generated images and debugging grids."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid


def _to_01(image_tensor: torch.Tensor) -> torch.Tensor:
    """Convert image tensor to [0, 1]."""

    tensor = image_tensor.detach().float().cpu()

    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor [C, H, W], got {tuple(tensor.shape)}")

    if tensor.min() < 0.0:
        tensor = (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
    else:
        tensor = tensor.clamp(0.0, 1.0)

    return tensor.clamp(0.0, 1.0)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a single image tensor into a PIL image.

    Supports tensors in [-1, 1] or [0, 1].
    """

    tensor = _to_01(image_tensor)

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    if tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[0]}")

    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")

    return Image.fromarray(array)


def save_image_tensor(
    image_tensor: torch.Tensor,
    output_path: str | Path,
) -> None:
    """Save a single image tensor to disk."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tensor_to_pil(image_tensor).save(output_path)


def save_image_grid(
    images: torch.Tensor,
    output_path: str | Path,
    nrow: int = 4,
) -> None:
    """Save a batch of image tensors as a grid."""

    if images.ndim != 4:
        raise ValueError(f"Expected batch tensor [B, C, H, W], got {tuple(images.shape)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images_01 = images.detach().float().cpu()

    if images_01.min() < 0.0:
        images_01 = (images_01.clamp(-1.0, 1.0) + 1.0) / 2.0
    else:
        images_01 = images_01.clamp(0.0, 1.0)

    grid = make_grid(images_01, nrow=nrow)
    tensor_to_pil(grid).save(output_path)