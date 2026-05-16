"""Image transform helpers for Controlla datasets."""

from __future__ import annotations

from torchvision import transforms


def build_image_transform(image_size: int) -> transforms.Compose:
    """Build the default image preprocessing pipeline.

    Returned tensors have shape `[3, image_size, image_size]` and values in `[-1, 1]`.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )