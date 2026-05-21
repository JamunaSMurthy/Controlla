"""Image transform helpers for Controlla datasets."""

from __future__ import annotations

from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_image_transform(
    image_size: int,
    mode: str = "diffusion",
) -> transforms.Compose:
    """Build image preprocessing pipeline.

    Args:
        image_size: Output spatial size.
        mode:
            "diffusion" or "cnn": values normalized to [-1, 1].
            "clip": CLIP-style normalization.

    Returns:
        torchvision transform.
    """

    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    if mode in {"diffusion", "cnn"}:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif mode == "clip":
        mean = CLIP_MEAN
        std = CLIP_STD
    else:
        raise ValueError(f"Unsupported image transform mode: {mode}")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )