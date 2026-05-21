"""CLI inference entrypoint for Controlla.

Examples:
    python -m controlla.pipelines.infer_controlla \
        --config configs/infer.yaml \
        --prompt "a happy portrait" \
        --reference-image examples/reference.jpg \
        --emotion-label 1 \
        --output-path outputs/sample.png

For real Stable Diffusion inference:
    python -m controlla.pipelines.infer_controlla \
        --config configs/real_sd_local.yaml \
        --diffusion-model-path /path/to/local/stable-diffusion-v1-5 \
        --checkpoint-path outputs/train/checkpoint.pt \
        --prompt "a surprised portrait" \
        --reference-image examples/reference.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from controlla.data.transforms import build_image_transform
from controlla.models.controlla_adapter import ControllaModel
from controlla.trainers.eval_step import compute_eval_step
from controlla.utils.checkpoint import load_checkpoint
from controlla.utils.config import load_config
from controlla.utils.seed import set_seed
from controlla.utils.visualization import save_image_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Controlla inference")

    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--input-image", type=str, default=None)
    parser.add_argument("--reference-image", type=str, default=None)
    parser.add_argument("--emotion-label", type=int, default=0)
    parser.add_argument("--audio-feature-file", type=str, default=None)

    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)

    parser.add_argument("--diffusion-model-path", type=str, default=None)
    parser.add_argument("--diffusion-backbone", type=str, default=None, choices=["mock", "diffusers"])
    parser.add_argument("--diffusion-local-only", action="store_true")

    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.seed is not None:
        config["seed"] = args.seed

    if args.device is not None:
        config["device"] = args.device

    if args.diffusion_model_path is not None:
        config.setdefault("model", {})["diffusion_model_path"] = args.diffusion_model_path

    if args.diffusion_backbone is not None:
        config.setdefault("model", {})["diffusion_backbone"] = args.diffusion_backbone

    if args.diffusion_local_only:
        config.setdefault("model", {})["diffusion_local_only"] = True

    if args.num_inference_steps is not None:
        config.setdefault("infer", {})["num_inference_steps"] = args.num_inference_steps

    if args.output_path is not None:
        config.setdefault("infer", {})["output_path"] = args.output_path

    if args.checkpoint_path is not None:
        config.setdefault("infer", {})["checkpoint_path"] = args.checkpoint_path

    return config


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cpu"))

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[Controlla] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested)


def _image_size(config: dict[str, Any]) -> int:
    if "data" in config and "image_size" in config["data"]:
        return int(config["data"]["image_size"])
    return int(config["model"].get("image_size", 512))


def _load_image_or_zero(path: str | None, image_size: int, transform_mode: str) -> tuple[torch.Tensor, float]:
    transform = build_image_transform(image_size, mode=transform_mode)

    if path is None:
        return torch.zeros(3, image_size, image_size), 0.0

    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = transform(Image.open(image_path).convert("RGB"))
    return image, 1.0


def _load_audio_or_zero(path: str | None, audio_dim: int) -> tuple[torch.Tensor, float]:
    if path is None:
        return torch.zeros(audio_dim), 0.0

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio feature file not found: {audio_path}")

    audio = torch.as_tensor(np.load(audio_path), dtype=torch.float32).flatten()

    if audio.numel() < audio_dim:
        audio = torch.cat([audio, torch.zeros(audio_dim - audio.numel())], dim=0)

    audio = audio[:audio_dim]
    return audio, 1.0


def build_inference_batch(
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Build one-sample inference batch.

    Controlla expects both:
        image: target/input image slot
        reference_image: identity anchor

    If --input-image is not given, the reference image is reused as the input
    image for reference-grounded editing.
    """

    image_size = _image_size(config)
    audio_dim = int(config["model"]["audio_dim"])

    image_backend = str(config["model"].get("image_encoder_backend", "cnn"))
    transform_mode = "clip" if image_backend == "clip" else "diffusion"

    reference, has_reference = _load_image_or_zero(
        args.reference_image,
        image_size=image_size,
        transform_mode=transform_mode,
    )

    if args.input_image is not None:
        input_image, _ = _load_image_or_zero(
            args.input_image,
            image_size=image_size,
            transform_mode=transform_mode,
        )
    elif args.reference_image is not None:
        input_image = reference.clone()
    else:
        input_image = torch.zeros_like(reference)

    audio, has_audio = _load_audio_or_zero(args.audio_feature_file, audio_dim=audio_dim)

    batch = {
        "prompt": [args.prompt],
        "image": input_image.unsqueeze(0),
        "reference_image": reference.unsqueeze(0),
        "emotion_label": torch.tensor([args.emotion_label], dtype=torch.long),
        "audio_features": audio.unsqueeze(0),
        "has_reference": torch.tensor([has_reference], dtype=torch.float32),
        "has_audio": torch.tensor([has_audio], dtype=torch.float32),
    }

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | None,
    device: torch.device,
) -> None:
    if not checkpoint_path:
        return

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[Controlla] Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[Controlla] Unexpected checkpoint keys: {len(unexpected)}")


def main() -> None:
    args = parse_args()

    config = apply_overrides(load_config(args.config), args)

    seed = int(config.get("seed", 1234))
    set_seed(seed)

    device = resolve_device(config)

    print(f"[Controlla] Device: {device}")
    print(f"[Controlla] Seed: {seed}")
    print(f"[Controlla] Diffusion backbone: {config['model'].get('diffusion_backbone')}")

    model = ControllaModel(config).to(device)

    checkpoint_path = config.get("infer", {}).get("checkpoint_path")
    load_model_checkpoint(model, checkpoint_path, device)

    model.eval()

    batch = build_inference_batch(config, args, device)

    with torch.no_grad():
        outputs = compute_eval_step(
            model,
            batch,
            device,
            num_inference_steps=int(config["infer"]["num_inference_steps"]),
        )

    output_path = config.get("infer", {}).get("output_path", "outputs/infer/sample.png")
    save_image_tensor(outputs["generated_images"][0], output_path)

    print(f"saved inference output to {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()