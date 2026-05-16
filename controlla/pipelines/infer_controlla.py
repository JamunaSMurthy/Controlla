"""CLI inference entrypoint for Controlla."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.transforms import build_image_transform
from models.controlla_adapter import ControllaModel
from trainers.eval_step import compute_eval_step
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.seed import set_seed
from utils.visualization import save_image_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Controlla inference")
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--reference-image", type=str, default=None)
    parser.add_argument("--emotion-label", type=int, default=0)
    parser.add_argument("--audio-feature-file", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--diffusion-model-path", type=str, default=None)
    parser.add_argument("--diffusion-local-only", action="store_true")
    return parser.parse_args()


def build_inference_batch(config: dict, args: argparse.Namespace, device: torch.device) -> dict:
    image_size = int(config["model"]["image_size"])
    audio_dim = int(config["model"]["audio_dim"])
    transform = build_image_transform(image_size)

    if args.reference_image is not None:
        reference = transform(Image.open(args.reference_image).convert("RGB"))
        has_reference = torch.tensor([1.0])
    else:
        reference = torch.zeros(3, image_size, image_size)
        has_reference = torch.tensor([0.0])

    if args.audio_feature_file is not None:
        audio = torch.as_tensor(np.load(args.audio_feature_file), dtype=torch.float32).flatten()[:audio_dim]
        if audio.numel() < audio_dim:
            audio = torch.cat([audio, torch.zeros(audio_dim - audio.numel())], dim=0)
        has_audio = torch.tensor([1.0])
    else:
        audio = torch.zeros(audio_dim)
        has_audio = torch.tensor([0.0])

    batch = {
        "prompt": [args.prompt],
        "image": reference.unsqueeze(0),
        "reference_image": reference.unsqueeze(0),
        "emotion_label": torch.tensor([args.emotion_label], dtype=torch.long),
        "audio_features": audio.unsqueeze(0),
        "has_reference": has_reference,
        "has_audio": has_audio,
    }
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.diffusion_model_path is not None:
        config["model"]["diffusion_model_path"] = args.diffusion_model_path
    if args.diffusion_local_only:
        config["model"]["diffusion_local_only"] = True
    set_seed(int(config["seed"]))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = ControllaModel(config).to(device)

    checkpoint_path = args.checkpoint_path or config["infer"].get("checkpoint_path")
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)

    model.eval()
    batch = build_inference_batch(config, args, device)
    outputs = compute_eval_step(model, batch, device, num_inference_steps=int(config["infer"]["num_inference_steps"]))
    output_path = args.output_path or config["infer"]["output_path"]
    save_image_tensor(outputs["generated_images"][0], output_path)
    print(f"saved inference output to {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()