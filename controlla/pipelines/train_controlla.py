"""CLI training entrypoint for Controlla.

This script is intended for public, training-ready use.

Examples:
    python -m controlla.pipelines.train_controlla --config configs/train.yaml --dummy-data

    python -m controlla.pipelines.train_controlla \
        --config experiments/configs/train_controlla.yaml \
        --dataset-path experiments/outputs/manifests/affecthuman43k_manifest.csv \
        --image-root ../AffectHuman-43K \
        --diffusion-model-path /path/to/local/stable-diffusion-v1-5
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from controlla.data.dataset import build_dataloader
from controlla.models.controlla_adapter import ControllaModel
from controlla.trainers.trainer import Trainer
from controlla.utils.config import load_config
from controlla.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Controlla")

    parser.add_argument("--config", type=str, default="configs/train.yaml")

    # Data overrides
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--val-split-name", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)

    parser.add_argument("--prompt-column", type=str, default=None)
    parser.add_argument("--image-column", type=str, default=None)
    parser.add_argument("--reference-column", type=str, default=None)
    parser.add_argument("--emotion-column", type=str, default=None)
    parser.add_argument("--audio-column", type=str, default=None)
    parser.add_argument("--identity-column", type=str, default=None)

    # Model/checkpoint overrides
    parser.add_argument("--diffusion-model-path", type=str, default=None)
    parser.add_argument("--diffusion-local-only", action="store_true")
    parser.add_argument("--diffusion-backbone", type=str, default=None, choices=["mock", "diffusers"])
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    # Training overrides
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--training-iterations", type=int, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dummy-data", action="store_true")

    return parser.parse_args()


def _set_nested(config: dict[str, Any], section: str, key: str, value: Any | None) -> None:
    if value is not None:
        config.setdefault(section, {})[key] = value


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides to loaded config."""

    if args.seed is not None:
        config["seed"] = args.seed

    if args.device is not None:
        config["device"] = args.device

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    if args.mixed_precision:
        config["mixed_precision"] = True

    if args.dataset_path is not None:
        config.setdefault("data", {})["manifest_path"] = args.dataset_path
        config.setdefault("data", {})["csv_path"] = args.dataset_path

    if args.split_dir is not None:
        config.setdefault("data", {})["split_dir"] = args.split_dir
        if args.dataset_path is None:
            config["data"]["manifest_path"] = None
            config["data"]["csv_path"] = None

    _set_nested(config, "data", "split_name", args.split_name)
    _set_nested(config, "data", "val_split_name", args.val_split_name)
    _set_nested(config, "data", "image_root", args.image_root)

    _set_nested(config, "data", "prompt_column", args.prompt_column)
    _set_nested(config, "data", "image_column", args.image_column)
    _set_nested(config, "data", "reference_column", args.reference_column)
    _set_nested(config, "data", "emotion_column", args.emotion_column)
    _set_nested(config, "data", "audio_column", args.audio_column)
    _set_nested(config, "data", "identity_column", args.identity_column)

    _set_nested(config, "model", "diffusion_model_path", args.diffusion_model_path)
    _set_nested(config, "model", "diffusion_backbone", args.diffusion_backbone)

    if args.diffusion_local_only:
        config.setdefault("model", {})["diffusion_local_only"] = True

    _set_nested(config, "train", "batch_size", args.batch_size)
    _set_nested(config, "train", "learning_rate", args.learning_rate)
    _set_nested(config, "train", "num_epochs", args.num_epochs)
    _set_nested(config, "train", "training_iterations", args.training_iterations)

    return config


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cpu"))

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[Controlla] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested)


def prepare_output_dir(config: dict[str, Any], config_path: str) -> Path:
    output_dir = Path(config.get("output_dir", "outputs/train")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_copy_path = output_dir / "resolved_config.json"
    config_copy_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    original_config_path = Path(config_path)
    if original_config_path.exists():
        shutil.copy2(original_config_path, output_dir / original_config_path.name)

    return output_dir


def maybe_load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | None,
    device: torch.device,
    strict: bool = False,
) -> None:
    if checkpoint_path is None:
        return

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"[Controlla] Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[Controlla] Unexpected checkpoint keys: {len(unexpected)}")


def main() -> None:
    args = parse_args()

    config = apply_overrides(load_config(args.config), args)

    seed = int(config.get("seed", 1234))
    set_seed(seed)

    output_dir = prepare_output_dir(config, args.config)
    device = resolve_device(config)

    print(f"[Controlla] Output dir: {output_dir}")
    print(f"[Controlla] Device: {device}")
    print(f"[Controlla] Seed: {seed}")
    print(f"[Controlla] Diffusion backbone: {config['model'].get('diffusion_backbone')}")

    train_loader = build_dataloader(
        config,
        train=True,
        dummy_data=args.dummy_data,
    )

    model = ControllaModel(config).to(device)

    if args.resume or args.checkpoint_path:
        maybe_load_checkpoint(
            model=model,
            checkpoint_path=args.checkpoint_path,
            device=device,
            strict=False,
        )

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
    )

    trainer.fit(train_loader)

    print("[Controlla] Training finished.")


if __name__ == "__main__":
    main()