"""CLI training entrypoint for Controlla."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from data.dataset import build_dataloader
from models.controlla_adapter import ControllaModel
from trainers.trainer import Trainer
from utils.config import load_config
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Controlla prototype")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--prompt-column", type=str, default=None)
    parser.add_argument("--image-column", type=str, default=None)
    parser.add_argument("--reference-column", type=str, default=None)
    parser.add_argument("--emotion-column", type=str, default=None)
    parser.add_argument("--audio-column", type=str, default=None)
    parser.add_argument("--diffusion-model-path", type=str, default=None)
    parser.add_argument("--diffusion-local-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-save-path", type=str, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--dummy-data", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_path is not None:
        config["data"]["manifest_path"] = args.dataset_path
        config["data"]["csv_path"] = args.dataset_path
    if args.split_dir is not None:
        config["data"]["split_dir"] = args.split_dir
        if args.dataset_path is None:
            config["data"]["manifest_path"] = None
            config["data"]["csv_path"] = None
    if args.split_name is not None:
        config["data"]["split_name"] = args.split_name
    if args.image_root is not None:
        config["data"]["image_root"] = args.image_root
    if args.prompt_column is not None:
        config["data"]["prompt_column"] = args.prompt_column
    if args.image_column is not None:
        config["data"]["image_column"] = args.image_column
    if args.reference_column is not None:
        config["data"]["reference_column"] = args.reference_column
    if args.emotion_column is not None:
        config["data"]["emotion_column"] = args.emotion_column
    if args.audio_column is not None:
        config["data"]["audio_column"] = args.audio_column
    if args.diffusion_model_path is not None:
        config["model"]["diffusion_model_path"] = args.diffusion_model_path
    if args.diffusion_local_only:
        config["model"]["diffusion_local_only"] = True
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        config["train"]["num_epochs"] = args.num_epochs
    if args.checkpoint_save_path is not None:
        config["output_dir"] = str(Path(args.checkpoint_save_path).parent)
    if args.mixed_precision:
        config["mixed_precision"] = True
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["seed"]))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    train_loader = build_dataloader(config, train=True, dummy_data=args.dummy_data)
    model = ControllaModel(config)
    trainer = Trainer(model=model, config=config, device=device)
    trainer.fit(train_loader)


if __name__ == "__main__":
    main()