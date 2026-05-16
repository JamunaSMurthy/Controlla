"""Forward-pass smoke tests for the full Controlla model."""

from __future__ import annotations

from pathlib import Path

from data.dataset import DummyControllaDataset
from models.controlla_adapter import ControllaModel
from utils.config import load_config


def test_controlla_forward_train() -> None:
    config = load_config(Path("configs/train.yaml"))
    dataset = DummyControllaDataset(config)
    batch = dataset[0]
    batch = {
        "prompt": [batch["prompt"]],
        "image": batch["image"].unsqueeze(0),
        "reference_image": batch["reference_image"].unsqueeze(0),
        "emotion_label": batch["emotion_label"].unsqueeze(0),
        "audio_features": batch["audio_features"].unsqueeze(0),
        "has_reference": batch["has_reference"].unsqueeze(0),
        "has_audio": batch["has_audio"].unsqueeze(0),
    }
    model = ControllaModel(config)
    outputs = model(batch, generate=False)
    assert outputs["generated_images"].shape == batch["image"].shape
    assert outputs["graph_output"].z_id.shape[0] == 1


def test_controlla_forward_infer() -> None:
    config = load_config(Path("configs/infer.yaml"))
    dataset = DummyControllaDataset(config)
    batch = dataset[0]
    batch = {
        "prompt": [batch["prompt"]],
        "image": batch["image"].unsqueeze(0),
        "reference_image": batch["reference_image"].unsqueeze(0),
        "emotion_label": batch["emotion_label"].unsqueeze(0),
        "audio_features": batch["audio_features"].unsqueeze(0),
        "has_reference": batch["has_reference"].unsqueeze(0),
        "has_audio": batch["has_audio"].unsqueeze(0),
    }
    model = ControllaModel(config)
    outputs = model(batch, generate=True, num_inference_steps=2)
    assert outputs["generated_images"].shape == batch["image"].shape