"""Forward-pass smoke tests for the full Controlla model."""

from __future__ import annotations

from pathlib import Path

import torch

from controlla.data.dataset import DummyControllaDataset
from controlla.models.controlla_adapter import ControllaModel
from controlla.utils.config import load_config


def _make_batch(config: dict) -> dict:
    dataset = DummyControllaDataset(config)
    sample = dataset[0]

    return {
        "prompt": [sample["prompt"]],
        "image": sample["image"].unsqueeze(0),
        "reference_image": sample["reference_image"].unsqueeze(0),
        "emotion_label": sample["emotion_label"].unsqueeze(0),
        "audio_features": sample["audio_features"].unsqueeze(0),
        "has_reference": sample["has_reference"].unsqueeze(0),
        "has_audio": sample["has_audio"].unsqueeze(0),
    }


def _force_mock_config(config: dict) -> dict:
    """Ensure tests do not require external diffusion/CLIP checkpoints."""

    config["device"] = "cpu"
    config["mixed_precision"] = False

    config.setdefault("model", {})
    config["model"]["diffusion_backbone"] = "mock"
    config["model"]["text_encoder_backend"] = "simple"
    config["model"]["image_encoder_backend"] = "cnn"
    config["model"]["diffusion_local_only"] = True

    config.setdefault("data", {})
    config["data"]["dummy_num_samples"] = 4
    config["data"]["image_size"] = int(config["model"].get("image_size", config["data"].get("image_size", 64)))

    return config


def test_controlla_forward_train() -> None:
    config = _force_mock_config(load_config(Path("configs/train.yaml")))

    batch = _make_batch(config)
    model = ControllaModel(config)

    with torch.no_grad():
        outputs = model(batch, generate=False)

    assert "generated_images" in outputs
    assert outputs["generated_images"].shape == batch["image"].shape
    assert outputs["graph_output"].z_id.shape[0] == 1
    assert outputs["graph_output"].z_attr.shape[0] == 1
    assert outputs["ot_output"].loss.ndim == 0


def test_controlla_forward_infer() -> None:
    config = _force_mock_config(load_config(Path("configs/infer.yaml")))

    batch = _make_batch(config)
    model = ControllaModel(config)

    with torch.no_grad():
        outputs = model(batch, generate=True, num_inference_steps=2)

    assert "generated_images" in outputs
    assert outputs["generated_images"].shape == batch["image"].shape