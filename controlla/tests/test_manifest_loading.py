"""Regression tests for split-aware manifest loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from controlla.data.dataset import ControllaDataset
from controlla.experiments.metrics.metric_utils import load_prediction_manifest
from controlla.utils.config import load_config
from controlla.utils.manifests import resolve_data_manifest_path


def test_resolve_data_manifest_path_defaults_to_train_split(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.jsonl").write_text("", encoding="utf-8")

    config = {
        "project_root": str(tmp_path),
        "data": {
            "manifest_path": None,
            "csv_path": None,
            "split_dir": "splits",
            "split_name": None,
        },
    }

    resolved = resolve_data_manifest_path(config)
    assert resolved == (split_dir / "train.jsonl").resolve()


def test_controlla_dataset_loads_jsonl_split_manifest(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    image_path = tmp_path / "image.png"
    reference_path = tmp_path / "reference.png"
    audio_path = tmp_path / "audio.npy"

    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(reference_path)
    np.save(audio_path, np.arange(128, dtype=np.float32))

    record = {
        "sample_id": "sample-1",
        "split": "train",
        "dataset_sources": {"image_source": "affectnet"},
        "image_path": str(image_path),
        "reference_image_path": str(reference_path),
        "audio_feature_path": str(audio_path),
        "text": "A fearful portrait.",
        "unified_emotion": "fearful",
        "modality_mask": {
            "has_image": True,
            "has_audio": True,
            "has_text": True,
            "has_reference_image": True,
        },
        "alignment_scores": {"final_score": 0.9},
    }

    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    config = load_config(Path("configs/train.yaml"))
    config["model"]["diffusion_backbone"] = "mock"
    config["model"]["text_encoder_backend"] = "simple"
    config["model"]["image_encoder_backend"] = "cnn"

    config["data"]["manifest_path"] = None
    config["data"]["csv_path"] = None
    config["data"]["split_dir"] = str(split_dir)
    config["data"]["split_name"] = "train"
    config["data"]["image_column"] = "image_path"
    config["data"]["reference_column"] = "reference_image_path"
    config["data"]["prompt_column"] = "text"
    config["data"]["emotion_column"] = "unified_emotion"
    config["data"]["audio_column"] = "audio_feature_path"

    dataset = ControllaDataset(config)
    sample = dataset[0]

    assert sample["prompt"] == "A fearful portrait."
    assert sample["emotion_label"].item() == config["data"]["emotion_taxonomy"].index("fearful")
    assert sample["has_audio"].item() == 1.0
    assert sample["has_reference"].item() == 1.0
    assert sample["audio_features"].shape[0] == int(config["model"]["audio_dim"])


def test_load_prediction_manifest_supports_jsonl(tmp_path: Path) -> None:
    manifest_path = tmp_path / "predictions.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "a", "text": "happy", "unified_emotion": "happy"}),
                json.dumps({"sample_id": "b", "text": "sad", "unified_emotion": "sad"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    frame = load_prediction_manifest(str(manifest_path))

    assert frame["sample_id"].tolist() == ["a", "b"]
    assert frame["unified_emotion"].tolist() == ["happy", "sad"]
    assert "target_emotion" in frame.columns