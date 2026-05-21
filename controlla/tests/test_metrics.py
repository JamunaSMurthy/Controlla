"""Tests for Controlla paper metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from controlla.experiments.metrics import (
    compute_controllability,
    compute_cross_modal_consistency,
    compute_disentanglement,
    compute_identity_preservation,
)
from controlla.experiments.metrics.paper_metrics import (
    compute_core_paper_metrics,
    compute_retrieval_metrics,
    prepare_prediction_frame,
)


def _write_prediction_manifest(path: Path) -> None:
    rows = [
        {
            "sample_id": "s1",
            "split": "test",
            "prompt": "a happy face",
            "target_emotion": "happy",
            "predicted_emotion": "happy",
            "image_path": "missing-image-1.png",
            "generated_image_path": "missing-generated-1.png",
            "reference_image_path": "missing-reference-1.png",
            "audio_path": "missing-audio-1.wav",
            "identity_id": "id1",
            "z_attr": json.dumps([0.1, 0.2, 0.3]),
            "z_id": json.dumps([0.3, 0.1, 0.0]),
        },
        {
            "sample_id": "s2",
            "split": "test",
            "prompt": "a sad face",
            "target_emotion": "sad",
            "predicted_emotion": "sad",
            "image_path": "missing-image-2.png",
            "generated_image_path": "missing-generated-2.png",
            "reference_image_path": "missing-reference-2.png",
            "audio_path": "missing-audio-2.wav",
            "identity_id": "id1",
            "z_attr": json.dumps([0.0, 0.4, 0.2]),
            "z_id": json.dumps([0.5, 0.2, 0.1]),
        },
    ]

    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def test_core_paper_metrics_smoke(tmp_path: Path) -> None:
    manifest_path = tmp_path / "predictions.csv"
    _write_prediction_manifest(manifest_path)

    frame = prepare_prediction_frame(manifest_path, split="test")
    metrics = compute_core_paper_metrics(frame)

    for key in ["Acc", "TS", "CLIP", "IB", "LDS", "GC", "ID"]:
        assert key in metrics

    assert metrics["Acc"] == 1.0


def test_retrieval_metrics_smoke(tmp_path: Path) -> None:
    manifest_path = tmp_path / "predictions.csv"
    _write_prediction_manifest(manifest_path)

    frame = prepare_prediction_frame(manifest_path, split="test")
    metrics = compute_retrieval_metrics(frame)

    assert "I2T_R@1" in metrics
    assert "I2T_R@5" in metrics
    assert "I2A_R@1" in metrics
    assert "I2A_R@5" in metrics


def test_individual_metric_entrypoints(tmp_path: Path) -> None:
    manifest_path = tmp_path / "predictions.csv"
    _write_prediction_manifest(manifest_path)

    controllability = compute_controllability(str(manifest_path))
    identity = compute_identity_preservation(str(manifest_path))
    consistency = compute_cross_modal_consistency(str(manifest_path))
    disentanglement = compute_disentanglement(str(manifest_path))

    assert "Acc" in controllability
    assert "ID" in identity
    assert "IB" in consistency
    assert "LDS" in disentanglement