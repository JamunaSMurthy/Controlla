"""Tests for the dataset leakage checker package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_leakage_checker.main import load_samples, run_all_checks


def _write_image(path: Path, color: int) -> None:
    image = Image.new("L", (16, 16), color=color)
    image.save(path)


def _write_audio(path: Path, content: bytes) -> None:
    path.write_bytes(content)


def test_leakage_checker_detects_cross_split_leaks(tmp_path: Path) -> None:
    image_train = tmp_path / "image_train.png"
    image_test = tmp_path / "image_test.png"
    image_other = tmp_path / "image_other.png"
    _write_image(image_train, 120)
    _write_image(image_test, 120)
    _write_image(image_other, 220)

    audio_train = tmp_path / "audio_train.wav"
    audio_test = tmp_path / "audio_test.wav"
    _write_audio(audio_train, b"same-audio")
    _write_audio(audio_test, b"same-audio")

    data_path = tmp_path / "samples.jsonl"
    records = [
        {
            "sample_id": "train-1",
            "split": "train",
            "image_path": str(image_train),
            "audio_path": str(audio_train),
            "text": "A calm neutral description.",
            "identity_id": "person-1",
            "speaker_id": "speaker-1",
            "dataset_source": "affectnet",
            "unified_emotion": "neutral",
        },
        {
            "sample_id": "test-1",
            "split": "test",
            "image_path": str(image_test),
            "audio_path": str(audio_test),
            "text": "A calm neutral description.",
            "identity_id": "person-1",
            "speaker_id": "speaker-1",
            "dataset_source": "rafdb",
            "unified_emotion": "neutral",
        },
        {
            "sample_id": "val-1",
            "split": "val",
            "image_path": str(image_other),
            "text": "Something else entirely.",
            "dataset_source": "celebahq",
            "unified_emotion": "happy",
        },
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    samples = load_samples(data_path)
    results = run_all_checks(
        samples,
        {
            "batch_size": 8,
            "image_embedding_dim": 32,
            "audio_embedding_dim": 32,
            "text_embedding_dim": 32,
            "identity_similarity_threshold": 0.9,
            "speaker_similarity_threshold": 0.9,
            "image_similarity_threshold": 0.95,
            "audio_similarity_threshold": 0.95,
            "text_similarity_threshold": 0.95,
            "text_fuzzy_ratio_threshold": 0.95,
            "image_phash_hamming_threshold": 2,
            "image_phash_prefix_length": 4,
            "enable_identity": True,
            "enable_speaker": True,
            "enable_duplicates": True,
            "enable_cross_modal": True,
            "enable_cross_dataset": True,
            "severity_thresholds": {"minor_percent": 0.1, "major_percent": 1.0, "minor_count": 1, "major_count": 10},
        },
    )
    assert results["summary"]["identity_leakage"] > 0.0
    assert results["summary"]["speaker_leakage"] > 0.0
    assert results["summary"]["audio_duplicates"] > 0
    assert results["summary"]["text_duplicates"] > 0
    assert results["summary"]["cross_dataset_overlap"] > 0