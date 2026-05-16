"""Tests for leakage remediation planning and example artifact export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dataset_leakage_checker.artifacts import export_leakage_examples
from dataset_leakage_checker.remediation import apply_remediation_plan, build_remediation_plan, write_remediation_outputs


def _img(path: Path, color: int) -> None:
    Image.new("RGB", (8, 8), color=(color, color, color)).save(path)


def test_remediation_plan_moves_leaked_identity_and_speaker() -> None:
    samples = [
        {"sample_id": "a", "split": "train", "identity_id": "id1", "speaker_id": "sp1", "dataset_source": "affectnet"},
        {"sample_id": "b", "split": "test", "identity_id": "id1", "speaker_id": "sp1", "dataset_source": "rafdb"},
        {"sample_id": "c", "split": "val", "identity_id": "id2", "speaker_id": None, "dataset_source": "celebahq"},
    ]
    report = {
        "identity_leakage": {"explicit_identity_id": {"leaked_identities": [{"identity_id": "id1", "sample_ids": ["a", "b"]}] }},
        "speaker_leakage": {"explicit_speaker_id": {"leaked_speakers": [{"speaker_id": "sp1", "sample_ids": ["a", "b"]}] }},
        "cross_modal_leakage": {"identity_emotion_overlap": {"items": []}, "speaker_emotion_overlap": {"items": []}},
    }
    plan = build_remediation_plan(report, samples)
    assert plan["num_proposed_moves"] == 1
    remediated = apply_remediation_plan(samples, plan)
    moved = {sample["sample_id"]: sample["split"] for sample in remediated}
    assert moved["b"] == "train"


def test_export_leakage_examples_writes_files(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    audio_left = tmp_path / "left.wav"
    audio_right = tmp_path / "right.wav"
    _img(left, 50)
    _img(right, 60)
    audio_left.write_bytes(b"left")
    audio_right.write_bytes(b"right")
    samples = [
        {"sample_id": "a", "image_path": str(left), "reference_image_path": None, "audio_path": str(audio_left), "text": "left text"},
        {"sample_id": "b", "image_path": str(right), "reference_image_path": None, "audio_path": str(audio_right), "text": "right text"},
    ]
    report = {
        "identity_leakage": {"explicit_identity_id": {"examples": [{"sample_ids": ["a", "b"]}]}},
        "duplicate_detection": {
            "image_duplicates": {"near_duplicates": {"examples": [{"left_image_path": str(left), "right_image_path": str(right)}]}},
            "audio_duplicates": {"near_duplicates": {"examples": [{"left_sample_id": "a", "right_sample_id": "b"}]}},
            "text_duplicates": {"fuzzy_duplicates": {"examples": [{"left_text_preview": "left text", "right_text_preview": "right text"}]}},
        },
    }
    outputs = export_leakage_examples(samples, report, tmp_path / "artifacts", max_examples_per_type=1)
    assert outputs["identity_examples"] == 1
    assert outputs["image_duplicate_examples"] == 1
    assert outputs["audio_duplicate_examples"] == 1
    assert outputs["text_duplicate_examples"] == 1
    assert Path(outputs["manifest"]).exists()


def test_write_remediation_outputs_strips_transient_arrays(tmp_path: Path) -> None:
    plan = {"num_proposed_moves": 1, "projected_split_counts": {"train": 2}, "examples": [], "moves": []}
    remediated_samples = [{"sample_id": "a", "split": "train", "_image_embedding": np.ones(4, dtype=np.float32)}]
    outputs = write_remediation_outputs(tmp_path, plan, remediated_samples)
    content = Path(outputs["remediated_jsonl"]).read_text(encoding="utf-8")
    assert "_image_embedding" not in content