"""Tests for final benchmark bundle metadata artifacts."""

from __future__ import annotations

from pathlib import Path

from dataset_builder.outputs.benchmark_bundle import build_benchmark_manifest, render_benchmark_card, write_benchmark_bundle
from dataset_builder.schemas import AlignmentScores, DatasetSources, ModalityMask
from dataset_builder.utils.io import read_json, write_json, write_jsonl


def test_build_benchmark_manifest_includes_eval_and_checksums(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir(parents=True, exist_ok=True)
    train_jsonl = package_root / "all_aligned" / "train.jsonl"
    write_jsonl([{"sample_id": "s1"}], train_jsonl)
    export_summary = {
        "source_manifest": str(tmp_path / "aligned_dataset.jsonl"),
        "num_samples": 10,
        "all_outputs": {"train_jsonl": str(train_jsonl)},
        "mode_outputs": {},
    }
    stats_report = {
        "split_counts": {"train": 8, "val": 1, "test": 1},
        "modality_coverage": {"has_audio": 1.0, "has_text": 1.0, "has_reference_image": 1.0},
        "alignment_score_stats": {"min": 0.5, "mean": 0.8, "max": 0.95},
    }
    eval_summary = {
        "eval_bench_path": str(tmp_path / "eval_bench.jsonl"),
        "num_samples": 8,
        "size_per_class": 1,
        "emotion_counts": {"happy": 1},
    }
    write_jsonl([{"sample_id": "bench1"}], Path(eval_summary["eval_bench_path"]))
    manifest = build_benchmark_manifest(export_summary, stats_report, eval_summary, package_root=package_root)
    assert manifest["eval_bench"]["num_samples"] == 8
    assert str(train_jsonl) in manifest["artifact_checksums"]


def test_render_benchmark_card_mentions_eval_bench() -> None:
    manifest = {
        "benchmark_name": "Controlla",
        "benchmark_version": "0.1.0",
        "num_total_samples": 10,
        "train_val_test_counts": {"train": 8, "val": 1, "test": 1},
        "eval_bench": {"num_samples": 8, "size_per_class": 1},
        "modality_coverage": {"has_audio": 1.0, "has_text": 1.0, "has_reference_image": 1.0},
        "emotion_taxonomy": ["happy", "sad"],
        "alignment_score_stats": {"min": 0.5, "mean": 0.8, "max": 0.95},
        "artifact_checksums": {"file": "abc"},
    }
    card = render_benchmark_card(manifest)
    assert "Controlla Benchmark Card" in card
    assert "Eval bench" in card


def test_write_benchmark_bundle_writes_outputs(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir(parents=True, exist_ok=True)
    export_summary = {
        "source_manifest": str(tmp_path / "aligned_dataset.jsonl"),
        "num_samples": 10,
        "all_outputs": {},
        "mode_outputs": {},
    }
    stats_report = {
        "split_counts": {"train": 8, "val": 1, "test": 1},
        "modality_coverage": {"has_audio": 1.0, "has_text": 1.0, "has_reference_image": 1.0},
        "alignment_score_stats": {"min": 0.5, "mean": 0.8, "max": 0.95},
    }
    eval_summary = {
        "eval_bench_path": str(tmp_path / "eval_bench.jsonl"),
        "num_samples": 8,
        "size_per_class": 1,
        "emotion_counts": {"happy": 1},
    }
    write_json(export_summary, package_root / "export_summary.json")
    write_json(stats_report, package_root / "stats_report.json")
    write_json(eval_summary, tmp_path / "eval_bench_summary.json")
    write_jsonl([{"sample_id": "bench1"}], Path(eval_summary["eval_bench_path"]))
    outputs = write_benchmark_bundle(
        package_root,
        package_root / "export_summary.json",
        package_root / "stats_report.json",
        tmp_path / "eval_bench_summary.json",
    )
    assert Path(outputs["benchmark_manifest"]).exists()
    assert Path(outputs["benchmark_card"]).exists()


def test_write_benchmark_bundle_from_split_dir(tmp_path: Path) -> None:
    split_dir = tmp_path / "strict_splits"
    write_jsonl(
        [
            {
                "sample_id": "train1",
                "split": "train",
                "dataset_sources": DatasetSources(image_source="affectnet").model_dump(mode="json"),
                "image_path": "/tmp/train1.jpg",
                "reference_image_path": "/tmp/train1_ref.jpg",
                "audio_feature_path": "/tmp/train1.npy",
                "text": "happy",
                "unified_emotion": "happy",
                "modality_mask": ModalityMask(has_image=True, has_audio=False, has_text=True, has_reference_image=True).model_dump(mode="json"),
                "alignment_scores": AlignmentScores(final_score=0.9).model_dump(mode="json"),
            }
        ],
        split_dir / "train.jsonl",
    )
    write_jsonl(
        [
            {
                "sample_id": "val1",
                "split": "val",
                "dataset_sources": DatasetSources(image_source="affectnet").model_dump(mode="json"),
                "image_path": "/tmp/val1.jpg",
                "reference_image_path": "/tmp/val1_ref.jpg",
                "audio_feature_path": "/tmp/val1.npy",
                "text": "sad",
                "unified_emotion": "sad",
                "modality_mask": ModalityMask(has_image=True, has_audio=False, has_text=True, has_reference_image=True).model_dump(mode="json"),
                "alignment_scores": AlignmentScores(final_score=0.8).model_dump(mode="json"),
            }
        ],
        split_dir / "val.jsonl",
    )
    write_jsonl(
        [
            {
                "sample_id": "test1",
                "split": "test",
                "dataset_sources": DatasetSources(image_source="affectnet").model_dump(mode="json"),
                "image_path": "/tmp/test1.jpg",
                "reference_image_path": "/tmp/test1_ref.jpg",
                "audio_feature_path": "/tmp/test1.npy",
                "text": "surprised",
                "unified_emotion": "surprised",
                "modality_mask": ModalityMask(has_image=True, has_audio=False, has_text=True, has_reference_image=True).model_dump(mode="json"),
                "alignment_scores": AlignmentScores(final_score=0.95).model_dump(mode="json"),
            }
        ],
        split_dir / "test.jsonl",
    )
    write_json(
        {
            "config": {"input_path": "outputs/aligned/aligned_dataset.jsonl"},
            "split_counts": {"train": 1, "val": 1, "test": 1},
        },
        split_dir / "split_report.json",
    )
    write_json(
        {
            "eval_bench_path": str(split_dir / "eval_bench.jsonl"),
            "num_samples": 1,
            "size_per_class": 1,
            "emotion_counts": {"surprised": 1},
        },
        split_dir / "eval_bench_summary.json",
    )
    write_jsonl([{"sample_id": "test1"}], split_dir / "eval_bench.jsonl")

    outputs = write_benchmark_bundle(package_root=split_dir / "benchmark", split_dir=split_dir)
    manifest = read_json(outputs["benchmark_manifest"])
    assert Path(outputs["benchmark_manifest"]).exists()
    assert Path(outputs["benchmark_card"]).exists()
    assert manifest["packaged_views"]["all_aligned"]["train_jsonl"] == str(split_dir / "train.jsonl")