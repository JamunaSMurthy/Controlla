"""Step 7 tests for split and export packaging."""

from __future__ import annotations

from pathlib import Path

from dataset_builder.outputs.manifest_writer import flatten_sample, write_manifest
from dataset_builder.outputs.split_builder import build_splits
from dataset_builder.outputs.stats_report import build_stats_report
from dataset_builder.runners.build_eval_bench import build_eval_bench, select_eval_bench
from dataset_builder.schemas import AlignedSample, AlignmentScores, DatasetSources, ModalityMask, SplitRatios
from dataset_builder.utils.io import write_jsonl


def _sample(sample_id: str, emotion: str, *, identity_id: str | None, score: float, split: str = "train") -> AlignedSample:
    return AlignedSample(
        sample_id=sample_id,
        split=split,
        dataset_sources=DatasetSources(image_source="affectnet", audio_source="iemocap", text_source="emobank"),
        image_path=f"/tmp/{sample_id}.jpg",
        reference_image_path=f"/tmp/{sample_id}_ref.jpg",
        audio_path=f"/tmp/{sample_id}.wav",
        audio_feature_path=f"/tmp/{sample_id}.npy",
        text=f"Emotion: {emotion}",
        raw_text=f"Emotion: {emotion}",
        unified_emotion=emotion,
        identity_id=identity_id,
        modality_mask=ModalityMask(has_image=True, has_audio=True, has_text=True, has_reference_image=True),
        alignment_scores=AlignmentScores(final_score=score, emotion_match_score=1.0, identity_similarity=0.9, clip_similarity=0.8, imagebind_similarity=0.7),
    )


def test_build_splits_prevents_identity_leakage() -> None:
    samples = [
        _sample("a1", "happy", identity_id="shared", score=0.9),
        _sample("a2", "happy", identity_id="shared", score=0.85),
        _sample("b1", "happy", identity_id="unique", score=0.8),
        _sample("c1", "sad", identity_id="other", score=0.7),
    ]
    split_samples = build_splits(samples, SplitRatios(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25), seed=3)
    shared_splits = {sample.split for sample in split_samples if sample.identity_id == "shared"}
    assert len(shared_splits) == 1


def test_write_manifest_creates_expected_files(tmp_path: Path) -> None:
    samples = [_sample("s1", "happy", identity_id="id1", score=0.9), _sample("s2", "sad", identity_id="id2", score=0.8, split="test")]
    outputs = write_manifest(samples, tmp_path, manifest_name="all_aligned")
    assert Path(outputs["all_jsonl"]).exists()
    assert Path(outputs["train_csv"]).exists()
    assert Path(outputs["test_jsonl"]).exists()


def test_stats_report_contains_split_counts(tmp_path: Path) -> None:
    samples = [_sample("s1", "happy", identity_id="id1", score=0.9), _sample("s2", "sad", identity_id="id2", score=0.8, split="test")]
    report = build_stats_report(samples, tmp_path)
    assert report["num_samples"] == 2
    assert report["split_counts"]["train"] == 1
    assert report["split_counts"]["test"] == 1


def test_select_eval_bench_keeps_top_examples_per_class() -> None:
    samples = [
        _sample("h1", "happy", identity_id="id1", score=0.9, split="test"),
        _sample("h2", "happy", identity_id="id2", score=0.8, split="test"),
        _sample("s1", "sad", identity_id="id3", score=0.95, split="test"),
        _sample("s2", "sad", identity_id="id4", score=0.5, split="test"),
    ]
    selected = select_eval_bench(samples, size_per_class=1)
    assert {sample.sample_id for sample in selected} == {"h1", "s1"}


def test_flatten_sample_exposes_nested_fields() -> None:
    flat = flatten_sample(_sample("flat1", "happy", identity_id="id1", score=0.9))
    assert flat["image_source"] == "affectnet"
    assert flat["has_reference_image"] is True
    assert flat["final_score"] == 0.9


def test_build_eval_bench_reads_strict_split_dir(tmp_path: Path) -> None:
    split_dir = tmp_path / "strict_splits"
    samples = [
        _sample("h1", "happy", identity_id="id1", score=0.9, split="test"),
        _sample("h2", "happy", identity_id="id2", score=0.8, split="test"),
        _sample("s1", "sad", identity_id="id3", score=0.95, split="test"),
        _sample("s2", "sad", identity_id="id4", score=0.5, split="test"),
    ]
    write_jsonl([sample.model_dump(mode="json") for sample in samples], split_dir / "test.jsonl")

    summary = build_eval_bench(split_dir=split_dir, split_name="test")
    assert summary["num_samples"] == 4
    assert Path(summary["eval_bench_path"]).exists()