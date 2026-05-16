"""Build the high-confidence Controlla evaluation bench subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder.config import load_build_dataset_config
from dataset_builder.outputs.manifest_writer import flatten_sample
from dataset_builder.outputs.split_builder import build_splits
from dataset_builder.schemas import AlignedSample
from dataset_builder.utils.io import read_jsonl, write_json, write_jsonl


def select_eval_bench(samples: list[AlignedSample], *, size_per_class: int) -> list[AlignedSample]:
    filtered = [
        sample
        for sample in samples
        if sample.split == "test"
        and sample.modality_mask.has_image
        and sample.modality_mask.has_audio
        and sample.modality_mask.has_text
        and sample.modality_mask.has_reference_image
    ]
    selected: list[AlignedSample] = []
    for emotion in sorted({sample.unified_emotion for sample in filtered}):
        emotion_samples = sorted(
            [sample for sample in filtered if sample.unified_emotion == emotion],
            key=lambda sample: (sample.alignment_scores.final_score, sample.sample_id),
            reverse=True,
        )
        selected.extend(emotion_samples[:size_per_class])
    return selected


def build_eval_bench(
    aligned_manifest_path: str | Path | None = None,
    build_config_path: str | Path | None = "configs/build_dataset.yaml",
    *,
    split_dir: str | Path = "outputs/strict_splits",
    split_name: str = "test",
) -> dict[str, object]:
    source_manifest_path = Path(aligned_manifest_path) if aligned_manifest_path is not None else Path(split_dir) / f"{split_name}.jsonl"
    samples = [AlignedSample.model_validate(record) for record in read_jsonl(source_manifest_path)]

    size_per_class = 64
    split_samples = samples
    if build_config_path is not None:
        config = load_build_dataset_config(build_config_path)
        size_per_class = config.outputs.eval_bench_size_per_class
        if aligned_manifest_path is not None:
            split_samples = build_splits(samples, config.split, seed=config.seed)
    eval_samples = select_eval_bench(split_samples, size_per_class=size_per_class)

    output_root = source_manifest_path.parent
    eval_jsonl_path = write_jsonl([sample.model_dump(mode="json") for sample in eval_samples], output_root / "eval_bench.jsonl")
    write_jsonl([flatten_sample(sample) for sample in eval_samples], output_root / "eval_bench_flat.jsonl")

    emotion_counts: dict[str, int] = {}
    for sample in eval_samples:
        emotion_counts[sample.unified_emotion] = emotion_counts.get(sample.unified_emotion, 0) + 1

    summary = {
        "source_manifest": str(source_manifest_path),
        "eval_bench_path": str(eval_jsonl_path),
        "num_samples": len(eval_samples),
        "size_per_class": size_per_class,
        "emotion_counts": emotion_counts,
    }
    write_json(summary, output_root / "eval_bench_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the high-confidence Controlla eval bench")
    parser.add_argument("--aligned-manifest", default=None)
    parser.add_argument("--split-dir", default="outputs/strict_splits")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--build-config", default="configs/build_dataset.yaml")
    args = parser.parse_args()
    build_eval_bench(
        args.aligned_manifest,
        args.build_config,
        split_dir=args.split_dir,
        split_name=args.split_name,
    )


if __name__ == "__main__":
    main()