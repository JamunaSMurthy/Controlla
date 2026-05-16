"""Export aligned manifests to Hugging Face DatasetDict or JSONL bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from dataset_builder.config import load_build_dataset_config
from dataset_builder.outputs.manifest_writer import flatten_sample, write_manifest
from dataset_builder.outputs.split_builder import build_splits
from dataset_builder.outputs.stats_report import build_stats_report
from dataset_builder.schemas import AlignedSample
from dataset_builder.utils.io import read_jsonl, write_json


def _load_aligned_samples(path: str | Path) -> list[AlignedSample]:
    return [AlignedSample.model_validate(record) for record in read_jsonl(path)]


def _mode_predicates(config) -> dict[str, Callable[[AlignedSample], bool]]:
    return {
        "full_multimodal": lambda sample: config.outputs.include_full_multimodal
        and sample.modality_mask.has_image
        and sample.modality_mask.has_audio
        and sample.modality_mask.has_text,
        "image_text": lambda sample: config.outputs.include_image_text
        and sample.modality_mask.has_image
        and sample.modality_mask.has_text,
        "image_audio": lambda sample: config.outputs.include_image_audio
        and sample.modality_mask.has_image
        and sample.modality_mask.has_audio,
        "text_audio_image_reference": lambda sample: config.outputs.include_text_audio_image_reference
        and sample.modality_mask.has_image
        and sample.modality_mask.has_audio
        and sample.modality_mask.has_text
        and sample.modality_mask.has_reference_image,
    }


def export_packaged_dataset(
    aligned_manifest_path: str | Path = "outputs/aligned/aligned_dataset.jsonl",
    build_config_path: str | Path = "configs/build_dataset.yaml",
) -> dict[str, object]:
    config = load_build_dataset_config(build_config_path)
    aligned_manifest_path = Path(aligned_manifest_path)
    samples = _load_aligned_samples(aligned_manifest_path)
    split_samples = build_splits(samples, config.split, seed=config.seed)

    package_root = aligned_manifest_path.parent / "package"
    package_root.mkdir(parents=True, exist_ok=True)

    all_outputs = write_manifest(split_samples, package_root, manifest_name="all_aligned")
    mode_outputs: dict[str, dict[str, str]] = {}
    for mode_name, predicate in _mode_predicates(config).items():
        mode_samples = [sample for sample in split_samples if predicate(sample)]
        mode_outputs[mode_name] = write_manifest(mode_samples, package_root, manifest_name=mode_name)

    stats_report = build_stats_report(split_samples, package_root)

    hf_export_summary = {
        "available": False,
        "saved_to": None,
        "reason": "datasets library not available",
    }
    try:
        from datasets import Dataset, DatasetDict  # type: ignore

        dataset_dict = DatasetDict(
            {
                split_name: Dataset.from_list([flatten_sample(sample) for sample in split_samples if sample.split == split_name])
                for split_name in ("train", "val", "test")
            }
        )
        hf_root = package_root / "hf_dataset"
        dataset_dict.save_to_disk(str(hf_root))
        hf_export_summary = {
            "available": True,
            "saved_to": str(hf_root),
            "reason": None,
        }
    except ImportError:
        pass

    summary = {
        "source_manifest": str(aligned_manifest_path),
        "package_root": str(package_root),
        "num_samples": len(split_samples),
        "all_outputs": all_outputs,
        "mode_outputs": mode_outputs,
        "stats_report": str(package_root / "stats_report.json"),
        "hf_export": hf_export_summary,
    }
    write_json(summary, package_root / "export_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export aligned manifests to packaged split bundles")
    parser.add_argument("--aligned-manifest", default="outputs/aligned/aligned_dataset.jsonl")
    parser.add_argument("--build-config", default="configs/build_dataset.yaml")
    args = parser.parse_args()
    export_packaged_dataset(args.aligned_manifest, args.build_config)


if __name__ == "__main__":
    main()