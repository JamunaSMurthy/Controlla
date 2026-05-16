"""Build harmonized text manifests from normalized raw indexes."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder.config import load_builder_paths
from dataset_builder.preprocess.text_sources import (
    default_text_output_root,
    load_normalized_records,
    load_text_generation_settings,
    build_text_manifest_records,
    write_text_outputs,
)
from dataset_builder.utils.io import write_json, write_jsonl
from dataset_builder.utils.logging import get_logger


LOGGER = get_logger("dataset_builder.runners.build_text_sources")


def build_text_sources(config_path: str | Path = "configs/text_generation.yaml", datasets: set[str] | None = None) -> dict[str, object]:
    text_config = load_text_generation_settings(config_path)
    dataset_paths = load_builder_paths("configs/datasets.yaml")
    output_root = default_text_output_root(text_config, Path(text_config.project_root))
    all_records: list[dict[str, object]] = []
    summary: dict[str, object] = {"datasets": {}, "output_root": str(output_root)}

    for dataset_dir in sorted(path for path in dataset_paths.normalized_root.iterdir() if path.is_dir()):
        dataset_name = dataset_dir.name
        if datasets is not None and dataset_name not in datasets:
            continue
        normalized_csv = dataset_dir / f"{dataset_name}_normalized.csv"
        if not normalized_csv.exists():
            continue
        normalized_records = load_normalized_records(normalized_csv)
        text_records = build_text_manifest_records(normalized_records, text_config)
        dataset_summary = write_text_outputs(dataset_name, text_records, output_root)
        summary["datasets"][dataset_name] = dataset_summary
        all_records.extend(text_records)
        LOGGER.info("Built %s text records for %s", dataset_summary["num_records"], dataset_name)

    write_jsonl(all_records, output_root / "all_text_sources.jsonl")
    write_json(summary, output_root / "build_text_sources_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build harmonized text manifests from normalized indexes")
    parser.add_argument("--config", default="configs/text_generation.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    selected = set(args.datasets) if args.datasets else None
    build_text_sources(args.config, selected)


if __name__ == "__main__":
    main()