"""Build normalized raw indexes for all enabled datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.download.verify_raw_datasets import verify_raw_datasets
from dataset_builder.preprocess.common import resolve_existing_path, write_normalized_outputs
from dataset_builder.preprocess.prepare_affectnet import build_records as build_affectnet_records
from dataset_builder.preprocess.prepare_celebahq import build_records as build_celebahq_records
from dataset_builder.preprocess.prepare_cremad import build_records as build_cremad_records
from dataset_builder.preprocess.prepare_emobank import build_records as build_emobank_records
from dataset_builder.preprocess.prepare_ffhq import build_records as build_ffhq_records
from dataset_builder.preprocess.prepare_iemocap import build_records as build_iemocap_records
from dataset_builder.preprocess.prepare_rafdb import build_records as build_rafdb_records
from dataset_builder.preprocess.prepare_ravdess import build_records as build_ravdess_records
from dataset_builder.preprocess.prepare_voxceleb import build_records as build_voxceleb_records
from dataset_builder.utils.io import write_json
from dataset_builder.utils.logging import get_logger


LOGGER = get_logger("dataset_builder.runners.build_raw_indexes")

PREPROCESSORS = {
    "affectnet": build_affectnet_records,
    "celebahq": build_celebahq_records,
    "cremad": build_cremad_records,
    "emobank": build_emobank_records,
    "ffhq": build_ffhq_records,
    "iemocap": build_iemocap_records,
    "rafdb": build_rafdb_records,
    "ravdess": build_ravdess_records,
    "voxceleb": build_voxceleb_records,
}


def build_raw_indexes(config_path: str | Path = "configs/datasets.yaml", datasets: set[str] | None = None) -> dict[str, object]:
    config = load_yaml_config(config_path)
    paths = load_builder_paths(config_path)
    verification = verify_raw_datasets(config_path)
    summary: dict[str, object] = {
        "verification_report": str(paths.reports_root / "raw_dataset_verification.json"),
        "datasets": {},
    }

    for dataset_name, dataset_settings in config["datasets"].items():
        if not dataset_settings.get("enabled", False):
            continue
        if datasets is not None and dataset_name not in datasets:
            continue
        if dataset_name not in PREPROCESSORS:
            LOGGER.warning("No Step 3 preprocessor registered for %s", dataset_name)
            continue

        raw_root = resolve_existing_path(paths.raw_datasets_root, str(dataset_settings["raw_root"]))
        if raw_root is None:
            LOGGER.warning("Skipping %s because raw root is missing", dataset_name)
            summary["datasets"][dataset_name] = {"status": "missing"}
            continue

        builder = PREPROCESSORS[dataset_name]
        records = builder(raw_root, dataset_settings, paths.raw_datasets_root)
        output_summary = write_normalized_outputs(dataset_name, records, paths.normalized_root)
        output_summary["verification_status"] = verification["datasets"][dataset_name]["status"]
        summary["datasets"][dataset_name] = output_summary
        LOGGER.info("Built %s normalized records for %s", output_summary["num_records"], dataset_name)

    write_json(summary, Path(paths.normalized_root) / "build_raw_indexes_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized raw indexes for enabled datasets")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    selected = set(args.datasets) if args.datasets else None
    build_raw_indexes(args.config, selected)


if __name__ == "__main__":
    main()