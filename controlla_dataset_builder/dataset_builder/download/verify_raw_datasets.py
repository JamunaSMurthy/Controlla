"""Verify that expected raw dataset folders exist before preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.utils.io import write_json
from dataset_builder.utils.logging import get_logger

from dataset_builder.preprocess.common import AUDIO_SUFFIXES, IMAGE_SUFFIXES, count_files, resolve_dataset_file, resolve_existing_path


LOGGER = get_logger("dataset_builder.download.verify_raw_datasets")


def verify_raw_datasets(config_path: str | Path = "configs/datasets.yaml") -> dict[str, object]:
    config = load_yaml_config(config_path)
    paths = load_builder_paths(config_path)
    report: dict[str, object] = {
        "raw_datasets_root": str(paths.raw_datasets_root),
        "datasets": {},
    }

    for dataset_name, dataset_settings in config["datasets"].items():
        raw_root_name = str(dataset_settings["raw_root"])
        resolved_root = resolve_existing_path(paths.raw_datasets_root, raw_root_name)
        expected_files = []
        missing_files = []
        for key, value in dataset_settings.items():
            if key.endswith("_file"):
                expected_path = resolve_dataset_file(paths.raw_datasets_root, str(value))
                expected_files.append(str(expected_path))
                if not expected_path.exists():
                    missing_files.append(str(expected_path))

        if resolved_root is None:
            dataset_report = {
                "enabled": bool(dataset_settings.get("enabled", False)),
                "status": "missing",
                "configured_raw_root": raw_root_name,
                "missing_expected_files": missing_files,
            }
        else:
            image_count = count_files(resolved_root, IMAGE_SUFFIXES)
            audio_count = count_files(resolved_root, AUDIO_SUFFIXES)
            media_count = image_count + audio_count
            if media_count == 0 and expected_files:
                status = "metadata_only"
            elif missing_files:
                status = "incomplete"
            else:
                status = "ready"
            dataset_report = {
                "enabled": bool(dataset_settings.get("enabled", False)),
                "status": status,
                "configured_raw_root": raw_root_name,
                "resolved_raw_root": str(resolved_root),
                "expected_files": expected_files,
                "missing_expected_files": missing_files,
                "image_count": image_count,
                "audio_count": audio_count,
            }
        report["datasets"][dataset_name] = dataset_report

    write_json(report, paths.reports_root / "raw_dataset_verification.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify raw dataset availability")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    report = verify_raw_datasets(args.config)
    ready = sum(1 for item in report["datasets"].values() if item["status"] == "ready")
    LOGGER.info("Verified raw datasets: %s ready", ready)


if __name__ == "__main__":
    main()