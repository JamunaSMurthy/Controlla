"""Prepare FFHQ normalized index."""

from __future__ import annotations

import argparse

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.utils.logging import get_logger

from .common import build_image_only_records


LOGGER = get_logger("dataset_builder.preprocess.ffhq")


def build_records(raw_root, dataset_settings, raw_datasets_root):
    del dataset_settings, raw_datasets_root
    return build_image_only_records("ffhq", raw_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FFHQ normalized index")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    paths = load_builder_paths(args.config)
    config = load_yaml_config(args.config)
    dataset_settings = config["datasets"]["ffhq"]
    records = build_records(paths.raw_datasets_root / str(dataset_settings["raw_root"]), dataset_settings, paths.raw_datasets_root)
    LOGGER.info("Prepared %s FFHQ records", len(records))


if __name__ == "__main__":
    main()