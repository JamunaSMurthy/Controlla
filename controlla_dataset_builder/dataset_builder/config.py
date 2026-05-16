"""Configuration helpers for the dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schemas import SplitRatios


def _infer_project_root(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent.resolve()
    return config_path.parent.resolve()


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and resolve relative project roots."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    declared_root = config.get("project_root")
    if declared_root is None:
        config["project_root"] = str(_infer_project_root(path))
    else:
        declared = Path(declared_root)
        config["project_root"] = str(declared.resolve() if declared.is_absolute() else (path.parent.parent / declared).resolve())
    return config


@dataclass(frozen=True)
class BuilderPaths:
    """Resolved filesystem paths used by the builder."""

    project_root: Path
    raw_datasets_root: Path
    normalized_root: Path
    feature_cache_root: Path
    aligned_output_root: Path
    reports_root: Path


def load_builder_paths(config_path: str | Path = "configs/datasets.yaml") -> BuilderPaths:
    """Resolve common filesystem roots from the datasets config."""
    config = load_yaml_config(config_path)
    project_root = Path(config["project_root"])

    def resolve(value: str) -> Path:
        target = Path(value)
        return target.resolve() if target.is_absolute() else (project_root / target).resolve()

    return BuilderPaths(
        project_root=project_root,
        raw_datasets_root=resolve(config["raw_datasets_root"]),
        normalized_root=resolve(config["normalized_root"]),
        feature_cache_root=resolve(config["feature_cache_root"]),
        aligned_output_root=resolve(config["aligned_output_root"]),
        reports_root=resolve(config["reports_root"]),
    )


class BuildRequirements(BaseModel):
    """Dataset modality requirements from build config."""

    model_config = ConfigDict(extra="ignore")

    require_audio: bool = False
    require_text: bool = True
    require_reference_image: bool = False


class AlignmentSettings(BaseModel):
    """Alignment settings from build config."""

    model_config = ConfigDict(extra="ignore")

    minimum_score: float = 0.55
    top_k_audio_candidates: int = 16
    top_k_text_candidates: int = 16
    top_k_reference_candidates: int = 8
    max_samples_per_class: int | None = None


class OutputSettings(BaseModel):
    """Output packaging toggles from build config."""

    model_config = ConfigDict(extra="ignore")

    include_full_multimodal: bool = True
    include_image_text: bool = True
    include_image_audio: bool = True
    include_text_audio_image_reference: bool = True
    eval_bench_size_per_class: int = 64


class BuildDatasetConfig(BaseModel):
    """Typed view over the build_dataset.yaml file."""

    model_config = ConfigDict(extra="allow")

    project_root: str
    seed: int = 0
    output_root: str = "outputs/aligned"
    split: SplitRatios = Field(default_factory=SplitRatios)
    requirements: BuildRequirements = Field(default_factory=BuildRequirements)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    outputs: OutputSettings = Field(default_factory=OutputSettings)


def load_build_dataset_config(config_path: str | Path = "configs/build_dataset.yaml") -> BuildDatasetConfig:
    """Load the typed build-dataset config."""
    return BuildDatasetConfig.model_validate(load_yaml_config(config_path))