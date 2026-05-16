"""Configuration loading utilities for Controlla.

The config system keeps experiment setup lightweight and YAML-driven while still
supporting base-config inheritance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _infer_project_root(config_path: Path) -> Path:
    """Infer the repository root from the config file location."""
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent.resolve()
    return config_path.parent.parent.resolve()


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config dictionaries without mutating the inputs."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and resolve optional `base_config` inheritance."""
    config_path = Path(config_path)
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    base_config_path = config.get("base_config")
    if base_config_path:
        resolved_base = (config_path.parent / base_config_path).resolve()
        if not resolved_base.exists():
            resolved_base = (config_path.parent.parent / base_config_path).resolve()
        base_config = load_config(resolved_base)
        config = merge_dicts(base_config, {key: value for key, value in config.items() if key != "base_config"})
    project_root = config.get("project_root")
    if project_root is None:
        config["project_root"] = str(_infer_project_root(config_path))
    else:
        config["project_root"] = str((config_path.parent / project_root).resolve()) if not Path(project_root).is_absolute() else str(Path(project_root).resolve())
    return config