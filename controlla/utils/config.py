"""Configuration loading utilities for Controlla.

The config system keeps experiment setup lightweight and YAML-driven while still
supporting base-config inheritance.

Supported:
- YAML configs
- recursive base_config inheritance
- project_root inference
- relative path resolution against either config directory or project root
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _infer_project_root(config_path: Path) -> Path:
    """Infer repository root from the config file location."""

    resolved = config_path.resolve()

    for parent in [resolved.parent, *resolved.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent.resolve()

    # Fallback: if config is under configs/ or experiments/configs/,
    # walk upward conservatively.
    for parent in resolved.parents:
        if parent.name in {"controlla", "Controlla"}:
            return parent.parent.resolve()

    return resolved.parent.resolve()


def _resolve_candidate_path(
    config_path: Path,
    raw_path: str | Path,
    project_root: Path,
) -> Path:
    """Resolve a path from config directory first, then project root."""

    candidate = Path(raw_path)

    if candidate.is_absolute():
        return candidate.resolve()

    config_relative = (config_path.parent / candidate).resolve()
    if config_relative.exists():
        return config_relative

    project_relative = (project_root / candidate).resolve()
    if project_relative.exists():
        return project_relative

    # Return project-relative even if it does not exist. This is useful for
    # output paths and paths that will be created later.
    return project_relative


def merge_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge config dictionaries without mutating inputs."""

    merged = deepcopy(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    return payload


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config and resolve optional base_config inheritance."""

    config_path = Path(config_path).resolve()
    config = load_yaml_file(config_path)

    project_root = _infer_project_root(config_path)

    base_config_path = config.get("base_config")

    if base_config_path:
        resolved_base = _resolve_candidate_path(
            config_path=config_path,
            raw_path=base_config_path,
            project_root=project_root,
        )

        if not resolved_base.exists():
            raise FileNotFoundError(
                f"base_config not found: {base_config_path} resolved to {resolved_base}"
            )

        base_config = load_config(resolved_base)
        config = merge_dicts(
            base_config,
            {
                key: value
                for key, value in config.items()
                if key != "base_config"
            },
        )

    if config.get("project_root") is None:
        config["project_root"] = str(project_root)
    else:
        configured_root = Path(str(config["project_root"]))
        if configured_root.is_absolute():
            config["project_root"] = str(configured_root.resolve())
        else:
            config["project_root"] = str(
                (config_path.parent / configured_root).resolve()
            )

    return config