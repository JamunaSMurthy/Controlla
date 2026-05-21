"""Experiment reproducibility helpers for Controlla."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(
    payload: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a JSON file with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return payload


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a stable hash for an experiment configuration."""

    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _safe_git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return result.stdout.strip() or None


def _safe_git_dirty(cwd: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return bool(result.stdout.strip())


def collect_environment_info(project_root: str | Path) -> dict[str, Any]:
    """Collect environment metadata for experiment reproducibility."""

    project_root = Path(project_root).resolve()

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cwd": str(project_root),
        "git_commit": _safe_git_commit(project_root),
        "git_dirty": _safe_git_dirty(project_root),
        "env": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key
            in {
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "CONTROLLA_USE_LOCAL_CHECKPOINTS",
            }
        },
    }