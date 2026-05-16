"""Filesystem path helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()