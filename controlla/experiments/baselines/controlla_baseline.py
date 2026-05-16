"""Controlla baseline wrapper."""

from __future__ import annotations

from pathlib import Path


def build_controlla_train_command(
    project_root: str,
    config_path: str,
    dataset_path: str | None = None,
    split_dir: str | None = None,
    split_name: str | None = None,
) -> list[str]:
    """Return the Controlla training command for a prepared manifest."""
    command = [
        "python",
        str((Path(project_root) / "pipelines/train_controlla.py").resolve()),
        "--config",
        config_path,
    ]
    if dataset_path is not None:
        command.extend(["--dataset-path", dataset_path])
    if split_dir is not None:
        command.extend(["--split-dir", split_dir])
    if split_name is not None:
        command.extend(["--split-name", split_name])
    return command