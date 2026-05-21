"""Controlla proposed-method wrapper."""

from __future__ import annotations

from pathlib import Path

from .base import BaselineCommand, maybe_extend
from .registry import get_baseline_spec


def build_controlla_train_command(
    project_root: str,
    config_path: str,
    dataset_path: str | None = None,
    split_dir: str | None = None,
    split_name: str | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """Return the Controlla training command for a prepared manifest."""

    command = [
        "python",
        str((Path(project_root) / "pipelines/train_controlla.py").resolve()),
        "--config",
        config_path,
    ]

    maybe_extend(command, "--dataset-path", dataset_path)
    maybe_extend(command, "--split-dir", split_dir)
    maybe_extend(command, "--split-name", split_name)
    maybe_extend(command, "--output-dir", output_dir)

    return command


def build_controlla_eval_command(
    project_root: str,
    config_path: str,
    checkpoint_path: str | None = None,
    dataset_path: str | None = None,
    split_name: str | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """Return the Controlla evaluation command."""

    command = [
        "python",
        str((Path(project_root) / "experiments/runners/run_main_table.py").resolve()),
        "--config",
        config_path,
        "--baseline",
        "controlla_full",
    ]

    maybe_extend(command, "--checkpoint", checkpoint_path)
    maybe_extend(command, "--dataset-path", dataset_path)
    maybe_extend(command, "--split-name", split_name)
    maybe_extend(command, "--output-dir", output_dir)

    return command


def build_controlla_baseline_command(
    project_root: str,
    config_path: str,
    mode: str = "train",
    **kwargs: str | None,
) -> BaselineCommand:
    """Build a standardized Controlla baseline command."""

    spec = get_baseline_spec("controlla_full")

    if mode == "train":
        command = build_controlla_train_command(
            project_root=project_root,
            config_path=config_path,
            dataset_path=kwargs.get("dataset_path"),
            split_dir=kwargs.get("split_dir"),
            split_name=kwargs.get("split_name"),
            output_dir=kwargs.get("output_dir"),
        )
    elif mode == "eval":
        command = build_controlla_eval_command(
            project_root=project_root,
            config_path=config_path,
            checkpoint_path=kwargs.get("checkpoint_path"),
            dataset_path=kwargs.get("dataset_path"),
            split_name=kwargs.get("split_name"),
            output_dir=kwargs.get("output_dir"),
        )
    else:
        raise ValueError(f"Unsupported Controlla mode: {mode}")

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=Path(project_root).resolve(),
        notes=spec.notes,
    )