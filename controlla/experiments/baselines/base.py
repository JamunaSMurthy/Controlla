"""Shared baseline abstractions for Controlla experiments.

The paper evaluates Controlla against external baselines such as StyleCLIP,
ControlNet, ICEdit, FLUX.1 Kontext, DreamBooth+ControlNet++, SDXL, and
SDXL+ControlNet++. Most of these methods live in external repositories or
require separate checkpoints. This module defines a common command/spec layer
so experiment runners can invoke them consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineCommand:
    """Executable command for a baseline run.

    Attributes:
        name: Canonical baseline name.
        command: Shell command as a list of strings.
        cwd: Optional working directory.
        env: Optional environment variables.
        notes: Human-readable notes.
    """

    name: str
    command: list[str]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class BaselineSpec:
    """Description of an external baseline and its integration points."""

    name: str
    display_name: str
    repo_dir: str
    entrypoint: str
    paper_role: str
    notes: str = ""
    requires_training: bool = False
    requires_reference: bool = False
    supports_text: bool = True
    supports_audio: bool = False
    supports_identity: bool = False
    supports_image_condition: bool = False

    def repo_path(self, project_root: str | Path) -> Path:
        return (Path(project_root) / self.repo_dir).resolve()

    def entrypoint_path(self, project_root: str | Path) -> Path:
        return self.repo_path(project_root) / self.entrypoint


def maybe_extend(command: list[str], flag: str, value: Any | None) -> None:
    """Append a CLI flag-value pair when value is not None."""

    if value is not None:
        command.extend([flag, str(value)])


def ensure_repo_exists(spec: BaselineSpec, project_root: str | Path) -> None:
    """Raise a helpful error if an external baseline repo is missing."""

    repo_path = spec.repo_path(project_root)

    if not repo_path.exists():
        raise FileNotFoundError(
            f"Baseline repo for {spec.display_name} was not found at: {repo_path}\n"
            f"Install or clone the external baseline repo, or update repo_dir in registry.py."
        )