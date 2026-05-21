"""DiffusionCLIP baseline wrapper.

This is an optional extra baseline. It is not one of the main Controlla paper
baselines unless explicitly reported in additional experiments.
"""

from __future__ import annotations

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_diffusionclip_command(
    project_root: str,
    input_image: str | None = None,
    text_prompt: str | None = None,
    output_dir: str | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return a command template for DiffusionCLIP execution."""

    spec = get_baseline_spec("diffusionclip")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.entrypoint_path(project_root))]
    maybe_extend(command, "--input", input_image)
    maybe_extend(command, "--text", text_prompt)
    maybe_extend(command, "--output_dir", output_dir)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="Optional DiffusionCLIP text-driven editing baseline.",
    )