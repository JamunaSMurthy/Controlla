"""ControlNet baseline wrapper."""

from __future__ import annotations

from pathlib import Path

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_controlnet_command(
    project_root: str,
    input_image: str | None = None,
    prompt: str | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return a command template for ControlNet execution.

    Paper alignment:
        ControlNet is used as an adapter-style conditioning baseline. It does
        not natively use Controlla's audio or graph geometry; audio controls
        should be converted to normalized text/emotion prompts before calling.
    """

    spec = get_baseline_spec("controlnet")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    entrypoint = spec.entrypoint_path(project_root)

    command = ["python", str(entrypoint)]
    maybe_extend(command, "--input_image", input_image)
    maybe_extend(command, "--prompt", prompt)
    maybe_extend(command, "--output_dir", output_dir)
    maybe_extend(command, "--config", config_path)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes=(
            "ControlNet baseline. Use same prompts/target emotions and held-out "
            "identities as Controlla; audio is converted to text/emotion prompt."
        ),
    )