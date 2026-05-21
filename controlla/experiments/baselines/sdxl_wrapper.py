"""SDXL architecture-level baseline wrapper."""

from __future__ import annotations

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_sdxl_command(
    project_root: str,
    prompt: str | None = None,
    output_dir: str | None = None,
    model_path: str | None = None,
    num_inference_steps: int | None = None,
    seed: int | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return an SDXL baseline command.

    Paper alignment:
        SDXL is used for architecture-level comparison to test whether
        Controlla's gains are merely due to backbone choice.
    """

    spec = get_baseline_spec("sdxl")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.entrypoint_path(project_root))]
    maybe_extend(command, "--prompt", prompt)
    maybe_extend(command, "--output_dir", output_dir)
    maybe_extend(command, "--model_path", model_path)
    maybe_extend(command, "--num_inference_steps", num_inference_steps)
    maybe_extend(command, "--seed", seed)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="SDXL architecture-level baseline.",
    )