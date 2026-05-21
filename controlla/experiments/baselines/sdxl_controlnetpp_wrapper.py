"""SDXL + ControlNet++ architecture-level baseline wrapper."""

from __future__ import annotations

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_sdxl_controlnetpp_command(
    project_root: str,
    input_image: str | None = None,
    condition_image: str | None = None,
    prompt: str | None = None,
    output_dir: str | None = None,
    sdxl_model_path: str | None = None,
    controlnetpp_model_path: str | None = None,
    num_inference_steps: int | None = None,
    seed: int | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return an SDXL + ControlNet++ baseline command.

    Paper alignment:
        This is the architecture-level control baseline used against Controlla
        to show that the graph-geometry gains are not only due to using a
        stronger diffusion backbone.
    """

    spec = get_baseline_spec("sdxl_controlnetpp")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.entrypoint_path(project_root))]
    maybe_extend(command, "--input_image", input_image)
    maybe_extend(command, "--condition_image", condition_image)
    maybe_extend(command, "--prompt", prompt)
    maybe_extend(command, "--output_dir", output_dir)
    maybe_extend(command, "--sdxl_model_path", sdxl_model_path)
    maybe_extend(command, "--controlnetpp_model_path", controlnetpp_model_path)
    maybe_extend(command, "--num_inference_steps", num_inference_steps)
    maybe_extend(command, "--seed", seed)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="SDXL + ControlNet++ architecture-level control baseline.",
    )