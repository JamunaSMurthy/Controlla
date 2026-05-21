"""FLUX.1 Kontext baseline wrapper."""

from __future__ import annotations

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_flux_kontext_command(
    project_root: str,
    input_image: str | None = None,
    reference_image: str | None = None,
    prompt: str | None = None,
    output_dir: str | None = None,
    model_path: str | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return a command template for FLUX.1 Kontext.

    Paper alignment:
        FLUX.1 Kontext is a modern image-editing / instruction-following
        baseline. It tests whether Controlla's gains are due to graph geometry
        rather than only stronger generic image editing.
    """

    spec = get_baseline_spec("flux_kontext")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.entrypoint_path(project_root))]
    maybe_extend(command, "--input_image", input_image)
    maybe_extend(command, "--reference_image", reference_image)
    maybe_extend(command, "--prompt", prompt)
    maybe_extend(command, "--model_path", model_path)
    maybe_extend(command, "--output_dir", output_dir)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="FLUX.1 Kontext modern editing baseline.",
    )