"""ICEdit baseline wrapper."""

from __future__ import annotations

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_icedit_command(
    project_root: str,
    input_image: str | None = None,
    reference_image: str | None = None,
    prompt: str | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return a command template for ICEdit.

    Paper alignment:
        ICEdit is reported as a strong image-editing baseline. It should receive
        the same target affect prompt and reference/identity information when
        supported by the external implementation.
    """

    spec = get_baseline_spec("icedit")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.entrypoint_path(project_root))]
    maybe_extend(command, "--input_image", input_image)
    maybe_extend(command, "--reference_image", reference_image)
    maybe_extend(command, "--prompt", prompt)
    maybe_extend(command, "--output_dir", output_dir)
    maybe_extend(command, "--config", config_path)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="ICEdit image editing baseline.",
    )