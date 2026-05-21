"""InstructPix2Pix baseline wrapper.

This is an optional extra baseline. It is useful for instruction-editing
comparisons but should not replace ICEdit/FLUX/ControlNet++ in paper tables.
"""

from __future__ import annotations

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_instructpix2pix_command(
    project_root: str,
    input_image: str | None = None,
    instruction: str | None = None,
    output_dir: str | None = None,
    checkpoint_path: str | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return a command template for InstructPix2Pix execution."""

    spec = get_baseline_spec("instructpix2pix")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.entrypoint_path(project_root))]
    maybe_extend(command, "--input", input_image)
    maybe_extend(command, "--edit", instruction)
    maybe_extend(command, "--outdir", output_dir)
    maybe_extend(command, "--ckpt", checkpoint_path)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="Optional InstructPix2Pix instruction editing baseline.",
    )