"""InstructPix2Pix baseline wrapper."""

from __future__ import annotations

from .registry import get_baseline_spec


def build_instructpix2pix_command(project_root: str) -> list[str]:
    """Return a lightweight command template for InstructPix2Pix execution."""
    spec = get_baseline_spec("instructpix2pix")
    return ["python", str(spec.repo_path(project_root) / spec.entrypoint)]