"""DiffusionCLIP baseline wrapper."""

from __future__ import annotations

from .registry import get_baseline_spec


def build_diffusionclip_command(project_root: str) -> list[str]:
    """Return a lightweight command template for DiffusionCLIP execution."""
    spec = get_baseline_spec("diffusionclip")
    return ["python", str(spec.repo_path(project_root) / spec.entrypoint)]