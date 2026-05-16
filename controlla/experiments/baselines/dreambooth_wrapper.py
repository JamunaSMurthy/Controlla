"""DreamBooth baseline wrapper."""

from __future__ import annotations

from .registry import get_baseline_spec


def build_dreambooth_command(project_root: str) -> list[str]:
    """Return a lightweight command template for DreamBooth execution."""
    spec = get_baseline_spec("dreambooth")
    return ["python", "-c", f"print('See {spec.repo_path(project_root) / spec.entrypoint} for DreamBooth training instructions')"]