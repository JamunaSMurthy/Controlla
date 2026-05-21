"""StyleCLIP baseline wrapper."""

from __future__ import annotations

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_styleclip_command(
    project_root: str,
    input_image: str | None = None,
    text_prompt: str | None = None,
    output_dir: str | None = None,
    latent_path: str | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return a command template for StyleCLIP.

    Paper alignment:
        StyleCLIP is a latent editing baseline. It is useful for testing
        endpoint semantic/emotion editing but does not model graph traversal,
        multimodal audio control, or identity/attribute factorization.
    """

    spec = get_baseline_spec("styleclip")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.entrypoint_path(project_root))]
    maybe_extend(command, "--input", input_image)
    maybe_extend(command, "--text", text_prompt)
    maybe_extend(command, "--latent", latent_path)
    maybe_extend(command, "--output_dir", output_dir)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="StyleCLIP latent editing baseline.",
    )