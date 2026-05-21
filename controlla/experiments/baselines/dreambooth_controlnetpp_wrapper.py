"""DreamBooth + ControlNet++ baseline wrapper."""

from __future__ import annotations

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_dreambooth_controlnetpp_command(
    project_root: str,
    dreambooth_model_dir: str | None = None,
    input_image: str | None = None,
    condition_image: str | None = None,
    reference_image: str | None = None,
    prompt: str | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return a DreamBooth + ControlNet++ command template.

    Paper alignment:
        This is the strongest identity-personalization plus control baseline
        reported in the main paper. It should receive the same identity/reference
        and target emotion prompt as Controlla whenever supported.
    """

    spec = get_baseline_spec("dreambooth_controlnetpp")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.entrypoint_path(project_root))]
    maybe_extend(command, "--dreambooth_model_dir", dreambooth_model_dir)
    maybe_extend(command, "--input_image", input_image)
    maybe_extend(command, "--condition_image", condition_image)
    maybe_extend(command, "--reference_image", reference_image)
    maybe_extend(command, "--prompt", prompt)
    maybe_extend(command, "--output_dir", output_dir)
    maybe_extend(command, "--config", config_path)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="DreamBooth + ControlNet++ identity/control baseline.",
    )