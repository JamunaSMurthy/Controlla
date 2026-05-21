"""DreamBooth baseline wrapper."""

from __future__ import annotations

from .base import BaselineCommand, ensure_repo_exists, maybe_extend
from .registry import get_baseline_spec


def build_dreambooth_train_command(
    project_root: str,
    instance_data_dir: str | None = None,
    output_dir: str | None = None,
    instance_prompt: str | None = None,
    pretrained_model_name_or_path: str | None = None,
    max_train_steps: int | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return a DreamBooth training command template.

    Paper alignment:
        DreamBooth is an identity personalization baseline. In the paper tables,
        the stronger reported baseline is DreamBooth + ControlNet++, but keeping
        the standalone wrapper is useful for ablations.
    """

    spec = get_baseline_spec("dreambooth")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.entrypoint_path(project_root))]
    maybe_extend(command, "--instance_data_dir", instance_data_dir)
    maybe_extend(command, "--output_dir", output_dir)
    maybe_extend(command, "--instance_prompt", instance_prompt)
    maybe_extend(command, "--pretrained_model_name_or_path", pretrained_model_name_or_path)
    maybe_extend(command, "--max_train_steps", max_train_steps)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="DreamBooth identity personalization baseline.",
    )


def build_dreambooth_infer_command(
    project_root: str,
    model_dir: str | None = None,
    prompt: str | None = None,
    output_dir: str | None = None,
    check_repo: bool = False,
) -> BaselineCommand:
    """Return a DreamBooth inference command template."""

    spec = get_baseline_spec("dreambooth")

    if check_repo:
        ensure_repo_exists(spec, project_root)

    command = ["python", str(spec.repo_path(project_root) / "infer.py")]
    maybe_extend(command, "--model_dir", model_dir)
    maybe_extend(command, "--prompt", prompt)
    maybe_extend(command, "--output_dir", output_dir)

    return BaselineCommand(
        name=spec.name,
        command=command,
        cwd=spec.repo_path(project_root),
        notes="DreamBooth inference after identity personalization.",
    )