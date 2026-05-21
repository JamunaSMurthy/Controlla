"""Tests for baseline registry and command wrappers."""

from __future__ import annotations

from pathlib import Path

from controlla.experiments.baselines import (
    PAPER_BASELINES,
    build_controlla_baseline_command,
    build_controlnet_command,
    get_baseline_spec,
    list_baselines,
)


def test_paper_baselines_registered() -> None:
    expected = {
        "styleclip",
        "controlnet",
        "icedit",
        "flux_kontext",
        "dreambooth_controlnetpp",
        "sdxl",
        "sdxl_controlnetpp",
        "clip_retrieval",
        "imagebind_retrieval",
        "controlla_full",
    }

    assert expected.issubset(set(PAPER_BASELINES))
    assert expected.issubset(set(list_baselines()))


def test_get_baseline_spec() -> None:
    spec = get_baseline_spec("controlla_full")

    assert spec.display_name == "Controlla"
    assert spec.supports_text is True
    assert spec.supports_identity is True


def test_controlla_command_builds(tmp_path: Path) -> None:
    command = build_controlla_baseline_command(
        project_root=str(tmp_path),
        config_path="configs/train.yaml",
        mode="train",
        dataset_path="data.csv",
    )

    assert command.name == "controlla_full"
    assert "train_controlla.py" in " ".join(command.command)


def test_controlnet_command_builds_without_repo_check(tmp_path: Path) -> None:
    command = build_controlnet_command(
        project_root=str(tmp_path),
        input_image="input.png",
        prompt="a happy face",
        output_dir="outputs",
        check_repo=False,
    )

    assert command.name == "controlnet"
    assert "--prompt" in command.command