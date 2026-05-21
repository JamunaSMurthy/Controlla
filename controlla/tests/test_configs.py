"""Tests for public Controlla configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from controlla.utils.config import load_config


CONFIGS_THAT_SHOULD_LOAD = [
    "configs/base.yaml",
    "configs/train.yaml",
    "configs/infer.yaml",
    "configs/ablation_full.yaml",
    "configs/ablation_no_identity.yaml",
    "configs/ablation_no_ot.yaml",
    "configs/ablation_no_graph.yaml",
    "configs/ablation_no_audio.yaml",
    "configs/ablation_text_only.yaml",
    "configs/ablation_image_text.yaml",
    "configs/ablation_text_audio.yaml",
    "experiments/configs/datasets.yaml",
    "experiments/configs/eval_paper.yaml",
    "experiments/configs/eval_ablation.yaml",
    "experiments/configs/eval_sensitivity.yaml",
    "experiments/configs/train_controlla.yaml",
]


@pytest.mark.parametrize("config_path", CONFIGS_THAT_SHOULD_LOAD)
def test_config_loads(config_path: str) -> None:
    path = Path(config_path)

    if not path.exists():
        pytest.skip(f"Config does not exist in this checkout: {config_path}")

    config = load_config(path)

    assert isinstance(config, dict)
    assert len(config) > 0


def test_base_config_uses_mock_by_default() -> None:
    config_path = Path("configs/base.yaml")

    if not config_path.exists():
        pytest.skip("configs/base.yaml not found")

    config = load_config(config_path)

    assert config["model"]["diffusion_backbone"] == "mock"


def test_no_deprecated_eval_main_required() -> None:
    deprecated = Path("experiments/configs/eval_main.yaml")

    if deprecated.exists():
        pytest.skip("eval_main.yaml exists as a deprecated compatibility file")

    assert not deprecated.exists()