"""Train or inspect baseline command wiring for Controlla experiments."""

from __future__ import annotations

import argparse
import shlex

from controlla.utils import load_config, resolve_experiment_manifest_path

from controlla.experiments.baselines.controlla_baseline import build_controlla_train_command
from controlla.experiments.baselines.controlnet_wrapper import build_controlnet_command
from controlla.experiments.baselines.diffusionclip_wrapper import build_diffusionclip_command
from controlla.experiments.baselines.dreambooth_wrapper import build_dreambooth_command
from controlla.experiments.baselines.instruct_pix2pix_wrapper import build_instructpix2pix_command
from controlla.experiments.baselines.registry import get_baseline_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve baseline training commands")
    parser.add_argument("--config", type=str, default="experiments/configs/train_controlla.yaml")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project_root = config["project_root"]
    method = args.method.lower()
    if method == "controlla_full":
        configured_split_dir = args.split_dir or config.get("split_dir")
        configured_split_name = args.split_name or config.get("split_name")
        dataset_path = args.dataset_path
        if dataset_path is None and configured_split_dir is None:
            dataset_path = str(resolve_experiment_manifest_path(config, default_split_name="train"))
        command = build_controlla_train_command(
            project_root,
            args.config,
            dataset_path=dataset_path,
            split_dir=configured_split_dir,
            split_name=configured_split_name,
        )
    elif method == "controlnet":
        command = build_controlnet_command(project_root)
    elif method == "dreambooth":
        command = build_dreambooth_command(project_root)
    elif method == "diffusionclip":
        command = build_diffusionclip_command(project_root)
    elif method == "instructpix2pix":
        command = build_instructpix2pix_command(project_root)
    else:
        spec = get_baseline_spec(method)
        command = ["python", "-c", f"print('No direct wrapper yet for {spec.name}; inspect {spec.repo_path(project_root)}')"]
    print(shlex.join(command))


if __name__ == "__main__":
    main()