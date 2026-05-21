"""Resolve baseline training/inference commands for Controlla experiments.

This script prints a command for a requested baseline. It does not execute the
command unless --run is provided.

External baselines require their repositories/checkpoints to exist at the paths
declared in experiments/baselines/registry.py.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from controlla.experiments.baselines.controlla_baseline import build_controlla_baseline_command
from controlla.experiments.baselines.controlnet_wrapper import build_controlnet_command
from controlla.experiments.baselines.controlnetpp_wrapper import build_controlnetpp_command
from controlla.experiments.baselines.diffusionclip_wrapper import build_diffusionclip_command
from controlla.experiments.baselines.dreambooth_controlnetpp_wrapper import (
    build_dreambooth_controlnetpp_command,
)
from controlla.experiments.baselines.dreambooth_wrapper import (
    build_dreambooth_infer_command,
    build_dreambooth_train_command,
)
from controlla.experiments.baselines.flux_kontext_wrapper import build_flux_kontext_command
from controlla.experiments.baselines.icedit_wrapper import build_icedit_command
from controlla.experiments.baselines.instruct_pix2pix_wrapper import build_instructpix2pix_command
from controlla.experiments.baselines.registry import get_baseline_spec, list_baselines
from controlla.experiments.baselines.sdxl_controlnetpp_wrapper import build_sdxl_controlnetpp_command
from controlla.experiments.baselines.sdxl_wrapper import build_sdxl_command
from controlla.experiments.baselines.styleclip_wrapper import build_styleclip_command
from controlla.utils import load_config, resolve_experiment_manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve baseline training/inference commands")
    parser.add_argument("--config", type=str, default="experiments/configs/train_controlla.yaml")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--mode", type=str, default="infer", choices=["train", "infer", "eval"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--input-image", type=str, default=None)
    parser.add_argument("--reference-image", type=str, default=None)
    parser.add_argument("--condition-image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--check-repo", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def _build_command(args: argparse.Namespace):
    config = load_config(args.config)
    project_root = str(Path(config["project_root"]).resolve())
    method = args.method.strip().lower()

    if method == "controlla":
        method = "controlla_full"

    if method == "controlla_full":
        dataset_path = args.dataset_path
        configured_split_dir = args.split_dir or config.get("split_dir")
        configured_split_name = args.split_name or config.get("split_name")

        if dataset_path is None and configured_split_dir is None and args.mode == "train":
            dataset_path = str(resolve_experiment_manifest_path(config, default_split_name="train"))

        return build_controlla_baseline_command(
            project_root=project_root,
            config_path=args.config,
            mode="train" if args.mode == "train" else "eval",
            dataset_path=dataset_path,
            split_dir=configured_split_dir,
            split_name=configured_split_name,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
        )

    if method == "styleclip":
        return build_styleclip_command(
            project_root=project_root,
            input_image=args.input_image,
            text_prompt=args.prompt,
            output_dir=args.output_dir,
            check_repo=args.check_repo,
        )

    if method == "controlnet":
        return build_controlnet_command(
            project_root=project_root,
            input_image=args.input_image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            config_path=args.config,
            check_repo=args.check_repo,
        )

    if method == "controlnetpp":
        return build_controlnetpp_command(
            project_root=project_root,
            input_image=args.input_image,
            condition_image=args.condition_image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            config_path=args.config,
            check_repo=args.check_repo,
        )

    if method == "icedit":
        return build_icedit_command(
            project_root=project_root,
            input_image=args.input_image,
            reference_image=args.reference_image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            config_path=args.config,
            check_repo=args.check_repo,
        )

    if method == "flux_kontext":
        return build_flux_kontext_command(
            project_root=project_root,
            input_image=args.input_image,
            reference_image=args.reference_image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            model_path=args.model_path,
            check_repo=args.check_repo,
        )

    if method == "dreambooth":
        if args.mode == "train":
            return build_dreambooth_train_command(
                project_root=project_root,
                instance_data_dir=args.dataset_path,
                output_dir=args.output_dir,
                instance_prompt=args.prompt,
                pretrained_model_name_or_path=args.model_path,
                check_repo=args.check_repo,
            )

        return build_dreambooth_infer_command(
            project_root=project_root,
            model_dir=args.model_path,
            prompt=args.prompt,
            output_dir=args.output_dir,
            check_repo=args.check_repo,
        )

    if method == "dreambooth_controlnetpp":
        return build_dreambooth_controlnetpp_command(
            project_root=project_root,
            dreambooth_model_dir=args.model_path,
            input_image=args.input_image,
            condition_image=args.condition_image,
            reference_image=args.reference_image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            config_path=args.config,
            check_repo=args.check_repo,
        )

    if method == "sdxl":
        return build_sdxl_command(
            project_root=project_root,
            prompt=args.prompt,
            output_dir=args.output_dir,
            model_path=args.model_path,
            check_repo=args.check_repo,
        )

    if method == "sdxl_controlnetpp":
        return build_sdxl_controlnetpp_command(
            project_root=project_root,
            input_image=args.input_image,
            condition_image=args.condition_image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            sdxl_model_path=args.model_path,
            controlnetpp_model_path=args.checkpoint_path,
            check_repo=args.check_repo,
        )

    if method == "diffusionclip":
        return build_diffusionclip_command(
            project_root=project_root,
            input_image=args.input_image,
            text_prompt=args.prompt,
            output_dir=args.output_dir,
            check_repo=args.check_repo,
        )

    if method == "instructpix2pix":
        return build_instructpix2pix_command(
            project_root=project_root,
            input_image=args.input_image,
            instruction=args.prompt,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
            check_repo=args.check_repo,
        )

    spec = get_baseline_spec(method)
    raise NotImplementedError(
        f"No direct command wrapper for {spec.display_name}. "
        f"External repo expected at {spec.repo_dir}, entrypoint {spec.entrypoint}."
    )


def main() -> None:
    args = parse_args()

    if args.list:
        for name in list_baselines(paper_only=False):
            print(name)
        return

    command = _build_command(args)
    print(shlex.join(command.command))

    if command.notes:
        print(f"# {command.notes}")

    if args.run:
        subprocess.run(command.command, cwd=command.cwd, env=command.env or None, check=True)


if __name__ == "__main__":
    main()