"""Baseline wrappers for Controlla experiments."""

from .base import BaselineCommand, BaselineSpec
from .controlla_baseline import (
    build_controlla_baseline_command,
    build_controlla_eval_command,
    build_controlla_train_command,
)
from .controlnet_wrapper import build_controlnet_command
from .controlnetpp_wrapper import build_controlnetpp_command
from .diffusionclip_wrapper import build_diffusionclip_command
from .dreambooth_controlnetpp_wrapper import build_dreambooth_controlnetpp_command
from .dreambooth_wrapper import build_dreambooth_infer_command, build_dreambooth_train_command
from .flux_kontext_wrapper import build_flux_kontext_command
from .icedit_wrapper import build_icedit_command
from .instruct_pix2pix_wrapper import build_instructpix2pix_command
from .registry import BASELINE_REGISTRY, PAPER_BASELINES, get_baseline_spec, list_baselines
from .retrieval_baselines import (
    clip_i2t_retrieval,
    imagebind_i2a_retrieval,
    imagebind_i2t_retrieval,
    run_retrieval_baselines,
)
from .sdxl_controlnetpp_wrapper import build_sdxl_controlnetpp_command
from .sdxl_wrapper import build_sdxl_command
from .styleclip_wrapper import build_styleclip_command

__all__ = [
    "BASELINE_REGISTRY",
    "PAPER_BASELINES",
    "BaselineCommand",
    "BaselineSpec",
    "build_controlla_baseline_command",
    "build_controlla_eval_command",
    "build_controlla_train_command",
    "build_controlnet_command",
    "build_controlnetpp_command",
    "build_diffusionclip_command",
    "build_dreambooth_controlnetpp_command",
    "build_dreambooth_infer_command",
    "build_dreambooth_train_command",
    "build_flux_kontext_command",
    "build_icedit_command",
    "build_instructpix2pix_command",
    "build_sdxl_command",
    "build_sdxl_controlnetpp_command",
    "build_styleclip_command",
    "clip_i2t_retrieval",
    "get_baseline_spec",
    "imagebind_i2a_retrieval",
    "imagebind_i2t_retrieval",
    "list_baselines",
    "run_retrieval_baselines",
]