"""Baseline registry for Controlla experiment runners.

This registry is paper-aligned. It includes the baselines reported in the
Controlla manuscript/tables:

- StyleCLIP
- ControlNet
- ICEdit
- FLUX.1 Kontext
- DreamBooth + ControlNet++
- SDXL
- SDXL + ControlNet++
- CLIP retrieval
- ImageBind retrieval
- Controlla full model

Additional legacy baselines are kept as optional extras.
"""

from __future__ import annotations

from .base import BaselineSpec


BASELINE_REGISTRY: dict[str, BaselineSpec] = {
    "controlla_full": BaselineSpec(
        name="controlla_full",
        display_name="Controlla",
        repo_dir=".",
        entrypoint="pipelines/train_controlla.py",
        paper_role="proposed_method",
        notes="Proposed full Controlla model.",
        requires_training=True,
        requires_reference=True,
        supports_text=True,
        supports_audio=True,
        supports_identity=True,
        supports_image_condition=True,
    ),

    "styleclip": BaselineSpec(
        name="styleclip",
        display_name="StyleCLIP",
        repo_dir="../Controlla_Benchmarks/StyleCLIP",
        entrypoint="cog_predict.py",
        paper_role="latent_editing_baseline",
        notes="Latent editing baseline used for semantic/attribute editing comparison.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=False,
        supports_identity=False,
        supports_image_condition=True,
    ),

    "controlnet": BaselineSpec(
        name="controlnet",
        display_name="ControlNet",
        repo_dir="../Controlla_Benchmarks/ControlNet",
        entrypoint="gradio_canny2image.py",
        paper_role="adapter_conditioning_baseline",
        notes="Adapter-style conditioning baseline.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=False,
        supports_identity=False,
        supports_image_condition=True,
    ),

    "controlnetpp": BaselineSpec(
        name="controlnetpp",
        display_name="ControlNet++",
        repo_dir="../Controlla_Benchmarks/ControlNetPlusPlus",
        entrypoint="infer.py",
        paper_role="strong_adapter_conditioning_baseline",
        notes="Stronger ControlNet-style conditioning baseline.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=False,
        supports_identity=False,
        supports_image_condition=True,
    ),

    "icedit": BaselineSpec(
        name="icedit",
        display_name="ICEdit",
        repo_dir="../Controlla_Benchmarks/ICEdit",
        entrypoint="infer.py",
        paper_role="image_editing_baseline",
        notes="Image-conditioned editing baseline reported in the paper.",
        requires_training=False,
        requires_reference=True,
        supports_text=True,
        supports_audio=False,
        supports_identity=True,
        supports_image_condition=True,
    ),

    "flux_kontext": BaselineSpec(
        name="flux_kontext",
        display_name="FLUX.1 Kontext",
        repo_dir="../Controlla_Benchmarks/FLUX-Kontext",
        entrypoint="infer.py",
        paper_role="modern_instruction_image_editing_baseline",
        notes="Modern FLUX Kontext image-editing baseline.",
        requires_training=False,
        requires_reference=True,
        supports_text=True,
        supports_audio=False,
        supports_identity=True,
        supports_image_condition=True,
    ),

    "dreambooth": BaselineSpec(
        name="dreambooth",
        display_name="DreamBooth",
        repo_dir="../Controlla_Benchmarks/DreamBooth",
        entrypoint="train_dreambooth.py",
        paper_role="identity_personalization_baseline",
        notes="Identity personalization baseline.",
        requires_training=True,
        requires_reference=True,
        supports_text=True,
        supports_audio=False,
        supports_identity=True,
        supports_image_condition=False,
    ),

    "dreambooth_controlnetpp": BaselineSpec(
        name="dreambooth_controlnetpp",
        display_name="DreamBooth + ControlNet++",
        repo_dir="../Controlla_Benchmarks/DreamBooth-ControlNetPlusPlus",
        entrypoint="run_pipeline.py",
        paper_role="identity_personalization_plus_control_baseline",
        notes="Combined identity personalization and ControlNet++ conditioning baseline.",
        requires_training=True,
        requires_reference=True,
        supports_text=True,
        supports_audio=False,
        supports_identity=True,
        supports_image_condition=True,
    ),

    "sdxl": BaselineSpec(
        name="sdxl",
        display_name="SDXL",
        repo_dir="../Controlla_Benchmarks/SDXL",
        entrypoint="infer.py",
        paper_role="architecture_level_backbone_baseline",
        notes="Architecture-level SDXL comparison baseline.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=False,
        supports_identity=False,
        supports_image_condition=False,
    ),

    "sdxl_controlnetpp": BaselineSpec(
        name="sdxl_controlnetpp",
        display_name="SDXL + ControlNet++",
        repo_dir="../Controlla_Benchmarks/SDXL-ControlNetPlusPlus",
        entrypoint="infer.py",
        paper_role="architecture_level_control_baseline",
        notes="Architecture-level SDXL plus ControlNet++ baseline.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=False,
        supports_identity=False,
        supports_image_condition=True,
    ),

    "clip_retrieval": BaselineSpec(
        name="clip_retrieval",
        display_name="CLIP",
        repo_dir=".",
        entrypoint="experiments/baselines/retrieval_baselines.py",
        paper_role="cross_modal_retrieval_baseline",
        notes="CLIP retrieval baseline for I2T retrieval.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=False,
        supports_identity=False,
        supports_image_condition=True,
    ),

    "imagebind_retrieval": BaselineSpec(
        name="imagebind_retrieval",
        display_name="ImageBind",
        repo_dir=".",
        entrypoint="experiments/baselines/retrieval_baselines.py",
        paper_role="cross_modal_retrieval_baseline",
        notes="ImageBind retrieval baseline for I2T/I2A retrieval.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=True,
        supports_identity=False,
        supports_image_condition=True,
    ),

    # Optional legacy/extra baselines. Keep for broader experiments, but these
    # should not replace the paper-critical baselines above.
    "diffusionclip": BaselineSpec(
        name="diffusionclip",
        display_name="DiffusionCLIP",
        repo_dir="../Controlla_Benchmarks/DiffusionCLIP",
        entrypoint="main.py",
        paper_role="extra_text_driven_editing_baseline",
        notes="Optional text-driven semantic editing baseline.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=False,
        supports_identity=False,
        supports_image_condition=True,
    ),

    "instructpix2pix": BaselineSpec(
        name="instructpix2pix",
        display_name="InstructPix2Pix",
        repo_dir="../Controlla_Benchmarks/Instruct-pix2pix",
        entrypoint="edit_cli.py",
        paper_role="extra_instruction_editing_baseline",
        notes="Optional instruction-driven editing baseline.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=False,
        supports_identity=False,
        supports_image_condition=True,
    ),

    "emogen": BaselineSpec(
        name="emogen",
        display_name="EmoGen",
        repo_dir="../Controlla_Benchmarks/EmoGen",
        entrypoint="README.md",
        paper_role="extra_emotion_generation_baseline",
        notes="Optional emotion generation baseline.",
        requires_training=False,
        requires_reference=False,
        supports_text=True,
        supports_audio=False,
        supports_identity=False,
        supports_image_condition=False,
    ),

    "emoedit": BaselineSpec(
        name="emoedit",
        display_name="EmoEdit",
        repo_dir="../Controlla_Benchmarks/EmoEdit",
        entrypoint="test.py",
        paper_role="extra_emotion_editing_baseline",
        notes="Optional emotion editing baseline.",
        requires_training=False,
        requires_reference=True,
        supports_text=True,
        supports_audio=False,
        supports_identity=True,
        supports_image_condition=True,
    ),

    "emoportraits": BaselineSpec(
        name="emoportraits",
        display_name="EMOPortraits",
        repo_dir="../Controlla_Benchmarks/EMOPortraits",
        entrypoint="run_video_driven_pipeline.py",
        paper_role="extra_portrait_animation_baseline",
        notes="Optional portrait animation baseline.",
        requires_training=False,
        requires_reference=True,
        supports_text=False,
        supports_audio=True,
        supports_identity=True,
        supports_image_condition=True,
    ),
}


PAPER_BASELINES = [
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
]


def get_baseline_spec(name: str) -> BaselineSpec:
    """Resolve a baseline specification by canonical name."""

    key = name.strip().lower()

    if key not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY))
        raise KeyError(f"Unknown baseline: {name}. Available baselines: {available}")

    return BASELINE_REGISTRY[key]


def list_baselines(paper_only: bool = False) -> list[str]:
    """List canonical baseline names."""

    if paper_only:
        return list(PAPER_BASELINES)

    return sorted(BASELINE_REGISTRY)