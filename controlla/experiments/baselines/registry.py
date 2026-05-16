"""Baseline registry for Controlla experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaselineSpec:
    """Description of an external baseline and its integration points."""

    name: str
    repo_dir: str
    entrypoint: str
    notes: str

    def repo_path(self, project_root: str | Path) -> Path:
        return (Path(project_root) / self.repo_dir).resolve()


BASELINE_REGISTRY = {
    "controlnet": BaselineSpec("controlnet", "../Controlla_Benchmarks/ControlNet", "gradio_canny2image.py", "Adapter-style conditioning baseline."),
    "dreambooth": BaselineSpec("dreambooth", "../Controlla_Benchmarks/DreamBooth", "README.md", "Identity personalization baseline."),
    "diffusionclip": BaselineSpec("diffusionclip", "../Controlla_Benchmarks/DiffusionCLIP", "main.py", "Text-driven semantic editing baseline."),
    "instructpix2pix": BaselineSpec("instructpix2pix", "../Controlla_Benchmarks/Instruct-pix2pix", "edit_cli.py", "Instruction-driven editing baseline."),
    "styleclip": BaselineSpec("styleclip", "../Controlla_Benchmarks/StyleCLIP", "cog_predict.py", "Latent editing baseline."),
    "emogen": BaselineSpec("emogen", "../Controlla_Benchmarks/EmoGen", "README.md", "Emotion generation baseline."),
    "emoedit": BaselineSpec("emoedit", "../Controlla_Benchmarks/EmoEdit", "test.py", "Emotion editing baseline."),
    "emoportraits": BaselineSpec("emoportraits", "../Controlla_Benchmarks/EMOPortraits", "run_video_driven_pipeline.py", "Portrait animation baseline."),
    "controlla_full": BaselineSpec("controlla_full", ".", "pipelines/train_controlla.py", "Proposed full model."),
}


def get_baseline_spec(name: str) -> BaselineSpec:
    """Resolve a baseline specification by canonical name."""
    key = name.strip().lower()
    if key not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline: {name}")
    return BASELINE_REGISTRY[key]