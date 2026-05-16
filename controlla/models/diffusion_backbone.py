"""Diffusion backbone abstractions for Controlla.

Two backends are provided:

- `mock`: lightweight image generator for fast tests and smoke runs.
- `diffusers`: Stable Diffusion-style latent diffusion using diffusers modules.

The mock path is the lightweight fallback. The diffusers path is the default
research configuration and expects a local Stable Diffusion pipeline directory
when `diffusion_local_only` is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DiffusionBackboneOutput:
    """Outputs from the diffusion backbone.

    Attributes:
        images: Generated or reconstructed images with shape `[B, 3, H, W]`.
        diffusion_loss: Scalar training loss tensor.
        latents: Optional latent tensor for debugging.
    """

    images: torch.Tensor
    diffusion_loss: torch.Tensor
    latents: torch.Tensor | None = None


class MockDiffusionBackbone(nn.Module):
    """Small conditional generator used for tests and dummy runs."""

    def __init__(self, hidden_dim: int, z_dim: int, image_size: int, cross_attention_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        condition_dim = hidden_dim + z_dim * 2 + cross_attention_dim
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.generator = nn.Sequential(
            nn.Conv2d(hidden_dim + 3, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def _build_condition_map(
        self,
        condition_tokens: torch.Tensor,
        global_condition: torch.Tensor,
        z_id: torch.Tensor,
        z_attr: torch.Tensor,
    ) -> torch.Tensor:
        token_summary = condition_tokens.mean(dim=1)
        combined = torch.cat([global_condition, z_id, z_attr, token_summary], dim=-1)
        condition = self.condition_projection(combined)
        return condition.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.image_size, self.image_size)

    def forward(
        self,
        target_images: torch.Tensor,
        condition_tokens: torch.Tensor,
        global_condition: torch.Tensor,
        z_id: torch.Tensor,
        z_attr: torch.Tensor,
    ) -> DiffusionBackboneOutput:
        noise = torch.randn_like(target_images)
        condition_map = self._build_condition_map(condition_tokens, global_condition, z_id, z_attr)
        generated = self.generator(torch.cat([noise, condition_map], dim=1))
        diffusion_loss = F.mse_loss(generated, target_images)
        return DiffusionBackboneOutput(images=generated, diffusion_loss=diffusion_loss, latents=condition_map)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        condition_tokens: torch.Tensor,
        global_condition: torch.Tensor,
        z_id: torch.Tensor,
        z_attr: torch.Tensor,
        device: torch.device,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> DiffusionBackboneOutput:
        noise = torch.randn((batch_size, 3, self.image_size, self.image_size), device=device, generator=generator)
        condition_map = self._build_condition_map(condition_tokens, global_condition, z_id, z_attr)
        images = self.generator(torch.cat([noise, condition_map], dim=1))
        return DiffusionBackboneOutput(images=images, diffusion_loss=torch.zeros((), device=device), latents=condition_map)


class DiffusersStableDiffusionBackbone(nn.Module):
    """Minimal diffusers-based Stable Diffusion wrapper.

    TODO:
    Replace the simplified training path with a fuller fine-tuning strategy once
    the prototype loss stack and adapter design are stabilized.
    """

    def __init__(self, model_source: str, cross_attention_dim: int, local_files_only: bool = True) -> None:
        super().__init__()
        try:
            from diffusers import DDPMScheduler, StableDiffusionPipeline
        except ImportError as error:
            raise ImportError("diffusers backend requested but diffusers is not installed") from error

        pipeline = StableDiffusionPipeline.from_pretrained(model_source, local_files_only=local_files_only)
        self.vae = pipeline.vae
        self.unet = pipeline.unet
        self.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        self.cross_attention_dim = cross_attention_dim

        self.vae.requires_grad_(False)

    def forward(
        self,
        target_images: torch.Tensor,
        condition_tokens: torch.Tensor,
        global_condition: torch.Tensor,
        z_id: torch.Tensor,
        z_attr: torch.Tensor,
    ) -> DiffusionBackboneOutput:
        del global_condition, z_id, z_attr
        latents = self.vae.encode(target_images).latent_dist.sample() * self.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        noise_prediction = self.unet(noisy_latents, timesteps, encoder_hidden_states=condition_tokens).sample
        diffusion_loss = F.mse_loss(noise_prediction, noise)
        # TODO: decode a better preview during training instead of returning the target image.
        return DiffusionBackboneOutput(images=target_images, diffusion_loss=diffusion_loss, latents=latents)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        condition_tokens: torch.Tensor,
        global_condition: torch.Tensor,
        z_id: torch.Tensor,
        z_attr: torch.Tensor,
        device: torch.device,
        num_inference_steps: int = 20,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> DiffusionBackboneOutput:
        del global_condition, z_id, z_attr
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor
        latent_height = height // 8
        latent_width = width // 8
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, latent_height, latent_width),
            device=device,
            generator=generator,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for timestep in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, timestep)
            noise_prediction = self.unet(latent_input, timestep, encoder_hidden_states=condition_tokens).sample
            latents = self.scheduler.step(noise_prediction, timestep, latents).prev_sample

        scaled_latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(scaled_latents).sample
        images = images.clamp(-1.0, 1.0)
        return DiffusionBackboneOutput(images=images, diffusion_loss=torch.zeros((), device=device), latents=latents)

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)


def build_diffusion_backbone(config: dict[str, Any]) -> nn.Module:
    """Instantiate the configured diffusion backbone."""
    model_config = config["model"]
    backbone_name = model_config["diffusion_backbone"]
    if backbone_name == "mock":
        return MockDiffusionBackbone(
            hidden_dim=int(model_config["hidden_dim"]),
            z_dim=int(model_config["z_dim"]),
            image_size=int(model_config["image_size"]),
            cross_attention_dim=int(model_config["cross_attention_dim"]),
        )
    if backbone_name == "diffusers":
        model_source = str(model_config.get("diffusion_model_path") or model_config["diffusion_model_id"])
        local_only = bool(model_config.get("diffusion_local_only", True))
        model_path = Path(model_source)
        if local_only and not model_path.exists():
            raise FileNotFoundError(
                "Diffusers backbone requires local Stable Diffusion weights. "
                f"Set model.diffusion_model_path to an existing pipeline directory; got: {model_source}"
            )
        return DiffusersStableDiffusionBackbone(
            model_source=model_source,
            cross_attention_dim=int(model_config["cross_attention_dim"]),
            local_files_only=local_only,
        )
    raise ValueError(f"Unsupported diffusion backbone: {backbone_name}")