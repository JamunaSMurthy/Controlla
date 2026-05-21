"""
Diffusion backbone abstractions for Controlla.
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
        images: Generated or reconstructed images with shape [B, 3, H, W].
        diffusion_loss: Scalar training loss tensor.
        latents: Optional latent tensor for debugging.
    """

    images: torch.Tensor
    diffusion_loss: torch.Tensor
    latents: torch.Tensor | None = None


class MockDiffusionBackbone(nn.Module):
    """Small conditional generator used for tests and dummy runs."""

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        image_size: int,
        cross_attention_dim: int,
    ) -> None:
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
        if condition_tokens.ndim == 2:
            token_summary = condition_tokens
        elif condition_tokens.ndim == 3:
            token_summary = condition_tokens.mean(dim=1)
        else:
            raise ValueError(
                f"condition_tokens must have shape [B, D] or [B, T, D], got {tuple(condition_tokens.shape)}"
            )

        combined = torch.cat([global_condition, z_id, z_attr, token_summary], dim=-1)
        condition = self.condition_projection(combined)
        return condition.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.image_size, self.image_size
        )

    def forward(
        self,
        target_images: torch.Tensor,
        condition_tokens: torch.Tensor,
        global_condition: torch.Tensor,
        z_id: torch.Tensor,
        z_attr: torch.Tensor,
    ) -> DiffusionBackboneOutput:
        noise = torch.randn_like(target_images)
        condition_map = self._build_condition_map(
            condition_tokens=condition_tokens,
            global_condition=global_condition,
            z_id=z_id,
            z_attr=z_attr,
        )
        generated = self.generator(torch.cat([noise, condition_map], dim=1))
        diffusion_loss = F.mse_loss(generated, target_images)

        return DiffusionBackboneOutput(
            images=generated,
            diffusion_loss=diffusion_loss,
            latents=condition_map,
        )

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
        noise = torch.randn(
            (batch_size, 3, self.image_size, self.image_size),
            device=device,
            generator=generator,
        )
        condition_map = self._build_condition_map(
            condition_tokens=condition_tokens,
            global_condition=global_condition,
            z_id=z_id,
            z_attr=z_attr,
        )
        images = self.generator(torch.cat([noise, condition_map], dim=1))

        return DiffusionBackboneOutput(
            images=images,
            diffusion_loss=torch.zeros((), device=device),
            latents=condition_map,
        )


class DiffusersStableDiffusionBackbone(nn.Module):
    """Stable Diffusion wrapper with Controlla factor conditioning.

    This backend injects Controlla's learned factors into the UNet by appending
    trainable factor-conditioning tokens to the encoder_hidden_states.

    Expected inputs:
    - condition_tokens: [B, T, D_cond] or [B, D_cond]
    - global_condition: [B, D_global]
    - z_id: [B, D_z]
    - z_attr: [B, D_z]

    The adapter maps [global_condition; z_id; z_attr] into one or more
    cross-attention tokens with dimension equal to the UNet cross-attention dim.
    """

    def __init__(
        self,
        model_source: str,
        condition_token_dim: int,
        global_condition_dim: int,
        z_dim: int,
        local_files_only: bool = True,
        freeze_unet: bool = True,
        num_factor_tokens: int = 4,
        train_condition_token_projection: bool = True,
    ) -> None:
        super().__init__()

        try:
            from diffusers import DDPMScheduler, StableDiffusionPipeline
        except ImportError as error:
            raise ImportError(
                "diffusers backend requested but diffusers is not installed. "
                "Install it with: pip install diffusers transformers accelerate safetensors"
            ) from error

        pipeline = StableDiffusionPipeline.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )

        self.vae = pipeline.vae
        self.unet = pipeline.unet
        self.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

        self.condition_token_dim = int(condition_token_dim)
        self.global_condition_dim = int(global_condition_dim)
        self.z_dim = int(z_dim)
        self.num_factor_tokens = int(num_factor_tokens)

        self.unet_cross_attention_dim = int(self.unet.config.cross_attention_dim)

        self.vae.requires_grad_(False)

        if freeze_unet:
            self.unet.requires_grad_(False)

        if self.condition_token_dim != self.unet_cross_attention_dim:
            self.condition_token_projection: nn.Module = nn.Linear(
                self.condition_token_dim,
                self.unet_cross_attention_dim,
            )
        else:
            self.condition_token_projection = nn.Identity()

        if not train_condition_token_projection:
            self.condition_token_projection.requires_grad_(False)

        factor_input_dim = self.global_condition_dim + (2 * self.z_dim)

        self.factor_adapter = nn.Sequential(
            nn.LayerNorm(factor_input_dim),
            nn.Linear(factor_input_dim, self.unet_cross_attention_dim * 2),
            nn.GELU(),
            nn.Linear(
                self.unet_cross_attention_dim * 2,
                self.num_factor_tokens * self.unet_cross_attention_dim,
            ),
        )

    def _prepare_condition_tokens(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        if condition_tokens.ndim == 2:
            condition_tokens = condition_tokens.unsqueeze(1)

        if condition_tokens.ndim != 3:
            raise ValueError(
                "condition_tokens must have shape [B, D] or [B, T, D], "
                f"got {tuple(condition_tokens.shape)}"
            )

        return self.condition_token_projection(condition_tokens)

    def _build_encoder_hidden_states(
        self,
        condition_tokens: torch.Tensor,
        global_condition: torch.Tensor,
        z_id: torch.Tensor,
        z_attr: torch.Tensor,
    ) -> torch.Tensor:
        condition_tokens = self._prepare_condition_tokens(condition_tokens)

        if global_condition.ndim != 2:
            raise ValueError(
                f"global_condition must have shape [B, D], got {tuple(global_condition.shape)}"
            )
        if z_id.ndim != 2:
            raise ValueError(f"z_id must have shape [B, D], got {tuple(z_id.shape)}")
        if z_attr.ndim != 2:
            raise ValueError(f"z_attr must have shape [B, D], got {tuple(z_attr.shape)}")

        batch_size = condition_tokens.shape[0]

        if global_condition.shape[0] != batch_size or z_id.shape[0] != batch_size or z_attr.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch among condition_tokens, global_condition, z_id, and z_attr"
            )

        factor_condition = torch.cat([global_condition, z_id, z_attr], dim=-1)

        factor_tokens = self.factor_adapter(factor_condition)
        factor_tokens = factor_tokens.view(
            batch_size,
            self.num_factor_tokens,
            self.unet_cross_attention_dim,
        )

        encoder_hidden_states = torch.cat([condition_tokens, factor_tokens], dim=1)
        return encoder_hidden_states

    def _encode_images_to_latents(self, target_images: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(target_images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def _decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        scaled_latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(scaled_latents).sample
        return images.clamp(-1.0, 1.0)

    def forward(
        self,
        target_images: torch.Tensor,
        condition_tokens: torch.Tensor,
        global_condition: torch.Tensor,
        z_id: torch.Tensor,
        z_attr: torch.Tensor,
    ) -> DiffusionBackboneOutput:
        encoder_hidden_states = self._build_encoder_hidden_states(
            condition_tokens=condition_tokens,
            global_condition=global_condition,
            z_id=z_id,
            z_attr=z_attr,
        )

        latents = self._encode_images_to_latents(target_images)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            low=0,
            high=int(self.scheduler.config.num_train_timesteps),
            size=(latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        noise_prediction = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        diffusion_loss = F.mse_loss(noise_prediction.float(), noise.float())

        with torch.no_grad():
            preview_images = self._decode_latents_to_images(latents)

        return DiffusionBackboneOutput(
            images=preview_images,
            diffusion_loss=diffusion_loss,
            latents=latents,
        )

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
        encoder_hidden_states = self._build_encoder_hidden_states(
            condition_tokens=condition_tokens,
            global_condition=global_condition,
            z_id=z_id,
            z_attr=z_attr,
        )

        height = int(self.unet.config.sample_size) * self.vae_scale_factor
        width = int(self.unet.config.sample_size) * self.vae_scale_factor

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        latents = torch.randn(
            (
                batch_size,
                int(self.unet.config.in_channels),
                latent_height,
                latent_width,
            ),
            device=device,
            generator=generator,
            dtype=encoder_hidden_states.dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for timestep in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, timestep)

            noise_prediction = self.unet(
                latent_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            latents = self.scheduler.step(
                noise_prediction,
                timestep,
                latents,
            ).prev_sample

        images = self._decode_latents_to_images(latents)

        return DiffusionBackboneOutput(
            images=images,
            diffusion_loss=torch.zeros((), device=device),
            latents=latents,
        )

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)


def build_diffusion_backbone(config: dict[str, Any]) -> nn.Module:
    """Instantiate the configured diffusion backbone."""

    model_config = config["model"]
    backbone_name = str(model_config["diffusion_backbone"])

    if backbone_name == "mock":
        return MockDiffusionBackbone(
            hidden_dim=int(model_config["hidden_dim"]),
            z_dim=int(model_config["z_dim"]),
            image_size=int(model_config["image_size"]),
            cross_attention_dim=int(model_config["cross_attention_dim"]),
        )

    if backbone_name == "diffusers":
        model_source = str(
            model_config.get("diffusion_model_path")
            or model_config["diffusion_model_id"]
        )
        local_only = bool(model_config.get("diffusion_local_only", True))
        model_path = Path(model_source)

        if local_only and not model_path.exists():
            raise FileNotFoundError(
                "Diffusers backbone requires local Stable Diffusion weights when "
                "model.diffusion_local_only=True. Set model.diffusion_model_path "
                f"to an existing pipeline directory. Got: {model_source}"
            )

        return DiffusersStableDiffusionBackbone(
            model_source=model_source,
            condition_token_dim=int(model_config["cross_attention_dim"]),
            global_condition_dim=int(model_config["hidden_dim"]),
            z_dim=int(model_config["z_dim"]),
            local_files_only=local_only,
            freeze_unet=bool(model_config.get("freeze_unet", True)),
            num_factor_tokens=int(model_config.get("num_factor_tokens", 4)),
            train_condition_token_projection=bool(
                model_config.get("train_condition_token_projection", True)
            ),
        )

    raise ValueError(f"Unsupported diffusion backbone: {backbone_name}")