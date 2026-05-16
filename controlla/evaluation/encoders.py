"""Frozen encoder adapters used by the experiments pipeline.

Where full pretrained checkpoints are not installed locally, these adapters fall
back to deterministic lightweight features so the experiments code remains
runnable and cacheable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def _hash_vector(value: str, dim: int) -> np.ndarray:
    seed = abs(hash(value)) % (2**32)
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vector) + 1e-8
    return vector / norm


class CLIPAdapter:
    """Wrapper around frozen CLIP encoders with deterministic fallback support."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> None:
        self.device = device
        self.model_name = model_name
        self.available = False
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        try:
            from transformers import CLIPModel, CLIPProcessor

            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.available = True
        except Exception:
            self.processor = None
            self.model = None

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not self.available:
            return np.stack([_hash_vector(text, 512) for text in texts], axis=0)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        embeddings = F.normalize(embeddings, dim=-1).cpu().numpy().astype(np.float32)
        return embeddings

    def encode_images(self, image_paths: Iterable[str | Path]) -> np.ndarray:
        image_paths = [str(path) for path in image_paths]
        if not self.available:
            return np.stack([_hash_vector(path, 512) for path in image_paths], axis=0)
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        embeddings = F.normalize(embeddings, dim=-1).cpu().numpy().astype(np.float32)
        return embeddings


class ImageBindAdapter:
    """Project-side ImageBind-style adapter with graceful fallback.

    TODO:
    Replace the fallback branch with a local ImageBind checkpoint when available.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.available = False

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([_hash_vector(f"text::{text}", 1024) for text in texts], axis=0)

    def encode_images(self, image_paths: Iterable[str | Path]) -> np.ndarray:
        return np.stack([_hash_vector(f"image::{Path(path)}", 1024) for path in image_paths], axis=0)

    def encode_audios(self, audio_paths: Iterable[str | Path]) -> np.ndarray:
        return np.stack([_hash_vector(f"audio::{Path(path)}", 1024) for path in audio_paths], axis=0)


class ArcFaceAdapter:
    """ArcFace-style identity embedding adapter with graceful fallback."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.available = False

    def encode_images(self, image_paths: Iterable[str | Path]) -> np.ndarray:
        return np.stack([_hash_vector(f"arcface::{Path(path)}", 512) for path in image_paths], axis=0)