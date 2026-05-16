"""Multimodal consistency scoring with real-backend hooks and deterministic fallbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .embedding_matcher import text_to_embedding
from .identity_matcher import cosine_similarity


def _load_optional_vector(record: dict[str, Any], *keys: str) -> np.ndarray | None:
    for key in keys:
        value = record.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return np.load(path).astype(np.float32)
    vector = record.get("_vector")
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32)
    return None


def _text_vector(record: dict[str, Any]) -> np.ndarray:
    vector = _load_optional_vector(record, "text_embedding_path", "clip_text_embedding_path", "imagebind_text_embedding_path")
    if vector is not None:
        return vector
    return text_to_embedding(record.get("conditioning_text") or record.get("text") or "")


def _image_vector(record: dict[str, Any]) -> np.ndarray:
    vector = _load_optional_vector(record, "clip_image_embedding_path", "imagebind_image_embedding_path", "identity_feature_path")
    if vector is not None:
        return vector
    return text_to_embedding(str(record.get("image_path") or record.get("sample_id") or "missing-image"))


def _audio_vector(record: dict[str, Any] | None) -> np.ndarray:
    if record is None:
        return np.zeros(64, dtype=np.float32)
    vector = _load_optional_vector(record, "imagebind_audio_embedding_path", "audio_feature_path")
    if vector is not None:
        return vector
    return text_to_embedding(str(record.get("audio_path") or record.get("sample_id") or "missing-audio"))


def _clip_score_from_local_checkpoint(image_path: str, text: str, *, model_name: str, device: str) -> float | None:
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
        model.eval()
        image = Image.open(image_path).convert("RGB")
        image_inputs = processor(images=[image], return_tensors="pt").to(device)
        text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            cosine = float((image_features * text_features).sum(dim=-1).item())
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
    except Exception:
        return None


def _imagebind_scores_from_installed_package(
    image_path: str,
    audio_path: str,
    text: str,
    *,
    device: str,
) -> tuple[float, float] | None:
    try:
        import torch
        from imagebind import data
        import imagebind.models.imagebind_model as imagebind_model
        from imagebind.models.imagebind_model import ModalityType

        model = imagebind_model.imagebind_huge(pretrained=True).to(device)
        model.eval()
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data([image_path], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], device),
            ModalityType.TEXT: data.load_and_transform_text([text], device),
        }
        with torch.no_grad():
            embeddings = model(inputs)
            image = torch.nn.functional.normalize(embeddings[ModalityType.VISION], dim=-1)
            audio = torch.nn.functional.normalize(embeddings[ModalityType.AUDIO], dim=-1)
            text_embedding = torch.nn.functional.normalize(embeddings[ModalityType.TEXT], dim=-1)
            image_audio = float((image * audio).sum(dim=-1).item())
            text_audio = float((text_embedding * audio).sum(dim=-1).item())
        return (
            max(0.0, min(1.0, (image_audio + 1.0) / 2.0)),
            max(0.0, min(1.0, (text_audio + 1.0) / 2.0)),
        )
    except Exception:
        return None


@dataclass(frozen=True)
class ConsistencyScores:
    image_text_similarity: float
    image_audio_similarity: float
    text_audio_similarity: float
    clip_backend: str
    imagebind_backend: str


class MultimodalConsistencyScorer:
    """Score image-text, image-audio, and text-audio consistency.

    The scorer uses real CLIP/ImageBind backends when they are available from
    local checkpoints or installed packages. It falls back to deterministic
    proxy features for tests and offline builds, and records which backend was
    used so release metadata does not silently overclaim.
    """

    def __init__(
        self,
        *,
        clip_model_name: str | None = None,
        device: str | None = None,
        enable_real_clip: bool | None = None,
        enable_real_imagebind: bool | None = None,
    ) -> None:
        self.clip_model_name = clip_model_name or os.getenv("CONTROLLA_CLIP_MODEL", "openai/clip-vit-base-patch32")
        self.device = device or os.getenv("CONTROLLA_DEVICE", "cpu")
        self.enable_real_clip = bool(int(os.getenv("CONTROLLA_USE_REAL_CLIP", "0"))) if enable_real_clip is None else enable_real_clip
        self.enable_real_imagebind = (
            bool(int(os.getenv("CONTROLLA_USE_REAL_IMAGEBIND", "0"))) if enable_real_imagebind is None else enable_real_imagebind
        )

    def score(
        self,
        *,
        image_record: dict[str, Any],
        text_record: dict[str, Any],
        audio_record: dict[str, Any] | None,
    ) -> ConsistencyScores:
        image_text = None
        text = text_record.get("conditioning_text") or text_record.get("text") or ""
        image_path = str(image_record.get("image_path") or "")

        if self.enable_real_clip and image_path and text:
            image_text = _clip_score_from_local_checkpoint(
                image_path,
                text,
                model_name=self.clip_model_name,
                device=self.device,
            )
        clip_backend = "clip" if image_text is not None else "deterministic_proxy"
        if image_text is None:
            image_text = cosine_similarity(_image_vector(image_record), _text_vector(text_record))

        image_audio = None
        text_audio = None
        imagebind_backend = "deterministic_proxy"
        audio_path = str(audio_record.get("audio_path") or "") if audio_record else ""
        if self.enable_real_imagebind and image_path and audio_path and text:
            imagebind_scores = _imagebind_scores_from_installed_package(
                image_path,
                audio_path,
                text,
                device=self.device,
            )
            if imagebind_scores is not None:
                image_audio, text_audio = imagebind_scores
                imagebind_backend = "imagebind"

        if image_audio is None or text_audio is None:
            image_vector = _image_vector(image_record)
            audio_vector = _audio_vector(audio_record)
            text_vector = _text_vector(text_record)
            image_audio = cosine_similarity(image_vector, audio_vector)
            text_audio = cosine_similarity(text_vector, audio_vector)

        return ConsistencyScores(
            image_text_similarity=float(image_text),
            image_audio_similarity=float(image_audio),
            text_audio_similarity=float(text_audio),
            clip_backend=clip_backend,
            imagebind_backend=imagebind_backend,
        )
