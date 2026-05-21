"""CLIP and ImageBind retrieval baselines for Controlla.

Paper alignment:
    Table 7 reports cross-modal retrieval for CLIP, ImageBind, and Controlla.
    This file provides wrappers for CLIP/I2T and ImageBind/I2T/I2A retrieval
    baselines using the shared evaluation encoders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from evaluation.encoders import CLIPAdapter, ImageBindAdapter


def _normalize(array: np.ndarray) -> np.ndarray:
    return array / (np.linalg.norm(array, axis=-1, keepdims=True) + 1e-8)


def recall_at_k(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    k: int,
    ground_truth_indices: np.ndarray | None = None,
) -> float:
    """Compute retrieval recall@k.

    Assumes query i matches target i when ground_truth_indices is None.
    """

    query_embeddings = _normalize(np.asarray(query_embeddings, dtype=np.float32))
    target_embeddings = _normalize(np.asarray(target_embeddings, dtype=np.float32))

    if query_embeddings.shape[0] == 0:
        return 0.0

    similarity = query_embeddings @ target_embeddings.T
    topk = np.argsort(-similarity, axis=1)[:, :k]

    if ground_truth_indices is None:
        ground_truth_indices = np.arange(query_embeddings.shape[0])

    hits = [
        int(ground_truth_indices[index]) in topk[index]
        for index in range(query_embeddings.shape[0])
    ]

    return float(np.mean(hits))


def clip_i2t_retrieval(
    image_paths: Iterable[str | Path],
    texts: Iterable[str],
    device: str = "cpu",
) -> dict[str, float]:
    """Run CLIP image-to-text retrieval baseline."""

    adapter = CLIPAdapter(device=device)

    image_embeddings = adapter.encode_images(image_paths)
    text_embeddings = adapter.encode_texts(texts)

    return {
        "CLIP_I2T_R@1": recall_at_k(image_embeddings, text_embeddings, k=1),
        "CLIP_I2T_R@5": recall_at_k(image_embeddings, text_embeddings, k=5),
    }


def imagebind_i2t_retrieval(
    image_paths: Iterable[str | Path],
    texts: Iterable[str],
    device: str = "cpu",
) -> dict[str, float]:
    """Run ImageBind image-to-text retrieval baseline."""

    adapter = ImageBindAdapter(device=device)

    image_embeddings = adapter.encode_images(image_paths)
    text_embeddings = adapter.encode_texts(texts)

    return {
        "ImageBind_I2T_R@1": recall_at_k(image_embeddings, text_embeddings, k=1),
        "ImageBind_I2T_R@5": recall_at_k(image_embeddings, text_embeddings, k=5),
    }


def imagebind_i2a_retrieval(
    image_paths: Iterable[str | Path],
    audio_paths: Iterable[str | Path],
    device: str = "cpu",
) -> dict[str, float]:
    """Run ImageBind image-to-audio retrieval baseline."""

    adapter = ImageBindAdapter(device=device)

    image_embeddings = adapter.encode_images(image_paths)
    audio_embeddings = adapter.encode_audios(audio_paths)

    return {
        "ImageBind_I2A_R@1": recall_at_k(image_embeddings, audio_embeddings, k=1),
        "ImageBind_I2A_R@5": recall_at_k(image_embeddings, audio_embeddings, k=5),
    }


def run_retrieval_baselines(
    image_paths: Iterable[str | Path],
    texts: Iterable[str],
    audio_paths: Iterable[str | Path] | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    """Run all retrieval baselines needed for the paper table."""

    image_paths = list(image_paths)
    texts = list(texts)

    results = {}
    results.update(clip_i2t_retrieval(image_paths, texts, device=device))
    results.update(imagebind_i2t_retrieval(image_paths, texts, device=device))

    if audio_paths is not None:
        results.update(imagebind_i2a_retrieval(image_paths, list(audio_paths), device=device))

    return results