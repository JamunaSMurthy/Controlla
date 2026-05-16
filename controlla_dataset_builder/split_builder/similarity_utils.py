"""Similarity and hashing helpers used by the split builder."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image


TOKEN_RE = re.compile(r"[a-z0-9']+")


def _normalize_path(path: str | None) -> str | None:
    if not path:
        return None
    return str(Path(path))


def batched(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start : start + max(1, batch_size)]


def sha256_file(path: str | None) -> str | None:
    normalized_path = _normalize_path(path)
    if not normalized_path:
        return None
    return _sha256_file_cached(normalized_path)


@lru_cache(maxsize=131072)
def _sha256_file_cached(normalized_path: str) -> str | None:
    target = Path(normalized_path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(TOKEN_RE.findall(text.lower()))


def sha256_text(text: str | None) -> str | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def safe_load_npy(path: str | None) -> np.ndarray | None:
    normalized_path = _normalize_path(path)
    if not normalized_path:
        return None
    return _safe_load_npy_cached(normalized_path)


@lru_cache(maxsize=32768)
def _safe_load_npy_cached(normalized_path: str) -> np.ndarray | None:
    target = Path(normalized_path)
    if not target.exists() or target.suffix.lower() != ".npy":
        return None
    try:
        array = np.load(target)
    except Exception:
        return None
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _resample(vector: np.ndarray, target_dim: int) -> np.ndarray:
    if vector.shape[0] == target_dim:
        return vector.astype(np.float32)
    source_axis = np.linspace(0.0, 1.0, num=vector.shape[0], dtype=np.float32)
    target_axis = np.linspace(0.0, 1.0, num=target_dim, dtype=np.float32)
    return np.interp(target_axis, source_axis, vector).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] != right.shape[0]:
        target = max(left.shape[0], right.shape[0])
        left = _resample(left, target)
        right = _resample(right, target)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    cosine = float(np.dot(left, right) / denominator)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def deterministic_stub_embedding(key: str, *, dimension: int = 64) -> np.ndarray:
    return _deterministic_stub_embedding_cached(key, dimension)


@lru_cache(maxsize=262144)
def _deterministic_stub_embedding_cached(key: str, dimension: int = 64) -> np.ndarray:
    digest = hashlib.sha256(key.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dimension).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def text_embedding(text: str | None, *, dimension: int = 64) -> np.ndarray:
    normalized = normalize_text(text)
    if not normalized:
        return np.zeros(dimension, dtype=np.float32)
    vector = np.zeros(dimension, dtype=np.float32)
    for token in TOKEN_RE.findall(normalized):
        index = sum((offset + 1) * ord(char) for offset, char in enumerate(token)) % dimension
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def image_perceptual_hash(path: str | None, *, size: int = 8) -> str | None:
    normalized_path = _normalize_path(path)
    if not normalized_path:
        return None
    return _image_perceptual_hash_cached(normalized_path, size)


@lru_cache(maxsize=131072)
def _image_perceptual_hash_cached(normalized_path: str, size: int = 8) -> str | None:
    target = Path(normalized_path)
    if not target.exists() or not target.is_file():
        return None
    try:
        with Image.open(target) as image:
            grayscale = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            pixels = np.asarray(grayscale, dtype=np.float32)
    except Exception:
        return None
    threshold = float(pixels.mean())
    bits = "".join("1" if value >= threshold else "0" for value in pixels.reshape(-1))
    return f"{int(bits, 2):0{size * size // 4}x}"


def hamming_distance(left_hash: str | None, right_hash: str | None) -> int | None:
    if not left_hash or not right_hash or len(left_hash) != len(right_hash):
        return None
    left_bits = bin(int(left_hash, 16))[2:].zfill(len(left_hash) * 4)
    right_bits = bin(int(right_hash, 16))[2:].zfill(len(right_hash) * 4)
    return sum(1 for left_bit, right_bit in zip(left_bits, right_bits) if left_bit != right_bit)


def vector_signature(vector: np.ndarray, prefix_dims: int = 8) -> str:
    return "".join("1" if value >= 0 else "0" for value in vector[:prefix_dims])


def resolve_embedding(sample: dict[str, Any], *, kind: str, dimension: int = 64) -> np.ndarray:
    if kind == "face":
        for key in ("arcface_embedding_path", "identity_feature_path", "image_embedding_path"):
            array = safe_load_npy(sample.get(key))
            if array is not None:
                norm = float(np.linalg.norm(array))
                return array / norm if norm > 0 else array
        phash = sample.get("_face_phash") or sample.get("face_phash") or sample.get("_image_phash") or sample.get("image_phash")
        if not phash:
            phash = image_perceptual_hash(sample.get("image_path") or sample.get("reference_image_path"))
        if phash:
            return deterministic_stub_embedding(f"face:{phash}", dimension=dimension)
        file_hash = sample.get("_image_hash") or sample.get("image_hash") or sha256_file(sample.get("image_path") or sample.get("reference_image_path"))
        return deterministic_stub_embedding(f"face:{file_hash or sample['sample_id']}", dimension=dimension)
    if kind == "audio":
        for key in ("audio_embedding_path", "audio_feature_path"):
            array = safe_load_npy(sample.get(key))
            if array is not None:
                norm = float(np.linalg.norm(array))
                return array / norm if norm > 0 else array
        file_hash = sample.get("_audio_hash") or sample.get("audio_hash") or sha256_file(sample.get("audio_path"))
        return deterministic_stub_embedding(f"audio:{file_hash or sample['sample_id']}", dimension=dimension)
    if kind == "image":
        for key in ("clip_embedding_path", "image_embedding_path"):
            array = safe_load_npy(sample.get(key))
            if array is not None:
                norm = float(np.linalg.norm(array))
                return array / norm if norm > 0 else array
        phash = sample.get("_image_phash") or sample.get("image_phash")
        if not phash:
            phash = image_perceptual_hash(sample.get("image_path"))
        return deterministic_stub_embedding(f"image:{phash or sample['sample_id']}", dimension=dimension)
    if kind == "text":
        return text_embedding(sample.get("text"), dimension=dimension)
    raise ValueError(f"Unsupported embedding kind: {kind}")


def modality_signature(sample: dict[str, Any]) -> str:
    return "".join(
        [
            "I" if sample.get("has_image") else "_",
            "A" if sample.get("has_audio") else "_",
            "T" if sample.get("has_text") else "_",
        ]
    )