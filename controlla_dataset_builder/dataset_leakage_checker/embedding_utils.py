"""Lightweight embedding and hashing helpers for leakage analysis.

The helpers intentionally use deterministic fallbacks so the checker remains
runnable without heavyweight pretrained models.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image


TOKEN_RE = re.compile(r"[a-z0-9']+")


def batch_iterable(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start : start + max(1, batch_size)]


def sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(TOKEN_RE.findall(text.lower()))


def safe_load_npy(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists() or target.suffix != ".npy":
        return None
    try:
        array = np.load(target)
    except Exception:
        return None
    if array.ndim == 0:
        return np.asarray([float(array)], dtype=np.float32)
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _resample_vector(vector: np.ndarray, target_dim: int) -> np.ndarray:
    if vector.shape[0] == target_dim:
        return vector.astype(np.float32)
    source_axis = np.linspace(0.0, 1.0, num=vector.shape[0], dtype=np.float32)
    target_axis = np.linspace(0.0, 1.0, num=target_dim, dtype=np.float32)
    return np.interp(target_axis, source_axis, vector).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] != right.shape[0]:
        target_dim = max(left.shape[0], right.shape[0])
        left = _resample_vector(left, target_dim)
        right = _resample_vector(right, target_dim)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    cosine = float(np.dot(left, right) / denominator)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def vector_signature(vector: np.ndarray, prefix_dims: int = 8) -> str:
    signs = ["1" if value >= 0 else "0" for value in vector[:prefix_dims]]
    return "".join(signs)


def deterministic_stub_embedding(key: str, *, dimension: int) -> np.ndarray:
    digest = hashlib.sha256(key.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(dimension).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def text_embedding(text: str | None, *, dimension: int = 64) -> np.ndarray:
    normalized = normalize_text(text)
    if not normalized:
        return np.zeros(dimension, dtype=np.float32)
    vector = np.zeros(dimension, dtype=np.float32)
    for token in TOKEN_RE.findall(normalized):
        index = sum((offset + 1) * ord(character) for offset, character in enumerate(token)) % dimension
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def image_perceptual_hash(path: str | Path | None, *, size: int = 8) -> str | None:
    if not path:
        return None
    target = Path(path)
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
    if not left_hash or not right_hash:
        return None
    if len(left_hash) != len(right_hash):
        return None
    left_bits = bin(int(left_hash, 16))[2:].zfill(len(left_hash) * 4)
    right_bits = bin(int(right_hash, 16))[2:].zfill(len(right_hash) * 4)
    return sum(1 for left_bit, right_bit in zip(left_bits, right_bits) if left_bit != right_bit)


def image_embedding(
    image_path: str | Path | None,
    *,
    feature_path: str | Path | None = None,
    dimension: int = 64,
) -> np.ndarray:
    cached = safe_load_npy(feature_path)
    if cached is not None:
        norm = float(np.linalg.norm(cached))
        return cached / norm if norm > 0 else cached
    phash = image_perceptual_hash(image_path)
    if phash:
        return deterministic_stub_embedding(f"image:{phash}", dimension=dimension)
    file_hash = sha256_file(image_path)
    if file_hash:
        return deterministic_stub_embedding(f"image:{file_hash}", dimension=dimension)
    return np.zeros(dimension, dtype=np.float32)


def audio_embedding(
    audio_path: str | Path | None,
    *,
    feature_path: str | Path | None = None,
    dimension: int = 64,
) -> np.ndarray:
    cached = safe_load_npy(feature_path)
    if cached is not None:
        norm = float(np.linalg.norm(cached))
        return cached / norm if norm > 0 else cached
    file_hash = sha256_file(audio_path)
    if file_hash:
        return deterministic_stub_embedding(f"audio:{file_hash}", dimension=dimension)
    if audio_path:
        return deterministic_stub_embedding(f"audio-path:{audio_path}", dimension=dimension)
    return np.zeros(dimension, dtype=np.float32)


def audio_exact_hash(audio_path: str | Path | None) -> str | None:
    return sha256_file(audio_path)


def image_exact_hash(image_path: str | Path | None) -> str | None:
    return sha256_file(image_path)


def jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentage(count: int, total: int) -> float:
    return 0.0 if total <= 0 else round(100.0 * count / total, 4)


def representative_examples(items: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return items[:limit]
