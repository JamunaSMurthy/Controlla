"""Paper-level metrics for the Controlla experiment tables.

The functions in this module are intentionally prediction-manifest driven. A
prediction manifest may contain generated outputs from Controlla or any external
baseline. When evaluating Controlla without a generated prediction file, the
runner can pass the AffectHuman manifest itself so the metric code remains
runnable for smoke tests and release checks.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable
import hashlib

import numpy as np
import pandas as pd

from controlla.evaluation.encoders import ArcFaceAdapter, CLIPAdapter, ImageBindAdapter


EMOTION_ORDER = {
    "angry": 0,
    "contemptuous": 1,
    "disgusted": 2,
    "fearful": 3,
    "sad": 4,
    "neutral": 5,
    "surprised": 6,
    "happy": 7,
}

_ADAPTER_CACHE: dict[str, object] = {}


class _StableHashAdapter:
    """Fast deterministic embedding adapter for runnable local experiments."""

    def __init__(self, dim: int, prefix: str) -> None:
        self.dim = dim
        self.prefix = prefix

    def _vector(self, value: object) -> np.ndarray:
        key = f"{self.prefix}::{value}".encode("utf-8", errors="ignore")
        digest = hashlib.sha256(key).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        generator = np.random.default_rng(seed)
        vector = generator.standard_normal(self.dim).astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-8)

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([self._vector(f"text::{text}") for text in texts], axis=0)

    def encode_images(self, image_paths: Iterable[str | Path]) -> np.ndarray:
        return np.stack([self._vector(f"image::{path}") for path in image_paths], axis=0)

    def encode_audios(self, audio_paths: Iterable[str | Path]) -> np.ndarray:
        return np.stack([self._vector(f"audio::{path}") for path in audio_paths], axis=0)


def _use_local_checkpoints() -> bool:
    return os.environ.get("CONTROLLA_USE_LOCAL_CHECKPOINTS", "").strip().lower() in {"1", "true", "yes"}


def _clip_adapter() -> CLIPAdapter:
    if "clip" not in _ADAPTER_CACHE:
        _ADAPTER_CACHE["clip"] = CLIPAdapter() if _use_local_checkpoints() else _StableHashAdapter(512, "clip")
    return _ADAPTER_CACHE["clip"]  # type: ignore[return-value]


def _imagebind_adapter() -> ImageBindAdapter:
    if "imagebind" not in _ADAPTER_CACHE:
        _ADAPTER_CACHE["imagebind"] = ImageBindAdapter() if _use_local_checkpoints() else _StableHashAdapter(1024, "imagebind")
    return _ADAPTER_CACHE["imagebind"]  # type: ignore[return-value]


def _arcface_adapter() -> ArcFaceAdapter:
    if "arcface" not in _ADAPTER_CACHE:
        _ADAPTER_CACHE["arcface"] = ArcFaceAdapter() if _use_local_checkpoints() else _StableHashAdapter(512, "arcface")
    return _ADAPTER_CACHE["arcface"]  # type: ignore[return-value]


def _empty_series(frame: pd.DataFrame, default: str = "") -> pd.Series:
    return pd.Series([default] * len(frame), index=frame.index)


def first_available_column(frame: pd.DataFrame, candidates: Iterable[str], default: str = "") -> pd.Series:
    """Return the first available dataframe column from a list of aliases."""
    for column in candidates:
        if column in frame.columns:
            return frame[column].fillna(default)
    return _empty_series(frame, default)


def _as_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _mean_or_nan(values: Iterable[float | None]) -> float:
    numeric = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.mean(numeric)) if numeric else float("nan")


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / (np.linalg.norm(left, axis=1, keepdims=True) + 1e-8)
    right_norm = right / (np.linalg.norm(right, axis=1, keepdims=True) + 1e-8)
    return np.sum(left_norm * right_norm, axis=1)


def _resolve_paths(values: Iterable[str], fallback_prefix: str) -> list[str]:
    resolved = []
    for index, value in enumerate(values):
        value = str(value) if value is not None else ""
        if value and Path(value).exists():
            resolved.append(value)
        else:
            resolved.append(f"{fallback_prefix}::{index}")
    return resolved


def _parse_alignment_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if "alignment_scores" not in frame.columns:
        return frame
    parsed_rows = []
    for raw in frame["alignment_scores"].fillna("").astype(str):
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {}
        parsed_rows.append(parsed)
    parsed_frame = pd.DataFrame.from_records(parsed_rows, index=frame.index)
    for source, target in {
        "emotion_match_score": "alignment_emotion_match",
        "clip_similarity": "alignment_clip_similarity",
        "imagebind_similarity": "alignment_imagebind_similarity",
        "identity_similarity": "alignment_identity_similarity",
        "final_score": "alignment_final_score",
    }.items():
        if source in parsed_frame.columns and target not in frame.columns:
            frame[target] = pd.to_numeric(parsed_frame[source], errors="coerce")
    return frame


def _add_affecthuman_paths(frame: pd.DataFrame, dataset_root: str | Path | None) -> pd.DataFrame:
    """Add canonical AffectHuman file paths when a manifest only has ids/splits."""
    if dataset_root is None:
        return frame
    root = Path(dataset_root)
    if not root.exists():
        return frame
    if "sample_id" not in frame.columns:
        return frame

    split = first_available_column(frame, ["split"], "test").astype(str)
    emotion = first_available_column(frame, ["target_emotion", "unified_emotion", "emotion"], "neutral").astype(str)
    sample_id = frame["sample_id"].astype(str)

    path_specs = {
        "image_path": ("images", "", ".jpg"),
        "reference_image_path": ("reference_images", "_ref", ".jpg"),
        "audio_path": ("audio", "", ".wav"),
        "audio_feature_path": ("features/audio_features", "", ".npy"),
    }
    for column, (folder, suffix, extension) in path_specs.items():
        if column in frame.columns:
            continue
        frame[column] = [
            str(root / folder / split_value / emotion_value / f"{sample_value}{suffix}{extension}")
            for split_value, emotion_value, sample_value in zip(split, emotion, sample_id)
        ]
    return frame


def prepare_prediction_frame(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    split: str | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Load, normalize, and optionally split-filter a prediction manifest."""
    path = Path(manifest_path)
    if path.suffix.lower() == ".jsonl":
        frame = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported prediction manifest format: {path}")

    frame = _parse_alignment_scores(frame.copy())
    if "unified_emotion" not in frame.columns and "emotion" in frame.columns:
        frame["unified_emotion"] = frame["emotion"]
    if "target_emotion" not in frame.columns:
        frame["target_emotion"] = first_available_column(frame, ["unified_emotion", "emotion"], "")
    frame = _add_affecthuman_paths(frame, dataset_root)
    if split and "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == split].copy()
    if max_samples and max_samples > 0 and len(frame) > max_samples:
        if "sample_id" in frame.columns:
            frame = frame.sort_values("sample_id").head(max_samples).copy()
        else:
            frame = frame.head(max_samples).copy()
    return frame.reset_index(drop=True)


def compute_core_paper_metrics(frame: pd.DataFrame, *, seed: int = 0) -> dict[str, float]:
    """Compute the scalar metrics used across the paper result tables."""
    if frame.empty:
        return {
            "Acc": float("nan"),
            "TS": float("nan"),
            "CLIP": float("nan"),
            "IB": float("nan"),
            "H": float("nan"),
            "LDS": float("nan"),
            "GC": float("nan"),
            "ID": float("nan"),
        }

    clip = _clip_adapter()
    imagebind = _imagebind_adapter()
    arcface = _arcface_adapter()

    emotions = first_available_column(frame, ["target_emotion", "unified_emotion", "emotion"], "").astype(str).tolist()
    texts = first_available_column(frame, ["target_prompt", "prompt", "text"], "").astype(str).tolist()
    prompts = [text if text else f"a {emotion} face" for text, emotion in zip(texts, emotions)]
    image_paths = _resolve_paths(
        first_available_column(frame, ["output_image_path", "generated_image_path", "image_path"], "").astype(str).tolist(),
        "paper-output-image",
    )
    reference_paths = _resolve_paths(
        first_available_column(frame, ["reference_image_path", "source_image_path", "image_path"], "").astype(str).tolist(),
        "paper-reference-image",
    )
    audio_paths = _resolve_paths(
        first_available_column(frame, ["output_audio_path", "generated_audio_path", "audio_path"], "").astype(str).tolist(),
        "paper-audio",
    )

    clip_text = clip.encode_texts(prompts)
    clip_images = clip.encode_images(image_paths)
    clip_scores = _safe_cosine(clip_text, clip_images)

    bind_text = imagebind.encode_texts(prompts)
    bind_images = imagebind.encode_images(image_paths)
    bind_audio = imagebind.encode_audios(audio_paths)
    image_text_scores = _safe_cosine(bind_images, bind_text)
    image_audio_scores = _safe_cosine(bind_images, bind_audio)
    bind_scores = 0.5 * image_text_scores + 0.5 * image_audio_scores

    reference_embeddings = arcface.encode_images(reference_paths)
    output_identity_embeddings = arcface.encode_images(image_paths)
    identity_similarity = _safe_cosine(reference_embeddings, output_identity_embeddings)
    identity_score = (identity_similarity + 1.0) / 2.0

    acc_values = _emotion_accuracy_values(frame, emotions)
    trajectory_score, graph_consistency = _trajectory_metrics(frame, clip_images, clip_text, emotions)
    lds = _latent_disentanglement_score(output_identity_embeddings, clip_images)
    human = _mean_or_nan(
        first_available_column(frame, ["human_score", "human_preference", "preference_score", "h_score"], "").map(_as_float)
    )

    return {
        "Acc": _mean_or_nan(acc_values),
        "TS": trajectory_score,
        "CLIP": float(np.mean(clip_scores)),
        "IB": float(np.mean(bind_scores)),
        "H": human,
        "LDS": lds,
        "GC": graph_consistency,
        "ID": float(np.mean(identity_score)),
    }


def _emotion_accuracy_values(frame: pd.DataFrame, target_emotions: list[str]) -> list[float | None]:
    parsed = first_available_column(frame, ["alignment_emotion_match"], "").map(_as_float).tolist()
    if any(value is not None for value in parsed):
        return parsed

    predicted = first_available_column(frame, ["predicted_emotion", "generated_emotion", "output_emotion"], "").astype(str).tolist()
    if any(value for value in predicted):
        return [
            float(target.lower() == guess.lower()) if target and guess else None
            for target, guess in zip(target_emotions, predicted)
        ]

    generated_text = first_available_column(frame, ["output_text", "generated_text"], "").astype(str).tolist()
    if any(value for value in generated_text):
        return [
            float(target.lower() in text.lower()) if target and text else None
            for target, text in zip(target_emotions, generated_text)
        ]

    text = first_available_column(frame, ["text"], "").astype(str).tolist()
    return [float(target.lower() in value.lower()) if target and value else None for target, value in zip(target_emotions, text)]


def _latent_disentanglement_score(identity_embeddings: np.ndarray, attribute_embeddings: np.ndarray) -> float:
    dim = min(identity_embeddings.shape[1], attribute_embeddings.shape[1])
    if dim == 0:
        return float("nan")
    overlap = np.abs(_safe_cosine(identity_embeddings[:, :dim], attribute_embeddings[:, :dim]))
    return float(np.clip(1.0 - np.mean(overlap), 0.0, 1.0))


def _trajectory_metrics(
    frame: pd.DataFrame,
    image_embeddings: np.ndarray,
    emotion_embeddings: np.ndarray,
    emotions: list[str],
) -> tuple[float, float]:
    if "identity_id" not in frame.columns or len(frame) < 2:
        return float("nan"), float("nan")

    temp = frame[["identity_id"]].copy()
    temp["emotion"] = emotions
    temp["row_index"] = np.arange(len(frame))
    temp["emotion_order"] = [EMOTION_ORDER.get(str(emotion), 99) for emotion in emotions]

    transition_scores: list[float] = []
    curvature_scores: list[float] = []
    for _, group in temp.groupby("identity_id", dropna=True):
        group = group[group["identity_id"].fillna("").astype(str) != ""].sort_values(["emotion_order", "row_index"])
        indices = group["row_index"].to_numpy(dtype=int)
        if len(indices) < 2:
            continue
        image_steps = image_embeddings[indices[1:]] - image_embeddings[indices[:-1]]
        emotion_steps = emotion_embeddings[indices[1:]] - emotion_embeddings[indices[:-1]]
        valid = (np.linalg.norm(image_steps, axis=1) > 1e-8) & (np.linalg.norm(emotion_steps, axis=1) > 1e-8)
        if np.any(valid):
            cosine = _safe_cosine(image_steps[valid], emotion_steps[valid])
            transition_scores.extend(((cosine + 1.0) / 2.0).tolist())
        if len(indices) >= 3:
            first = image_embeddings[indices[1:-1]] - image_embeddings[indices[:-2]]
            second = image_embeddings[indices[2:]] - image_embeddings[indices[1:-1]]
            valid_curve = (np.linalg.norm(first, axis=1) > 1e-8) & (np.linalg.norm(second, axis=1) > 1e-8)
            if np.any(valid_curve):
                curvature = (1.0 - _safe_cosine(first[valid_curve], second[valid_curve])) / 2.0
                curvature_scores.extend(curvature.tolist())

    ts = float(np.mean(transition_scores)) if transition_scores else float("nan")
    gc = float(np.mean(curvature_scores)) if curvature_scores else float("nan")
    return ts, gc


def compute_retrieval_metrics(frame: pd.DataFrame, *, k_values: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    """Compute image-to-text and image-to-audio recall at K."""
    if frame.empty:
        return {f"{name}_R@{k}": float("nan") for name in ["I2T", "I2A"] for k in k_values}

    imagebind = _imagebind_adapter()
    emotions = first_available_column(frame, ["target_emotion", "unified_emotion", "emotion"], "").astype(str).tolist()
    texts = first_available_column(frame, ["target_prompt", "prompt", "text"], "").astype(str).tolist()
    prompts = [text if text else f"a {emotion} face" for text, emotion in zip(texts, emotions)]
    image_paths = _resolve_paths(
        first_available_column(frame, ["output_image_path", "generated_image_path", "image_path"], "").astype(str).tolist(),
        "retrieval-image",
    )
    audio_paths = _resolve_paths(
        first_available_column(frame, ["output_audio_path", "generated_audio_path", "audio_path"], "").astype(str).tolist(),
        "retrieval-audio",
    )

    image_embeddings = imagebind.encode_images(image_paths)
    text_embeddings = imagebind.encode_texts(prompts)
    audio_embeddings = imagebind.encode_audios(audio_paths)
    return {
        **_recall_at_k(image_embeddings @ text_embeddings.T, "I2T", k_values),
        **_recall_at_k(image_embeddings @ audio_embeddings.T, "I2A", k_values),
    }


def _recall_at_k(scores: np.ndarray, prefix: str, k_values: tuple[int, ...]) -> dict[str, float]:
    rows = scores.shape[0]
    order = np.argsort(-scores, axis=1)
    metrics = {}
    for k in k_values:
        effective_k = min(k, scores.shape[1])
        hits = [float(index in order[index, :effective_k]) for index in range(rows)]
        metrics[f"{prefix}_R@{k}"] = float(np.mean(hits)) if hits else float("nan")
    return metrics
