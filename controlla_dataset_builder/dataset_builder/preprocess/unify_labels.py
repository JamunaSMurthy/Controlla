"""Unified emotion label mapping helpers."""

from __future__ import annotations

from dataset_builder.constants import (
    AFFECTNET_LABEL_MAP,
    CREMAD_LABEL_MAP,
    IEMOCAP_LABEL_MAP,
    RAFDB_LABEL_MAP,
    RAVDESS_LABEL_MAP,
    RELATED_EMOTION_MAP,
)


def canonicalize_emotion_label(value: str) -> str:
    lowered = value.strip().lower().replace("_", " ")
    lowered = " ".join(lowered.split())
    if lowered in RELATED_EMOTION_MAP:
        return RELATED_EMOTION_MAP[lowered]
    raise ValueError(f"Unsupported emotion label: {value}")


def infer_emotion_from_valence_arousal(valence: float, arousal: float) -> str:
    """Simple VAD-to-discrete heuristic for text-only corpora like EmoBank."""
    if valence <= 2.4:
        return "angry" if arousal >= 3.2 else "sad"
    if valence >= 3.4:
        return "happy"
    if arousal >= 3.6:
        return "surprised"
    return "neutral"


def map_dataset_emotion(dataset_name: str, raw_label: str | int | None) -> str | None:
    if raw_label is None:
        return None

    dataset = dataset_name.strip().lower()
    text_label = str(raw_label).strip()
    if not text_label:
        return None

    if dataset == "affectnet":
        if text_label.isdigit():
            return AFFECTNET_LABEL_MAP[int(text_label)]
        return canonicalize_emotion_label(text_label)
    if dataset == "rafdb":
        return RAFDB_LABEL_MAP[int(text_label)]
    if dataset == "cremad":
        return CREMAD_LABEL_MAP[text_label.upper()]
    if dataset == "iemocap":
        return IEMOCAP_LABEL_MAP.get(text_label.lower())
    if dataset == "ravdess":
        return RAVDESS_LABEL_MAP[text_label.zfill(2)]
    return canonicalize_emotion_label(text_label)


def normalize_emotion(
    dataset_name: str,
    raw_label: str | int | None = None,
    *,
    valence: float | None = None,
    arousal: float | None = None,
) -> str:
    mapped = map_dataset_emotion(dataset_name, raw_label)
    if mapped is not None:
        return mapped
    if valence is None or arousal is None:
        raise ValueError(f"Unable to normalize emotion for {dataset_name}")
    return infer_emotion_from_valence_arousal(float(valence), float(arousal))