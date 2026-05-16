"""Emotion matching between modality pools."""

from __future__ import annotations

from typing import Any


RELATED_EMOTION_GROUPS: tuple[set[str], ...] = (
    {"happy", "surprised"},
    {"sad", "fearful"},
    {"angry", "disgusted", "contemptuous"},
)


def related_emotions(emotion: str) -> set[str]:
    for group in RELATED_EMOTION_GROUPS:
        if emotion in group:
            return set(group)
    return {emotion}


def score_emotion_match(
    target_emotion: str,
    candidate_emotion: str,
    *,
    exact_emotion_only: bool = False,
    allow_related_emotions: bool = True,
) -> float:
    target = target_emotion.strip().lower()
    candidate = candidate_emotion.strip().lower()
    if target == candidate:
        return 1.0
    if exact_emotion_only:
        return 0.0
    if allow_related_emotions and candidate in related_emotions(target):
        return 0.72
    if allow_related_emotions and "neutral" in {target, candidate}:
        return 0.35
    return 0.0


def match_emotion(
    target_emotion: str,
    candidates: list[dict[str, Any]],
    *,
    emotion_key: str = "unified_emotion",
    exact_emotion_only: bool = False,
    allow_related_emotions: bool = True,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        score = score_emotion_match(
            target_emotion,
            str(candidate[emotion_key]),
            exact_emotion_only=exact_emotion_only,
            allow_related_emotions=allow_related_emotions,
        )
        if score <= 0.0:
            continue
        scored_candidates.append({**candidate, "emotion_match_score": score})
    scored_candidates.sort(key=lambda item: item["emotion_match_score"], reverse=True)
    return scored_candidates[:top_k] if top_k is not None else scored_candidates