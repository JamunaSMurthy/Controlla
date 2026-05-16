"""Weighted fusion of alignment scores."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dataset_builder.config import load_yaml_config
from dataset_builder.schemas import AlignmentScores


class AlignmentWeights(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emotion: float = 0.45
    identity: float = 0.25
    clip: float = 0.20
    imagebind: float = 0.10
    text_audio: float = 0.10


class AlignmentThresholds(BaseModel):
    model_config = ConfigDict(extra="ignore")

    minimum_final_score: float = 0.55
    minimum_emotion_score: float = 0.50
    minimum_identity_score: float = 0.40


class AlignmentMatchingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    exact_emotion_only: bool = False
    allow_related_emotions: bool = True
    allow_pseudo_identity: bool = True
    normalize_scores: bool = True


class AlignmentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project_root: str
    weights: AlignmentWeights = Field(default_factory=AlignmentWeights)
    thresholds: AlignmentThresholds = Field(default_factory=AlignmentThresholds)
    matching: AlignmentMatchingConfig = Field(default_factory=AlignmentMatchingConfig)


def load_alignment_config(config_path: str = "configs/alignment.yaml") -> AlignmentConfig:
    return AlignmentConfig.model_validate(load_yaml_config(config_path))


def _normalize_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def fuse_scores(
    *,
    emotion_match_score: float,
    identity_similarity: float,
    clip_similarity: float,
    imagebind_similarity: float,
    text_audio_similarity: float | None = None,
    config: AlignmentConfig,
) -> AlignmentScores:
    if text_audio_similarity is None:
        text_audio_similarity = (clip_similarity + imagebind_similarity) / 2.0

    if config.matching.normalize_scores:
        emotion_match_score = _normalize_score(emotion_match_score)
        identity_similarity = _normalize_score(identity_similarity)
        clip_similarity = _normalize_score(clip_similarity)
        imagebind_similarity = _normalize_score(imagebind_similarity)
        text_audio_similarity = _normalize_score(text_audio_similarity)

    weight_sum = max(
        config.weights.emotion
        + config.weights.identity
        + config.weights.clip
        + config.weights.imagebind
        + config.weights.text_audio,
        1e-8,
    )
    final_score = (
        config.weights.emotion * emotion_match_score
        + config.weights.identity * identity_similarity
        + config.weights.clip * clip_similarity
        + config.weights.imagebind * imagebind_similarity
        + config.weights.text_audio * text_audio_similarity
    ) / weight_sum
    return AlignmentScores(
        emotion_match_score=emotion_match_score,
        identity_similarity=identity_similarity,
        clip_similarity=clip_similarity,
        imagebind_similarity=imagebind_similarity,
        image_text_similarity=clip_similarity,
        image_audio_similarity=imagebind_similarity,
        text_audio_similarity=text_audio_similarity,
        final_score=final_score,
    )


def passes_thresholds(scores: AlignmentScores, config: AlignmentConfig, *, has_reference_image: bool) -> bool:
    if scores.final_score < config.thresholds.minimum_final_score:
        return False
    if scores.emotion_match_score < config.thresholds.minimum_emotion_score:
        return False
    if has_reference_image and scores.identity_similarity < config.thresholds.minimum_identity_score:
        return False
    return True
