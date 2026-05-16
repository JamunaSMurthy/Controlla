"""Validated data schemas for normalized records and aligned tuples."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import ALIGNMENT_SCORE_NAMES, EMOTION_TAXONOMY


class DatasetSources(BaseModel):
    """Track which datasets contributed each modality in an aligned sample."""

    model_config = ConfigDict(extra="forbid")

    image_source: str
    audio_source: str | None = None
    text_source: str | None = None


class ModalityMask(BaseModel):
    """Availability flags for each modality."""

    model_config = ConfigDict(extra="forbid")

    has_image: bool = True
    has_audio: bool = False
    has_text: bool = False
    has_reference_image: bool = False


class AlignmentScores(BaseModel):
    """Container for alignment quality scores."""

    model_config = ConfigDict(extra="allow")

    emotion_match_score: float = 0.0
    clip_similarity: float = 0.0
    imagebind_similarity: float = 0.0
    image_text_similarity: float = 0.0
    image_audio_similarity: float = 0.0
    text_audio_similarity: float = 0.0
    identity_similarity: float = 0.0
    final_score: float = 0.0

    @field_validator(*ALIGNMENT_SCORE_NAMES, mode="before")
    @classmethod
    def _cast_numeric_score(cls, value: Any) -> float:
        return float(value or 0.0)


class NormalizedRecord(BaseModel):
    """Normalized single-source record before multimodal alignment."""

    model_config = ConfigDict(extra="allow")

    source_dataset: str
    sample_id: str
    split: Literal["train", "val", "test"] | None = None
    image_path: str | None = None
    reference_image_path: str | None = None
    audio_path: str | None = None
    audio_feature_path: str | None = None
    text: str | None = None
    raw_text: str | None = None
    raw_label: str | None = None
    unified_emotion: str
    identity_id: str | None = None
    speaker_id: str | None = None
    valence: float | None = None
    arousal: float | None = None

    @field_validator("unified_emotion")
    @classmethod
    def _validate_emotion(cls, value: str) -> str:
        lowered = value.strip().lower()
        if lowered not in EMOTION_TAXONOMY:
            raise ValueError(f"Unsupported unified emotion: {value}")
        return lowered

    @model_validator(mode="after")
    def _require_at_least_one_modality(self) -> "NormalizedRecord":
        if not any([self.image_path, self.audio_path, self.text]):
            raise ValueError("A normalized record must contain at least one modality")
        return self


class AlignedSample(BaseModel):
    """Final aligned multimodal tuple used by Controlla training and evaluation."""

    model_config = ConfigDict(extra="allow")

    sample_id: str
    split: Literal["train", "val", "test"]
    dataset_sources: DatasetSources
    image_path: str
    reference_image_path: str | None = None
    audio_path: str | None = None
    audio_feature_path: str | None = None
    text: str | None = None
    raw_text: str | None = None
    unified_emotion: str
    raw_emotion_labels: dict[str, str] = Field(default_factory=dict)
    identity_id: str | None = None
    speaker_id: str | None = None
    valence: float | None = None
    arousal: float | None = None
    modality_mask: ModalityMask
    alignment_scores: AlignmentScores = Field(default_factory=AlignmentScores)

    @field_validator("unified_emotion")
    @classmethod
    def _validate_tuple_emotion(cls, value: str) -> str:
        lowered = value.strip().lower()
        if lowered not in EMOTION_TAXONOMY:
            raise ValueError(f"Unsupported unified emotion: {value}")
        return lowered


class SplitRatios(BaseModel):
    """Reproducible split configuration."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    prevent_identity_leakage: bool = True
    stratify_by_emotion: bool = True

    @model_validator(mode="after")
    def _validate_sum(self) -> "SplitRatios":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, found {total}")
        return self
