"""Core constants used throughout the dataset builder."""

from __future__ import annotations

from typing import Final


EMOTION_TAXONOMY: Final[list[str]] = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    "contemptuous",
]

RELATED_EMOTION_MAP: Final[dict[str, str]] = {
    "neutral": "neutral",
    "calm": "neutral",
    "bored": "neutral",
    "happy": "happy",
    "joy": "happy",
    "joyful": "happy",
    "excited": "happy",
    "sad": "sad",
    "angry": "angry",
    "anger": "angry",
    "frustration": "angry",
    "frustrated": "angry",
    "fear": "fearful",
    "fearful": "fearful",
    "scared": "fearful",
    "surprise": "surprised",
    "surprised": "surprised",
    "disgust": "disgusted",
    "disgusted": "disgusted",
    "contempt": "contemptuous",
    "contemptuous": "contemptuous",
}

AFFECTNET_LABEL_MAP: Final[dict[int, str]] = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "surprised",
    4: "fearful",
    5: "disgusted",
    6: "angry",
    7: "contemptuous",
}

RAFDB_LABEL_MAP: Final[dict[int, str]] = {
    1: "surprised",
    2: "fearful",
    3: "disgusted",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral",
}

CREMAD_LABEL_MAP: Final[dict[str, str]] = {
    "ANG": "angry",
    "DIS": "disgusted",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

IEMOCAP_LABEL_MAP: Final[dict[str, str]] = {
    "neu": "neutral",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "ang": "angry",
    "fru": "angry",
    "fea": "fearful",
    "sur": "surprised",
    "dis": "disgusted",
    "con": "contemptuous",
}

RAVDESS_LABEL_MAP: Final[dict[str, str]] = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised",
}

OUTPUT_DATASET_MODES: Final[list[str]] = [
    "full_multimodal",
    "image_text",
    "image_audio",
    "text_audio_image_reference",
    "eval_bench",
]

NORMALIZED_INDEX_COLUMNS: Final[list[str]] = [
    "source_dataset",
    "sample_id",
    "split",
    "image_path",
    "reference_image_path",
    "audio_path",
    "audio_feature_path",
    "text",
    "raw_text",
    "raw_label",
    "unified_emotion",
    "identity_id",
    "speaker_id",
    "valence",
    "arousal",
]

ALIGNMENT_SCORE_NAMES: Final[list[str]] = [
    "emotion_match_score",
    "clip_similarity",
    "imagebind_similarity",
    "image_text_similarity",
    "image_audio_similarity",
    "text_audio_similarity",
    "identity_similarity",
    "final_score",
]
