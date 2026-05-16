"""Aligned tuple construction."""

from __future__ import annotations

from dataset_builder.schemas import AlignedSample, AlignmentScores, DatasetSources, ModalityMask


def build_tuple(
    *,
    image_record: dict,
    text_record: dict,
    split: str,
    alignment_scores: AlignmentScores,
    audio_record: dict | None = None,
    reference_record: dict | None = None,
) -> AlignedSample:
    sample_id = "__".join(
        [
            image_record["sample_id"],
            text_record["sample_id"],
            audio_record["sample_id"] if audio_record else "noaudio",
            reference_record["sample_id"] if reference_record else "noref",
        ]
    )
    raw_emotion_labels = {
        "image_source": str(image_record.get("raw_label") or image_record.get("unified_emotion")),
        "text_source": str(text_record.get("raw_label") or text_record.get("unified_emotion")),
    }
    if audio_record is not None:
        raw_emotion_labels["audio_source"] = str(audio_record.get("unified_emotion"))

    return AlignedSample(
        sample_id=sample_id,
        split=split,
        dataset_sources=DatasetSources(
            image_source=image_record["source_dataset"],
            audio_source=audio_record["source_dataset"] if audio_record else None,
            text_source=text_record["source_dataset"],
        ),
        image_path=image_record["image_path"],
        reference_image_path=reference_record["image_path"] if reference_record else None,
        audio_path=audio_record.get("audio_path") if audio_record else None,
        audio_feature_path=audio_record.get("audio_feature_path") if audio_record else None,
        text=text_record.get("conditioning_text") or text_record.get("text"),
        raw_text=text_record.get("raw_text"),
        unified_emotion=image_record["unified_emotion"],
        raw_emotion_labels=raw_emotion_labels,
        identity_id=image_record.get("identity_id"),
        speaker_id=audio_record.get("speaker_id") if audio_record else text_record.get("speaker_id"),
        valence=text_record.get("valence"),
        arousal=text_record.get("arousal"),
        modality_mask=ModalityMask(
            has_image=True,
            has_audio=audio_record is not None,
            has_text=bool(text_record.get("conditioning_text") or text_record.get("text")),
            has_reference_image=reference_record is not None,
        ),
        alignment_scores=alignment_scores,
    )