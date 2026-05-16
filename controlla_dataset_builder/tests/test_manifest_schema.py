"""Schema validation tests for aligned tuple records."""

from dataset_builder.schemas import AlignedSample, DatasetSources, ModalityMask


def test_aligned_sample_schema_accepts_expected_fields() -> None:
    sample = AlignedSample(
        sample_id="demo-1",
        split="train",
        dataset_sources=DatasetSources(image_source="AffectNet", audio_source="IEMOCAP", text_source="EmoBank"),
        image_path="/tmp/image.jpg",
        reference_image_path="/tmp/ref.jpg",
        audio_path="/tmp/audio.wav",
        audio_feature_path="/tmp/audio.npy",
        text="A portrait of a person with a happy expression.",
        raw_text="happy portrait",
        unified_emotion="happy",
        raw_emotion_labels={"image_source": "happy", "audio_source": "excited"},
        identity_id="id-1",
        speaker_id="spk-1",
        valence=0.9,
        arousal=0.8,
        modality_mask=ModalityMask(has_image=True, has_audio=True, has_text=True, has_reference_image=True),
    )
    assert sample.unified_emotion == "happy"
    assert sample.modality_mask.has_reference_image is True