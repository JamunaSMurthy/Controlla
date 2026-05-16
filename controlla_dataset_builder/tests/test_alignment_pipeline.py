"""Step 6 tests for alignment scoring and tuple construction."""

from __future__ import annotations

import numpy as np

from dataset_builder.align.embedding_matcher import text_to_embedding
from dataset_builder.align.emotion_matcher import score_emotion_match
from dataset_builder.align.identity_matcher import cosine_similarity
from dataset_builder.align.multimodal_consistency import MultimodalConsistencyScorer
from dataset_builder.align.score_fusion import AlignmentConfig, fuse_scores, passes_thresholds
from dataset_builder.align.tuple_builder import build_tuple


def test_emotion_match_scores_related_classes_lower_than_exact() -> None:
    exact = score_emotion_match("happy", "happy")
    related = score_emotion_match("happy", "surprised")
    mismatch = score_emotion_match("happy", "angry")
    assert exact == 1.0
    assert 0.0 < related < exact
    assert mismatch == 0.0


def test_text_embedding_is_deterministic() -> None:
    first = text_to_embedding("A happy face.")
    second = text_to_embedding("A happy face.")
    assert np.allclose(first, second)
    assert first.shape == (64,)


def test_cosine_similarity_handles_dimension_mismatch() -> None:
    left = np.ones(56, dtype=np.float32)
    right = np.ones(64, dtype=np.float32)
    assert cosine_similarity(left, right) == 1.0


def test_score_fusion_and_thresholds_accept_strong_match() -> None:
    config = AlignmentConfig(project_root=".")
    scores = fuse_scores(
        emotion_match_score=1.0,
        identity_similarity=0.9,
        clip_similarity=0.8,
        imagebind_similarity=0.7,
        text_audio_similarity=0.6,
        config=config,
    )
    assert scores.image_text_similarity == scores.clip_similarity
    assert scores.image_audio_similarity == scores.imagebind_similarity
    assert scores.text_audio_similarity == 0.6
    assert scores.final_score > 0.8
    assert passes_thresholds(scores, config, has_reference_image=True) is True


def test_multimodal_consistency_scores_all_three_terms() -> None:
    scorer = MultimodalConsistencyScorer(enable_real_clip=False, enable_real_imagebind=False)
    scores = scorer.score(
        image_record={"sample_id": "img", "image_path": "/tmp/img.jpg", "_vector": np.ones(8, dtype=np.float32)},
        text_record={"sample_id": "txt", "text": "a happy face", "_vector": np.ones(8, dtype=np.float32)},
        audio_record={"sample_id": "aud", "audio_path": "/tmp/aud.wav", "_vector": np.ones(8, dtype=np.float32)},
    )
    assert scores.image_text_similarity == 1.0
    assert scores.image_audio_similarity == 1.0
    assert scores.text_audio_similarity == 1.0
    assert scores.clip_backend == "deterministic_proxy"
    assert scores.imagebind_backend == "deterministic_proxy"


def test_build_tuple_constructs_expected_modalities() -> None:
    sample = build_tuple(
        image_record={
            "sample_id": "img1",
            "source_dataset": "affectnet",
            "image_path": "/tmp/image.jpg",
            "raw_label": "anger",
            "unified_emotion": "angry",
            "identity_id": "id1",
        },
        text_record={
            "sample_id": "txt1",
            "source_dataset": "iemocap",
            "conditioning_text": "A person is angry.",
            "raw_text": "A person is angry.",
            "raw_label": "fru",
            "unified_emotion": "angry",
            "speaker_id": "spk1",
        },
        audio_record={
            "sample_id": "aud1",
            "source_dataset": "iemocap",
            "audio_path": "/tmp/audio.wav",
            "audio_feature_path": "/tmp/audio.npy",
            "unified_emotion": "angry",
            "speaker_id": "spk1",
        },
        reference_record={
            "sample_id": "ref1",
            "image_path": "/tmp/ref.jpg",
        },
        split="train",
        alignment_scores=fuse_scores(
            emotion_match_score=1.0,
            identity_similarity=0.9,
            clip_similarity=0.8,
            imagebind_similarity=0.7,
            config=AlignmentConfig(project_root="."),
        ),
    )
    assert sample.modality_mask.has_audio is True
    assert sample.modality_mask.has_reference_image is True
    assert sample.unified_emotion == "angry"
