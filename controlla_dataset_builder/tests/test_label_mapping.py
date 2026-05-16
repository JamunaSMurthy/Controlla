"""Foundational tests for emotion taxonomy constants."""

from dataset_builder.constants import EMOTION_TAXONOMY, IEMOCAP_LABEL_MAP, RELATED_EMOTION_MAP


def test_taxonomy_has_expected_size() -> None:
    assert len(EMOTION_TAXONOMY) == 8


def test_explicit_related_mappings_present() -> None:
    assert RELATED_EMOTION_MAP["frustration"] == "angry"
    assert RELATED_EMOTION_MAP["excited"] == "happy"
    assert RELATED_EMOTION_MAP["calm"] == "neutral"
    assert IEMOCAP_LABEL_MAP["fru"] == "angry"