"""Step 4 tests for text generation and label harmonization."""

from dataset_builder.preprocess.text_sources import (
    TextGenerationConfig,
    build_conditioning_text,
    clean_text,
    generate_text,
    has_meaningful_text,
    harmonize_record_label,
    select_template,
)
from dataset_builder.schemas import NormalizedRecord


def test_clean_text_normalizes_spacing_and_quotes() -> None:
    assert clean_text('  Hello  ""world""  ') == 'Hello "world"'


def test_template_selection_is_deterministic() -> None:
    templates = {"happy": ["a", "b", "c"]}
    assert select_template("happy", "sample-1", templates) == select_template("happy", "sample-1", templates)
    assert generate_text(None, "happy", sample_id="sample-1", templates=templates) in templates["happy"]


def test_build_conditioning_text_uses_hybrid_mode_for_meaningful_text() -> None:
    config = TextGenerationConfig(project_root=".")
    text, origin = build_conditioning_text("I finally made it", "happy", "template", config)
    assert text.endswith("Emotional tone: happy.")
    assert origin == "hybrid"


def test_has_meaningful_text_rejects_punctuation_only() -> None:
    assert has_meaningful_text("...") is False
    assert has_meaningful_text("Okay") is True


def test_harmonize_record_label_prefers_dataset_mapping_and_vad_fallback() -> None:
    iemocap_record = NormalizedRecord(
        source_dataset="iemocap",
        sample_id="demo-1",
        audio_path="/tmp/audio.wav",
        raw_label="fru",
        unified_emotion="neutral",
    )
    emobank_record = NormalizedRecord(
        source_dataset="emobank",
        sample_id="demo-2",
        text="A sentence",
        raw_label="vad_heuristic",
        unified_emotion="neutral",
        valence=3.6,
        arousal=3.2,
    )
    assert harmonize_record_label(iemocap_record) == "angry"
    assert harmonize_record_label(emobank_record) == "happy"