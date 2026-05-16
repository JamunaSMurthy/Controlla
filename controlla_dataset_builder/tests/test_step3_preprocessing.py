"""Step 3 regression tests for raw dataset preprocessing helpers."""

from pathlib import Path

from dataset_builder.preprocess.common import resolve_existing_path
from dataset_builder.preprocess.prepare_cremad import parse_cremad_filename
from dataset_builder.preprocess.prepare_iemocap import parse_evaluation_line
from dataset_builder.preprocess.prepare_ravdess import parse_ravdess_filename
from dataset_builder.preprocess.unify_labels import infer_emotion_from_valence_arousal, normalize_emotion


def test_resolve_existing_path_tolerates_trailing_space(tmp_path: Path) -> None:
    expected = tmp_path / "RAVDESS "
    expected.mkdir()
    resolved = resolve_existing_path(tmp_path, "RAVDESS")
    assert resolved == expected.resolve()


def test_parse_iemocap_evaluation_line_extracts_expected_fields() -> None:
    parsed = parse_evaluation_line("[38.9650 - 43.5900]\tSes01F_impro01_F006\tfru\t[2.0000, 3.5000, 3.5000]")
    assert parsed is not None
    assert parsed["turn"] == "Ses01F_impro01_F006"
    assert parsed["label"] == "fru"
    assert parsed["valence"] == "2.0000"


def test_parse_cremad_filename_extracts_speaker_and_emotion() -> None:
    parsed = parse_cremad_filename("1039_IEO_HAP_LO.wav")
    assert parsed["speaker_id"] == "1039"
    assert parsed["emotion_code"] == "HAP"


def test_parse_ravdess_filename_extracts_actor_and_emotion() -> None:
    parsed = parse_ravdess_filename("03-01-05-02-02-01-08.wav")
    assert parsed["emotion"] == "05"
    assert parsed["actor"] == "08"


def test_vad_heuristic_and_dataset_normalization_behave_as_expected() -> None:
    assert infer_emotion_from_valence_arousal(2.2, 3.5) == "angry"
    assert normalize_emotion("iemocap", "fru") == "angry"