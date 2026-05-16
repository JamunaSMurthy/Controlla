"""Integration tests for leakage-safe group-based splitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import split_builder.build_groups as build_groups_module
import split_builder.similarity_utils as similarity_utils_module
import split_builder.soft_overlap as soft_overlap_module
import split_builder.validate_splits as validate_splits_module

from split_builder.assign_splits import assign_groups_to_splits
from split_builder.build_groups import build_groups
from split_builder.config import load_config
from split_builder.load_data import load_metadata
from split_builder.main import run_stage_diagnostics
from split_builder.repair_splits import repair_split_assignments
from split_builder.save_outputs import save_split_outputs
from split_builder.soft_overlap import analyze_soft_overlaps
from split_builder.validate_splits import validate_split_assignments


def _write_image(path: Path, color: int) -> None:
    Image.new("RGB", (12, 12), color=(color, color, color)).save(path)


def _write_checker_image(path: Path) -> None:
    image = Image.new("RGB", (12, 12), color=(255, 255, 255))
    pixels = image.load()
    for row in range(12):
        for col in range(12):
            if (row + col) % 2 == 0:
                pixels[col, row] = (0, 0, 0)
    image.save(path)


def _write_near_duplicate_image(path: Path, color: int, tweak: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (12, 12), color=(color, color, color))
    pixels = image.load()
    pixels[0, 0] = tweak
    image.save(path)


def test_split_builder_groups_related_samples_and_prevents_identity_leakage(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_c = tmp_path / "c.jpg"
    _write_image(image_a, 80)
    _write_image(image_b, 80)
    _write_checker_image(image_c)

    face_a = tmp_path / "face_a.npy"
    face_b = tmp_path / "face_b.npy"
    audio_a = tmp_path / "audio_a.npy"
    audio_b = tmp_path / "audio_b.npy"
    np.save(face_a, np.ones(8, dtype=np.float32))
    np.save(face_b, np.ones(8, dtype=np.float32))
    np.save(audio_a, np.ones(8, dtype=np.float32))
    np.save(audio_b, np.ones(8, dtype=np.float32))

    metadata_path = tmp_path / "metadata.jsonl"
    records = [
        {
            "sample_id": "s1",
            "dataset_source": "affectnet",
            "image_path": str(image_a),
            "audio_embedding_path": str(audio_a),
            "arcface_embedding_path": str(face_a),
            "identity_id": "id1",
            "speaker_id": "sp1",
            "text": "same text",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "s2",
            "dataset_source": "ffhq",
            "image_path": str(image_b),
            "audio_embedding_path": str(audio_b),
            "arcface_embedding_path": str(face_b),
            "identity_id": "id1",
            "speaker_id": "sp1",
            "text": "same text",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "s3",
            "dataset_source": "iemocap",
            "image_path": str(image_c),
            "text": "different text",
            "unified_emotion": "sad",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    assert groups_result["num_groups"] == 2
    grouped_sizes = sorted(group["size"] for group in groups_result["groups"])
    assert grouped_sizes == [1, 2]
    assert groups_result["hard_group_report"]["num_hard_groups"] == 2
    assert groups_result["hard_group_report"]["edge_counts_by_rule"]["identity_id"] == 1

    assignment_result = assign_groups_to_splits(groups_result["groups"], config)
    outputs = save_split_outputs(config.output_dir, assignment_result["assigned_groups"], assignment_result["dropped_groups"])
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})
    validation = validate_split_assignments(assigned_records, config)

    assert validation["identity_overlap_count"] == 0
    assert validation["speaker_overlap_count"] == 0
    assert Path(outputs["train"]).exists()
    assert Path(outputs["grouped_metadata"]).exists()


def test_text_duplicates_are_soft_warnings_and_small_groups_can_be_repaired(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_c = tmp_path / "c.jpg"
    _write_image(image_a, 40)
    _write_checker_image(image_b)
    _write_image(image_c, 180)

    audio_a = tmp_path / "audio_a.npy"
    audio_b = tmp_path / "audio_b.npy"
    audio_c = tmp_path / "audio_c.npy"
    shared_audio = np.ones(8, dtype=np.float32)
    np.save(audio_a, shared_audio)
    np.save(audio_b, shared_audio)
    np.save(audio_c, shared_audio)

    metadata_path = tmp_path / "soft_metadata.jsonl"
    records = [
        {
            "sample_id": "g1_a",
            "dataset_source": "affectnet",
            "image_path": str(image_a),
            "audio_feature_path": str(audio_a),
            "identity_id": "identity_group",
            "text": "Repeated sentence for soft warning checks.",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "g1_b",
            "dataset_source": "affectnet",
            "image_path": str(image_b),
            "audio_feature_path": str(audio_b),
            "identity_id": "identity_group",
            "text": "Repeated sentence for soft warning checks.",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "g2_a",
            "dataset_source": "affectnet",
            "image_path": str(image_c),
            "audio_feature_path": str(audio_c),
            "text": "Repeated sentence for soft warning checks.",
            "unified_emotion": "happy",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "out_soft"),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    config.enable_face_similarity = False
    config.enable_image_similarity = False
    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    assert groups_result["num_groups"] == 2
    assert groups_result["hard_group_report"]["edge_counts_by_rule"]["identity_id"] == 1

    ordered_groups = sorted(groups_result["groups"], key=lambda group: (group["size"], group["group_id"]))
    small_group = {**ordered_groups[0], "assigned_split": "val"}
    large_group = {**ordered_groups[1], "assigned_split": "train"}
    assignment_result = {
        "assigned_groups": [large_group, small_group],
        "dropped_groups": [],
        "split_counts": {"train": large_group["size"], "val": small_group["size"], "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    soft_report = analyze_soft_overlaps(assigned_records, config)
    assert soft_report["warning_counts_by_type"]["text_duplicate"] >= 1
    assert soft_report["warning_counts_by_type"]["audio_similarity"] >= 1

    repair_result = repair_split_assignments(assignment_result, soft_report, config)
    repaired_split_by_group = {group["group_id"]: group["assigned_split"] for group in repair_result["assigned_groups"]}
    assert repaired_split_by_group[small_group["group_id"]] == "train"
    assert repair_result["repair_summary"]["num_reassigned_groups"] == 1


def test_stage_diagnostics_emit_all_stage_checkpoints(tmp_path: Path) -> None:
    image_a = tmp_path / "diag_a.jpg"
    image_b = tmp_path / "diag_b.jpg"
    _write_image(image_a, 60)
    _write_checker_image(image_b)

    metadata_path = tmp_path / "diag_metadata.jsonl"
    records = [
        {
            "sample_id": "diag_1",
            "dataset_source": "affectnet",
            "image_path": str(image_a),
            "audio_path": str(image_a),
            "identity_id": "diag_identity",
            "text": "diagnostic sample one",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "diag_2",
            "dataset_source": "affectnet",
            "image_path": str(image_b),
            "audio_path": str(image_b),
            "text": "diagnostic sample two",
            "unified_emotion": "sad",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    diagnostics = run_stage_diagnostics(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "diag_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        config_path=None,
        seed=0,
        batch_size=16,
        bottleneck_threshold_seconds=None,
    )

    assert diagnostics["stopped_at_stage"] is None
    assert [checkpoint["stage"] for checkpoint in diagnostics["checkpoints"]] == [
        "load_config+load_metadata",
        "build_groups",
        "assign_groups_to_splits",
        "analyze_soft_overlaps",
        "repair_split_assignments",
        "validate_split_assignments",
    ]


def test_build_groups_memoizes_file_hashes_by_path(tmp_path: Path, monkeypatch) -> None:
    shared_image = tmp_path / "shared_image.jpg"
    shared_audio = tmp_path / "shared_audio.wav"
    shared_reference = tmp_path / "shared_reference.jpg"
    _write_image(shared_image, 90)
    shared_audio.write_bytes(b"shared-audio-bytes")
    _write_checker_image(shared_reference)

    call_counts: dict[str, int] = {}
    original_sha256_file = build_groups_module.sha256_file

    def counting_sha256_file(path: str | None) -> str | None:
        if path:
            call_counts[path] = call_counts.get(path, 0) + 1
        return original_sha256_file(path)

    monkeypatch.setattr(build_groups_module, "sha256_file", counting_sha256_file)

    metadata_path = tmp_path / "memoized_metadata.jsonl"
    records = [
        {
            "sample_id": "memo_1",
            "dataset_source": "affectnet",
            "image_path": str(shared_image),
            "reference_image_path": str(shared_reference),
            "audio_path": str(shared_audio),
            "identity_id": "memo_identity_1",
            "text": "memoized record one",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "memo_2",
            "dataset_source": "affectnet",
            "image_path": str(shared_image),
            "reference_image_path": str(shared_reference),
            "audio_path": str(shared_audio),
            "identity_id": "memo_identity_2",
            "text": "memoized record two",
            "unified_emotion": "sad",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "memoized_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)

    assert groups_result["hard_group_report"]["hard_edge_diagnostics"]["memoized_file_hash_paths"] == 3
    assert call_counts == {
        str(shared_image): 1,
        str(shared_audio): 1,
        str(shared_reference): 1,
    }


def test_build_groups_skips_high_fanout_audio_hash_buckets(tmp_path: Path) -> None:
    shared_audio = tmp_path / "shared_audio.wav"
    shared_audio.write_bytes(b"fanout-audio-bytes")

    metadata_path = tmp_path / "audio_fanout_metadata.jsonl"
    records = []
    for index in range(3):
        image_path = tmp_path / f"fanout_{index}.jpg"
        _write_image(image_path, 30 + index * 40)
        records.append(
            {
                "sample_id": f"fanout_{index}",
                "dataset_source": "iemocap",
                "image_path": str(image_path),
                "audio_path": str(shared_audio),
                "identity_id": f"identity_{index}",
                "text": f"unique text {index}",
                "unified_emotion": "sad",
            }
        )
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "fanout_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    config.grouping.audio_hash_max_group_size_for_hard_link = 2
    config.grouping.audio_hash_max_distinct_identities_for_hard_link = 1
    config.grouping.audio_hash_max_distinct_texts_for_hard_link = 1

    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)

    assert groups_result["num_groups"] == 3
    diagnostics = groups_result["hard_group_report"]["hard_edge_diagnostics"]
    assert diagnostics["skipped_bucket_counts_by_rule"]["audio_hash"] == 1
    assert diagnostics["audio_hash_skip_reasons"]["group_size"] == 1
    assert diagnostics["skipped_audio_hash_record_count"] == 3
    assert groups_result["hard_group_report"]["edge_counts_by_rule"].get("audio_hash", 0) == 0


def test_soft_overlap_aggregates_text_duplicates_and_quarantines_ambiguous_audio_reuse(tmp_path: Path) -> None:
    shared_audio = tmp_path / "shared_quarantine_audio.wav"
    shared_audio.write_bytes(b"ambiguous-audio-quarantine")

    image_a = tmp_path / "quarantine_a.jpg"
    image_b = tmp_path / "quarantine_b.jpg"
    image_c = tmp_path / "quarantine_c.jpg"
    _write_image(image_a, 20)
    _write_checker_image(image_b)
    _write_image(image_c, 200)

    metadata_path = tmp_path / "quarantine_metadata.jsonl"
    records = [
        {
            "sample_id": "qa_1",
            "dataset_source": "iemocap",
            "image_path": str(image_a),
            "audio_path": str(shared_audio),
            "identity_id": "same_identity",
            "text": "shared duplicate sentence",
            "unified_emotion": "sad",
        },
        {
            "sample_id": "qa_2",
            "dataset_source": "iemocap",
            "image_path": str(image_b),
            "audio_path": str(shared_audio),
            "identity_id": "same_identity",
            "text": "shared duplicate sentence",
            "unified_emotion": "sad",
        },
        {
            "sample_id": "qb_1",
            "dataset_source": "iemocap",
            "image_path": str(image_c),
            "audio_path": str(shared_audio),
            "identity_id": "other_identity",
            "text": "shared duplicate sentence",
            "unified_emotion": "sad",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "quarantine_out"),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    config.enable_face_similarity = False
    config.enable_audio_similarity = False
    config.enable_image_similarity = False
    config.grouping.audio_hash_max_group_size_for_hard_link = 2
    config.grouping.audio_hash_max_distinct_identities_for_hard_link = 1
    config.grouping.audio_hash_max_distinct_texts_for_hard_link = 1
    config.repair.max_drop_group_size = 1
    config.repair.ambiguous_audio_quarantine_min_count = 1

    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    ordered_groups = sorted(groups_result["groups"], key=lambda group: (-group["size"], group["group_id"]))
    large_group = {**ordered_groups[0], "assigned_split": "train"}
    small_group = {**ordered_groups[1], "assigned_split": "val"}
    assignment_result = {
        "assigned_groups": [large_group, small_group],
        "dropped_groups": [],
        "split_counts": {"train": large_group["size"], "val": small_group["size"], "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    soft_report = analyze_soft_overlaps(assigned_records, config)
    assert soft_report["warning_counts_by_type"]["text_duplicate"] == 2
    assert soft_report["warning_counts_by_type"].get("exact_audio_duplicate", 0) == 0
    assert soft_report["warning_counts_by_type"]["ambiguous_audio_reuse"] == 2
    assert small_group["group_id"] in soft_report["quarantine_candidate_groups"]
    assert "exact_text_duplicates" in soft_report["analysis_timings_seconds"]

    repair_result = repair_split_assignments(assignment_result, soft_report, config)
    assert repair_result["repair_summary"]["action_counts_by_reason"]["ambiguous_audio_reuse_quarantine"] == 1
    assert len(repair_result["dropped_groups"]) == 1
    assert repair_result["dropped_groups"][0]["group_id"] == small_group["group_id"]


def test_soft_overlap_uses_phash_surrogates_for_face_and_image_without_embeddings(tmp_path: Path) -> None:
    image_a = tmp_path / "surrogate_a.jpg"
    image_b = tmp_path / "surrogate_b.jpg"
    _write_near_duplicate_image(image_a, 120, (121, 121, 121))
    _write_near_duplicate_image(image_b, 121, (122, 122, 122))

    metadata_path = tmp_path / "surrogate_metadata.jsonl"
    records = [
        {
            "sample_id": "surrogate_1",
            "dataset_source": "affectnet",
            "image_path": str(image_a),
            "text": "first",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "surrogate_2",
            "dataset_source": "rafdb",
            "image_path": str(image_b),
            "text": "second",
            "unified_emotion": "happy",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "surrogate_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    config.enable_audio_similarity = False
    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    assert groups_result["num_groups"] == 2

    assignment_result = {
        "assigned_groups": [
            {**groups_result["groups"][0], "assigned_split": "train"},
            {**groups_result["groups"][1], "assigned_split": "val"},
        ],
        "dropped_groups": [],
        "split_counts": {"train": 1, "val": 1, "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    soft_report = analyze_soft_overlaps(assigned_records, config)
    assert soft_report["warning_counts_by_type"]["face_similarity"] >= 1
    assert soft_report["warning_counts_by_type"]["image_similarity"] >= 1
    assert soft_report["analysis_stats"]["face_similarity_surrogate"]["candidates"] == 2
    assert soft_report["analysis_stats"]["face_similarity_embeddings"]["candidates"] == 0
    assert soft_report["analysis_stats"]["image_similarity_surrogate"]["candidates"] == 2
    assert soft_report["analysis_stats"]["image_similarity_embeddings"]["candidates"] == 0


def test_soft_overlap_reuses_precomputed_phashes_without_reopening_images(tmp_path: Path, monkeypatch) -> None:
    image_a = tmp_path / "cached_soft_a.jpg"
    image_b = tmp_path / "cached_soft_b.jpg"
    _write_near_duplicate_image(image_a, 150, (151, 151, 151))
    _write_near_duplicate_image(image_b, 151, (152, 152, 152))

    metadata_path = tmp_path / "cached_soft_metadata.jsonl"
    records = [
        {
            "sample_id": "cached_soft_1",
            "dataset_source": "affectnet",
            "image_path": str(image_a),
            "text": "cached one",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "cached_soft_2",
            "dataset_source": "rafdb",
            "image_path": str(image_b),
            "text": "cached two",
            "unified_emotion": "happy",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "cached_soft_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    config.enable_audio_similarity = False
    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    assignment_result = {
        "assigned_groups": [
            {**groups_result["groups"][0], "assigned_split": "train"},
            {**groups_result["groups"][1], "assigned_split": "val"},
        ],
        "dropped_groups": [],
        "split_counts": {"train": 1, "val": 1, "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    def _unexpected_phash(_: str | None, size: int = 8) -> str | None:
        raise AssertionError("soft overlap should reuse precomputed phashes")

    monkeypatch.setattr(soft_overlap_module, "image_perceptual_hash", _unexpected_phash)
    soft_report = analyze_soft_overlaps(assigned_records, config)

    assert soft_report["warning_counts_by_type"]["face_similarity"] >= 1
    assert soft_report["warning_counts_by_type"]["image_similarity"] >= 1


def test_validation_reuses_precomputed_hashes_and_phashes(tmp_path: Path, monkeypatch) -> None:
    image_a = tmp_path / "cached_validation_a.jpg"
    image_b = tmp_path / "cached_validation_b.jpg"
    audio_a = tmp_path / "cached_validation_a.wav"
    audio_b = tmp_path / "cached_validation_b.wav"
    _write_image(image_a, 100)
    _write_checker_image(image_b)
    audio_a.write_bytes(b"validation-a")
    audio_b.write_bytes(b"validation-b")

    metadata_path = tmp_path / "cached_validation_metadata.jsonl"
    records = [
        {
            "sample_id": "cached_validation_1",
            "dataset_source": "affectnet",
            "image_path": str(image_a),
            "audio_path": str(audio_a),
            "text": "validation one",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "cached_validation_2",
            "dataset_source": "rafdb",
            "image_path": str(image_b),
            "audio_path": str(audio_b),
            "text": "validation two",
            "unified_emotion": "sad",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "cached_validation_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    assignment_result = {
        "assigned_groups": [
            {**groups_result["groups"][0], "assigned_split": "train"},
            {**groups_result["groups"][1], "assigned_split": "val"},
        ],
        "dropped_groups": [],
        "split_counts": {"train": 1, "val": 1, "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    def _unexpected_validate_phash(_: str | None, size: int = 8) -> str | None:
        raise AssertionError("validation should reuse precomputed image phashes")

    def _unexpected_validate_hash(_: str | None) -> str | None:
        raise AssertionError("validation should reuse precomputed exact hashes")

    monkeypatch.setattr(validate_splits_module, "image_perceptual_hash", _unexpected_validate_phash)
    monkeypatch.setattr(validate_splits_module, "sha256_file", _unexpected_validate_hash)
    monkeypatch.setattr(similarity_utils_module, "image_perceptual_hash", _unexpected_validate_phash)

    validation = validate_split_assignments(assigned_records, config)
    assert validation["non_empty_split_count"] == 2


def test_low_risk_text_only_conflicts_do_not_trigger_generic_drop(tmp_path: Path) -> None:
    image_a = tmp_path / "lowrisk_a.jpg"
    image_b = tmp_path / "lowrisk_b.jpg"
    _write_image(image_a, 70)
    _write_checker_image(image_b)

    metadata_path = tmp_path / "lowrisk_metadata.jsonl"
    records = [
        {
            "sample_id": "lowrisk_1",
            "dataset_source": "affectnet",
            "image_path": str(image_a),
            "text": "shared low risk sentence",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "lowrisk_2",
            "dataset_source": "affectnet",
            "image_path": str(image_b),
            "text": "shared low risk sentence",
            "unified_emotion": "happy",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "lowrisk_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    config.enable_face_similarity = False
    config.enable_audio_similarity = False
    config.enable_image_similarity = False
    config.repair.max_reassign_group_size = 0

    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    assignment_result = {
        "assigned_groups": [
            {**groups_result["groups"][0], "assigned_split": "train"},
            {**groups_result["groups"][1], "assigned_split": "val"},
        ],
        "dropped_groups": [],
        "split_counts": {"train": 1, "val": 1, "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    soft_report = analyze_soft_overlaps(assigned_records, config)
    repair_result = repair_split_assignments(assignment_result, soft_report, config)

    assert soft_report["warning_counts_by_type"]["text_duplicate"] >= 1
    assert repair_result["repair_summary"]["num_dropped_groups"] == 0
    assert repair_result["repair_summary"]["num_reassigned_groups"] == 0


def test_train_optimized_assignment_biases_remainder_to_train(tmp_path: Path) -> None:
    metadata_path = tmp_path / "train_optimized_metadata.jsonl"
    records = []
    for index in range(10):
        image_path = tmp_path / f"train_optimized_{index}.jpg"
        _write_image(image_path, 20 + index * 10)
        records.append(
            {
                "sample_id": f"train_opt_{index}",
                "dataset_source": "affectnet",
                "image_path": str(image_path),
                "text": f"sample {index}",
                "unified_emotion": "happy",
            }
        )
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "train_optimized_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    config.assignment.strategy = "train_optimized"
    config.assignment.val_test_target_reserve_factor = 1.04
    config.enable_face_similarity = False
    config.enable_audio_similarity = False
    config.enable_image_similarity = False

    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    assignment_result = assign_groups_to_splits(groups_result["groups"], config)

    assert assignment_result["split_counts"] == {"train": 6, "val": 2, "test": 2}


def test_ambiguous_audio_reuse_prefers_train_reassign_before_drop(tmp_path: Path) -> None:
    shared_audio = tmp_path / "shared_train_pref_audio.wav"
    shared_audio.write_bytes(b"ambiguous-train-preference")

    image_a = tmp_path / "train_pref_a.jpg"
    image_b = tmp_path / "train_pref_b.jpg"
    image_c = tmp_path / "train_pref_c.jpg"
    _write_image(image_a, 35)
    _write_checker_image(image_b)
    _write_image(image_c, 210)

    metadata_path = tmp_path / "train_pref_metadata.jsonl"
    records = [
        {
            "sample_id": "ta_1",
            "dataset_source": "iemocap",
            "image_path": str(image_a),
            "audio_path": str(shared_audio),
            "identity_id": "shared_identity",
            "text": "shared sentence",
            "unified_emotion": "sad",
        },
        {
            "sample_id": "ta_2",
            "dataset_source": "iemocap",
            "image_path": str(image_b),
            "audio_path": str(shared_audio),
            "identity_id": "shared_identity",
            "text": "shared sentence",
            "unified_emotion": "sad",
        },
        {
            "sample_id": "tb_1",
            "dataset_source": "iemocap",
            "image_path": str(image_c),
            "audio_path": str(shared_audio),
            "identity_id": "other_identity",
            "text": "shared sentence",
            "unified_emotion": "sad",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "train_pref_out"),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    config.enable_face_similarity = False
    config.enable_audio_similarity = False
    config.enable_image_similarity = False
    config.grouping.audio_hash_max_group_size_for_hard_link = 2
    config.grouping.audio_hash_max_distinct_identities_for_hard_link = 1
    config.grouping.audio_hash_max_distinct_texts_for_hard_link = 1
    config.repair.max_drop_group_size = 1
    config.repair.ambiguous_audio_quarantine_min_count = 1
    config.repair.prefer_train_for_ambiguous_reassign = True
    config.repair.enable_generic_soft_reassign = False
    config.repair.enable_generic_soft_conflict_drop = False

    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    ordered_groups = sorted(groups_result["groups"], key=lambda group: (-group["size"], group["group_id"]))
    large_group = {**ordered_groups[0], "assigned_split": "train"}
    small_group = {**ordered_groups[1], "assigned_split": "val"}
    assignment_result = {
        "assigned_groups": [large_group, small_group],
        "dropped_groups": [],
        "split_counts": {"train": large_group["size"], "val": small_group["size"], "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    soft_report = analyze_soft_overlaps(assigned_records, config)
    repair_result = repair_split_assignments(assignment_result, soft_report, config)
    repaired_splits = {group["group_id"]: group["assigned_split"] for group in repair_result["assigned_groups"]}

    assert repair_result["repair_summary"]["action_counts_by_reason"]["ambiguous_audio_reuse_train_reassign"] == 1
    assert repaired_splits[small_group["group_id"]] == "train"
    assert len(repair_result["dropped_groups"]) == 0


def test_disabling_generic_soft_actions_avoids_text_only_repair_churn(tmp_path: Path) -> None:
    image_a = tmp_path / "generic_off_a.jpg"
    image_b = tmp_path / "generic_off_b.jpg"
    _write_image(image_a, 55)
    _write_checker_image(image_b)

    metadata_path = tmp_path / "generic_off_metadata.jsonl"
    records = [
        {
            "sample_id": "generic_off_1",
            "dataset_source": "affectnet",
            "image_path": str(image_a),
            "text": "shared generic sentence",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "generic_off_2",
            "dataset_source": "affectnet",
            "image_path": str(image_b),
            "text": "shared generic sentence",
            "unified_emotion": "happy",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "generic_off_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    config.enable_face_similarity = False
    config.enable_audio_similarity = False
    config.enable_image_similarity = False
    config.repair.enable_generic_soft_reassign = False
    config.repair.enable_generic_soft_conflict_drop = False

    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    assignment_result = {
        "assigned_groups": [
            {**groups_result["groups"][0], "assigned_split": "train"},
            {**groups_result["groups"][1], "assigned_split": "val"},
        ],
        "dropped_groups": [],
        "split_counts": {"train": 1, "val": 1, "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    soft_report = analyze_soft_overlaps(assigned_records, config)
    repair_result = repair_split_assignments(assignment_result, soft_report, config)

    assert soft_report["warning_counts_by_type"]["text_duplicate"] >= 1
    assert repair_result["repair_summary"]["num_reassigned_groups"] == 0
    assert repair_result["repair_summary"]["num_dropped_groups"] == 0


def test_exact_audio_duplicate_component_repair_coalesces_small_actionable_groups(tmp_path: Path) -> None:
    config = load_config(
        input_path=str(tmp_path / "manual_audio.jsonl"),
        output_dir=str(tmp_path / "manual_audio_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    config.enable_face_similarity = False
    config.enable_audio_similarity = False
    config.enable_image_similarity = False

    group_a = {
        "group_id": "ga",
        "size": 1,
        "assigned_split": "train",
        "samples": [
            {
                "sample_id": "audio_exact_1",
                "dataset_source": "iemocap",
                "audio_hash": "shared-audio-hash",
                "_audio_hash": "shared-audio-hash",
                "identity_id": "id_a",
                "text": "unique text a",
                "unified_emotion": "happy",
            }
        ],
    }
    group_b = {
        "group_id": "gb",
        "size": 1,
        "assigned_split": "val",
        "samples": [
            {
                "sample_id": "audio_exact_2",
                "dataset_source": "iemocap",
                "audio_hash": "shared-audio-hash",
                "_audio_hash": "shared-audio-hash",
                "identity_id": "id_b",
                "text": "unique text b",
                "unified_emotion": "sad",
            }
        ],
    }
    assignment_result = {
        "assigned_groups": [group_a, group_b],
        "dropped_groups": [],
        "split_counts": {"train": 1, "val": 1, "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    soft_report = analyze_soft_overlaps(assigned_records, config)

    assert soft_report["warning_counts_by_type"]["exact_audio_duplicate"] == 2
    assert soft_report["warning_counts_by_type"].get("ambiguous_audio_reuse", 0) == 0
    assert len(soft_report["repair_components_by_type"]["exact_audio_duplicate"]) == 1

    repair_result = repair_split_assignments(assignment_result, soft_report, config)

    assert repair_result["repair_summary"]["action_counts_by_reason"]["exact_audio_duplicate_drop"] == 1
    assert len(repair_result["dropped_groups"]) == 1


def test_image_similarity_component_repair_coalesces_near_duplicate_groups(tmp_path: Path) -> None:
    image_a = tmp_path / "repair_image_a.jpg"
    image_b = tmp_path / "repair_image_b.jpg"
    _write_near_duplicate_image(image_a, 135, (136, 136, 136))
    _write_near_duplicate_image(image_b, 136, (137, 137, 137))

    metadata_path = tmp_path / "repair_image_metadata.jsonl"
    records = [
        {
            "sample_id": "repair_image_1",
            "dataset_source": "affectnet",
            "image_path": str(image_a),
            "text": "repair image one",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "repair_image_2",
            "dataset_source": "affectnet",
            "image_path": str(image_b),
            "text": "repair image two",
            "unified_emotion": "happy",
        },
    ]
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config = load_config(
        input_path=str(metadata_path),
        output_dir=str(tmp_path / "repair_image_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    config.enable_face_similarity = False
    config.enable_audio_similarity = False
    samples = load_metadata(metadata_path)
    groups_result = build_groups(samples, config)
    assignment_result = {
        "assigned_groups": [
            {**groups_result["groups"][0], "assigned_split": "train"},
            {**groups_result["groups"][1], "assigned_split": "val"},
        ],
        "dropped_groups": [],
        "split_counts": {"train": 1, "val": 1, "test": 0},
    }
    assigned_records = []
    for group in assignment_result["assigned_groups"]:
        for sample in group["samples"]:
            assigned_records.append({**sample, "split": group["assigned_split"], "group_id": group["group_id"]})

    soft_report = analyze_soft_overlaps(assigned_records, config)
    repair_result = repair_split_assignments(assignment_result, soft_report, config)

    assert soft_report["warning_counts_by_type"]["image_similarity"] >= 1
    assert repair_result["repair_summary"]["action_counts_by_reason"]["image_similarity_drop"] == 1
    assert len(repair_result["dropped_groups"]) == 1


def test_validation_separates_ambiguous_audio_reuse_from_actionable_audio_leakage(tmp_path: Path) -> None:
    config = load_config(
        input_path=str(tmp_path / "validation_audio.jsonl"),
        output_dir=str(tmp_path / "validation_audio_out"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    config.grouping.audio_hash_max_group_size_for_hard_link = 2
    config.grouping.audio_hash_max_distinct_identities_for_hard_link = 1
    config.grouping.audio_hash_max_distinct_texts_for_hard_link = 1

    records = [
        {
            "sample_id": "ambiguous_audio_train",
            "group_id": "g_train",
            "split": "train",
            "audio_hash": "shared-audio-hash",
            "_audio_hash": "shared-audio-hash",
            "identity_id": "id_a",
            "text": "text a",
            "unified_emotion": "happy",
        },
        {
            "sample_id": "ambiguous_audio_val",
            "group_id": "g_val",
            "split": "val",
            "audio_hash": "shared-audio-hash",
            "_audio_hash": "shared-audio-hash",
            "identity_id": "id_b",
            "text": "text b",
            "unified_emotion": "sad",
        },
        {
            "sample_id": "clean_audio_test",
            "group_id": "g_test",
            "split": "test",
            "audio_hash": "other-audio-hash",
            "_audio_hash": "other-audio-hash",
            "identity_id": "id_c",
            "text": "text c",
            "unified_emotion": "neutral",
        },
    ]

    validation = validate_split_assignments(records, config)

    assert validation["exact_audio_duplicate_overlap"] == 1
    assert validation["actionable_exact_audio_duplicate_overlap"] == 0
    assert validation["ambiguous_audio_reuse_overlap_count"] == 1
    assert validation["status"] == "PASS"