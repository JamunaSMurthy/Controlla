"""Text sourcing and generation helpers built on normalized Step 3 indexes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.schemas import NormalizedRecord
from dataset_builder.utils.hashing import stable_hash
from dataset_builder.utils.io import write_json, write_jsonl
from dataset_builder.utils.paths import ensure_directory

from .unify_labels import normalize_emotion


class TextGenerationSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "emotion_template_hybrid"
    fallback_to_template: bool = True
    caption_model_name: str | None = None
    use_attributes_if_available: bool = True


class TextGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project_root: str
    text_output_root: str = "outputs/text"
    text_generation: TextGenerationSettings = Field(default_factory=TextGenerationSettings)
    templates: dict[str, list[str]] = Field(default_factory=dict)


def load_text_generation_settings(config_path: str | Path = "configs/text_generation.yaml") -> TextGenerationConfig:
    return TextGenerationConfig.model_validate(load_yaml_config(config_path))


def _normalize_csv_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def load_normalized_records(path: str | Path) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            payload: dict[str, Any] = {key: _normalize_csv_value(value) for key, value in row.items()}
            records.append(NormalizedRecord.model_validate(payload))
    return records


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace("\r", " ").replace("\n", " ").replace('""', '"').strip()
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def has_meaningful_text(text: str | None) -> bool:
    if text is None:
        return False
    alnum_count = sum(character.isalnum() for character in text)
    return alnum_count >= 3


def select_template(emotion: str, sample_id: str, templates: dict[str, list[str]]) -> str:
    choices = templates.get(emotion) or templates.get("neutral") or [f"A portrait with a {emotion} expression."]
    index = int(stable_hash(f"{emotion}:{sample_id}"), 16) % len(choices)
    return choices[index]


def generate_text(image_path: str | None, emotion: str, *, sample_id: str = "", templates: dict[str, list[str]] | None = None) -> str:
    del image_path
    return select_template(emotion, sample_id or emotion, templates or {})


def harmonize_record_label(record: NormalizedRecord) -> str:
    if record.raw_label not in {None, "", "unlabeled", "vad_heuristic"}:
        try:
            return normalize_emotion(record.source_dataset, record.raw_label)
        except Exception:
            pass
    if record.valence is not None and record.arousal is not None:
        return normalize_emotion(record.source_dataset, None, valence=record.valence, arousal=record.arousal)
    return record.unified_emotion


def build_conditioning_text(
    text: str | None,
    emotion: str,
    template_text: str,
    config: TextGenerationConfig,
) -> tuple[str, str]:
    cleaned_text = clean_text(text)
    informative = has_meaningful_text(cleaned_text)
    mode = config.text_generation.mode.strip().lower()

    if informative and mode == "original_only":
        return cleaned_text, "original"
    if informative and mode == "emotion_template_hybrid":
        punctuation = "" if cleaned_text.endswith((".", "!", "?")) else "."
        return f"{cleaned_text}{punctuation} Emotional tone: {emotion}.", "hybrid"
    if informative:
        return cleaned_text, "original"
    if config.text_generation.fallback_to_template:
        return template_text, "template"
    return cleaned_text or template_text, "template"


def build_text_manifest_records(records: list[NormalizedRecord], config: TextGenerationConfig) -> list[dict[str, Any]]:
    manifest_records: list[dict[str, Any]] = []
    for record in records:
        harmonized_emotion = harmonize_record_label(record)
        template_text = generate_text(
            record.image_path,
            harmonized_emotion,
            sample_id=record.sample_id,
            templates=config.templates,
        )
        conditioning_text, text_origin = build_conditioning_text(record.text or record.raw_text, harmonized_emotion, template_text, config)
        plain_text = clean_text(record.text or record.raw_text)
        manifest_records.append(
            {
                "source_dataset": record.source_dataset,
                "sample_id": record.sample_id,
                "split": record.split,
                "image_path": record.image_path,
                "audio_path": record.audio_path,
                "identity_id": record.identity_id,
                "speaker_id": record.speaker_id,
                "raw_label": record.raw_label,
                "original_unified_emotion": record.unified_emotion,
                "unified_emotion": harmonized_emotion,
                "text": plain_text or template_text,
                "conditioning_text": conditioning_text,
                "raw_text": clean_text(record.raw_text),
                "text_origin": text_origin,
                "valence": record.valence,
                "arousal": record.arousal,
            }
        )
    return manifest_records


def _write_csv(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_dataset",
        "sample_id",
        "split",
        "image_path",
        "audio_path",
        "identity_id",
        "speaker_id",
        "raw_label",
        "original_unified_emotion",
        "unified_emotion",
        "text",
        "conditioning_text",
        "raw_text",
        "text_origin",
        "valence",
        "arousal",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: "" if record.get(key) is None else record.get(key) for key in fieldnames})
    return path


def write_text_outputs(dataset_name: str, records: list[dict[str, Any]], output_root: str | Path) -> dict[str, Any]:
    dataset_dir = ensure_directory(Path(output_root) / dataset_name)
    csv_path = _write_csv(records, dataset_dir / f"{dataset_name}_text_manifest.csv")
    jsonl_path = write_jsonl(records, dataset_dir / f"{dataset_name}_text_manifest.jsonl")

    emotion_counts: dict[str, int] = {}
    origin_counts: dict[str, int] = {}
    for record in records:
        emotion_counts[record["unified_emotion"]] = emotion_counts.get(record["unified_emotion"], 0) + 1
        origin_counts[record["text_origin"]] = origin_counts.get(record["text_origin"], 0) + 1

    summary = {
        "dataset": dataset_name,
        "num_records": len(records),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "emotion_counts": emotion_counts,
        "text_origin_counts": origin_counts,
    }
    write_json(summary, dataset_dir / f"{dataset_name}_text_summary.json")
    return summary


def default_text_output_root(config: TextGenerationConfig, project_root: Path) -> Path:
    root = Path(config.text_output_root)
    return root.resolve() if root.is_absolute() else (project_root / root).resolve()