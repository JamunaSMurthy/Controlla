"""Optional export of worst leaked sample pairs for qualitative inspection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _sample_index(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {sample["sample_id"]: sample for sample in samples}


def _copy_if_exists(path: str | None, destination: Path) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.exists() or not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _identity_examples(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report["identity_leakage"]["explicit_identity_id"].get("examples", [])


def _image_examples(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report["duplicate_detection"]["image_duplicates"]["near_duplicates"].get("examples", [])


def _audio_examples(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report["duplicate_detection"]["audio_duplicates"]["near_duplicates"].get("examples", [])


def _text_examples(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report["duplicate_detection"]["text_duplicates"]["fuzzy_duplicates"].get("examples", [])


def export_leakage_examples(
    samples: list[dict[str, Any]],
    report: dict[str, Any],
    output_dir: str | Path,
    *,
    max_examples_per_type: int = 10,
) -> dict[str, Any]:
    sample_index = _sample_index(samples)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[dict[str, Any]]] = {
        "identity_examples": [],
        "image_duplicate_examples": [],
        "audio_duplicate_examples": [],
        "text_duplicate_examples": [],
    }

    for index, item in enumerate(_identity_examples(report)[:max_examples_per_type], start=1):
        example_dir = target_dir / "identity_examples" / f"example_{index:03d}"
        exported = {"metadata": item, "files": []}
        for sample_id in item.get("sample_ids", [])[:2]:
            sample = sample_index.get(sample_id)
            if sample is None:
                continue
            for field in ("image_path", "reference_image_path"):
                copied = _copy_if_exists(sample.get(field), example_dir / f"{sample_id}_{field}{Path(sample.get(field) or '').suffix}")
                if copied:
                    exported["files"].append(copied)
        manifest["identity_examples"].append(exported)

    for index, item in enumerate(_image_examples(report)[:max_examples_per_type], start=1):
        example_dir = target_dir / "image_duplicate_examples" / f"example_{index:03d}"
        exported = {"metadata": item, "files": []}
        exported["files"].append(_copy_if_exists(item.get("left_image_path"), example_dir / f"left{Path(item.get('left_image_path') or '').suffix}"))
        exported["files"].append(_copy_if_exists(item.get("right_image_path"), example_dir / f"right{Path(item.get('right_image_path') or '').suffix}"))
        exported["files"] = [path for path in exported["files"] if path]
        manifest["image_duplicate_examples"].append(exported)

    for index, item in enumerate(_audio_examples(report)[:max_examples_per_type], start=1):
        example_dir = target_dir / "audio_duplicate_examples" / f"example_{index:03d}"
        exported = {"metadata": item, "files": []}
        for side in ("left", "right"):
            sample = sample_index.get(item.get(f"{side}_sample_id"))
            if sample is None:
                continue
            copied = _copy_if_exists(sample.get("audio_path"), example_dir / f"{side}{Path(sample.get('audio_path') or '').suffix}")
            if copied:
                exported["files"].append(copied)
        manifest["audio_duplicate_examples"].append(exported)

    for index, item in enumerate(_text_examples(report)[:max_examples_per_type], start=1):
        example_dir = target_dir / "text_duplicate_examples" / f"example_{index:03d}"
        example_dir.mkdir(parents=True, exist_ok=True)
        left_txt = example_dir / "left.txt"
        right_txt = example_dir / "right.txt"
        left_txt.write_text(item.get("left_text_preview", ""), encoding="utf-8")
        right_txt.write_text(item.get("right_text_preview", ""), encoding="utf-8")
        manifest["text_duplicate_examples"].append(
            {
                "metadata": item,
                "files": [str(left_txt), str(right_txt)],
            }
        )

    manifest_path = target_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": str(manifest_path), **{key: len(value) for key, value in manifest.items()}}