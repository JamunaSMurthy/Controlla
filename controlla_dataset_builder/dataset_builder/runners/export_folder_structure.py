"""Convert aligned JSONL dataset to folder-based structure for dataset release.

This module converts the JSONL-based aligned dataset into a hierarchical folder structure
suitable for distribution to the research community and public dataset repositories.
The output mirrors the organization of established datasets (AffectNet, CelebHQ, etc.)
with multimodal data organized by modality and split.

Output structure:
    Controlla-Dataset/
    ├── README.md
    ├── metadata/
    │   ├── dataset_manifest.csv
    │   ├── dataset_manifest.json
    │   └── statistics.json
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── reference_images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── audio/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── text/
    │   ├── annotations.jsonl
    │   └── annotations.csv
    ├── features/
    │   ├── audio_features.hdf5 (or per-split NPY)
    │   └── identity_features.hdf5
    └── splits/
        ├── train_split.txt
        ├── val_split.txt
        └── test_split.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aligned dataset to folder-based structure"
    )
    parser.add_argument(
        "--input-jsonl",
        "--input",
        dest="input_jsonl",
        type=str,
        required=False,
        help="Path to aligned_dataset.jsonl or a split directory containing train/val/test JSONL files",
    )
    parser.add_argument(
        "--output-root",
        "--output",
        dest="output_root",
        type=str,
        required=False,
        help="Root output directory for folder structure",
    )
    parser.add_argument(
        "--use-symlinks",
        action="store_true",
        default=False,
        help="Use symbolic links instead of copying files",
    )
    parser.add_argument(
        "--use-hardlinks",
        action="store_true",
        default=False,
        help="Use hard links for unchanged files instead of copying bytes",
    )
    parser.add_argument(
        "--copy-audio-features",
        action="store_true",
        default=False,
        help="Copy audio feature NPY files to output",
    )
    parser.add_argument(
        "--organize-by-emotion",
        action="store_true",
        default=False,
        help="Create emotion subdirectories within each split",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        default=False,
        help="Remove source dataset identifiers from exported metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print plan without creating files",
    )
    args = parser.parse_args()
    if not args.input_jsonl:
        parser.error("--input-jsonl/--input is required")
    if not args.output_root:
        parser.error("--output-root/--output is required")
    return args


class DatasetExporter:
    """Convert JSONL dataset to folder structure."""

    def __init__(
        self,
        input_jsonl: Path,
        output_root: Path,
        use_symlinks: bool = False,
        use_hardlinks: bool = False,
        copy_audio_features: bool = False,
        organize_by_emotion: bool = False,
        anonymize: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.input_jsonl = Path(input_jsonl)
        self.output_root = Path(output_root)
        self.use_symlinks = use_symlinks
        self.use_hardlinks = use_hardlinks
        self.copy_audio_features = copy_audio_features
        self.organize_by_emotion = organize_by_emotion
        self.anonymize = anonymize
        self.dry_run = dry_run

        self.samples = []
        self._sample_ids: dict[int, str] = {}
        self._identity_ids: dict[str, str] = {}
        self.statistics = {
            "total_samples": 0,
            "split_counts": defaultdict(int),
            "emotion_counts": defaultdict(int),
            "modality_coverage": defaultdict(int),
        }

    def _input_files(self) -> list[Path]:
        """Return JSONL input files in deterministic split order."""
        if self.input_jsonl.is_dir():
            split_files = [self.input_jsonl / f"{split}.jsonl" for split in ("train", "val", "test")]
            existing = [path for path in split_files if path.exists()]
            if existing:
                return existing
            return sorted(self.input_jsonl.glob("*.jsonl"))
        return [self.input_jsonl]

    def load_jsonl(self) -> None:
        """Load JSONL dataset or split directory."""
        input_files = self._input_files()
        if not input_files:
            raise FileNotFoundError(f"No JSONL inputs found under {self.input_jsonl}")

        logger.info("Loading dataset from:")
        for input_file in input_files:
            logger.info(f"  {input_file}")
            with input_file.open("r") as f:
                for line_idx, line in enumerate(f):
                    if line.strip():
                        sample = json.loads(line)
                        if "split" not in sample:
                            sample["split"] = input_file.stem
                        self.samples.append(sample)
                    if (line_idx + 1) % 5000 == 0:
                        logger.info(f"  Loaded {line_idx + 1} rows from {input_file.name}")

        self.statistics["total_samples"] = len(self.samples)
        self._assign_export_ids()
        logger.info(f"Total samples loaded: {len(self.samples)}")

    def _safe_token(self, value: Any, fallback: str = "sample") -> str:
        """Create a filesystem-safe token without leaking path separators."""
        token = str(value or fallback).strip()
        safe = []
        for char in token:
            if char.isalnum() or char in ("-", "_", "."):
                safe.append(char)
            else:
                safe.append("_")
        return "".join(safe).strip("._") or fallback

    def _sample_fingerprint(self, sample: dict[str, Any], index: int) -> str:
        """Build a stable fingerprint from fields that distinguish duplicate sample IDs."""
        payload = {
            "sample_id": sample.get("sample_id", ""),
            "split": sample.get("split", ""),
            "image_path": sample.get("image_path", ""),
            "reference_image_path": sample.get("reference_image_path", ""),
            "audio_path": sample.get("audio_path", ""),
            "audio_feature_path": sample.get("audio_feature_path", ""),
            "text": sample.get("text", ""),
            "index": index,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)

    def _assign_export_ids(self) -> None:
        """Assign unique IDs once so every output file and manifest agrees."""
        used_ids: set[str] = set()
        duplicate_source_ids = defaultdict(int)

        for index, sample in enumerate(self.samples):
            original_id = sample.get("sample_id", f"sample_{index:06d}")
            if self.anonymize:
                digest = hashlib.sha256(self._sample_fingerprint(sample, index).encode()).hexdigest()[:12]
                candidate = f"sample_{digest}"
            else:
                duplicate_source_ids[original_id] += 1
                suffix = "" if duplicate_source_ids[original_id] == 1 else f"_{duplicate_source_ids[original_id]:03d}"
                candidate = f"{self._safe_token(original_id)}{suffix}"

            if candidate in used_ids:
                base = candidate
                counter = 2
                while f"{base}_{counter:03d}" in used_ids:
                    counter += 1
                candidate = f"{base}_{counter:03d}"

            used_ids.add(candidate)
            self._sample_ids[index] = candidate

        for sample in self.samples:
            identity_id = sample.get("identity_id")
            if identity_id and identity_id not in self._identity_ids:
                digest = hashlib.sha256(str(identity_id).encode()).hexdigest()[:12]
                self._identity_ids[str(identity_id)] = f"id_{digest}"

        logger.info(f"Assigned {len(used_ids)} unique export sample IDs")

    def _has_identity_features(self) -> bool:
        """Return whether any sample carries an identity feature file path."""
        return any(sample.get("identity_feature_path") for sample in self.samples)

    def compute_statistics(self) -> None:
        """Compute dataset statistics."""
        for sample in self.samples:
            split = sample.get("split", "train")
            emotion = sample.get("unified_emotion", "unknown")
            modality_mask = sample.get("modality_mask", {})

            self.statistics["split_counts"][split] += 1
            self.statistics["emotion_counts"][emotion] += 1

            # Track modality coverage
            if modality_mask.get("has_image"):
                self.statistics["modality_coverage"]["has_image"] += 1
            if modality_mask.get("has_audio"):
                self.statistics["modality_coverage"]["has_audio"] += 1
            if modality_mask.get("has_text"):
                self.statistics["modality_coverage"]["has_text"] += 1
            if modality_mask.get("has_reference_image"):
                self.statistics["modality_coverage"]["has_reference_image"] += 1

    def create_directories(self) -> None:
        """Create output folder structure."""
        if self.dry_run:
            logger.info("[DRY RUN] Would create directories")
            return

        dirs = [
            self.output_root,
            self.output_root / "metadata",
            self.output_root / "images" / "train",
            self.output_root / "images" / "val",
            self.output_root / "images" / "test",
            self.output_root / "reference_images" / "train",
            self.output_root / "reference_images" / "val",
            self.output_root / "reference_images" / "test",
            self.output_root / "audio" / "train",
            self.output_root / "audio" / "val",
            self.output_root / "audio" / "test",
            self.output_root / "text",
            self.output_root / "features",
            self.output_root / "splits",
        ]
        has_identity_features = self.copy_audio_features and self._has_identity_features()

        if self.copy_audio_features:
            dirs.extend(
                [
                    self.output_root / "features" / "audio_features" / "train",
                    self.output_root / "features" / "audio_features" / "val",
                    self.output_root / "features" / "audio_features" / "test",
                ]
            )
        if has_identity_features:
            dirs.extend(
                [
                    self.output_root / "features" / "identity_features" / "train",
                    self.output_root / "features" / "identity_features" / "val",
                    self.output_root / "features" / "identity_features" / "test",
                ]
            )

        if self.organize_by_emotion:
            emotions = set(s.get("unified_emotion", "unknown") for s in self.samples)
            for split in ["train", "val", "test"]:
                for emotion in emotions:
                    dirs.append(self.output_root / "images" / split / emotion)
                    dirs.append(self.output_root / "reference_images" / split / emotion)
                    dirs.append(self.output_root / "audio" / split / emotion)
                    if self.copy_audio_features:
                        dirs.append(self.output_root / "features" / "audio_features" / split / emotion)
                    if has_identity_features:
                        dirs.append(self.output_root / "features" / "identity_features" / split / emotion)

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {d}")

    def _copy_or_link_file(
        self, src: Path, dst: Path, name: str = "file"
    ) -> bool:
        """Copy or symlink a file."""
        if not src.exists():
            logger.warning(f"Source {name} does not exist: {src}")
            return False

        if self.dry_run:
            logger.debug(f"[DRY RUN] Would copy/link {src} -> {dst}")
            return True

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if self.use_symlinks:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            elif self.use_hardlinks:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.hardlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"Failed to copy/link {name}: {e}")
            return False

    def _copy_image_file(self, src: Path, dst: Path, name: str = "image") -> bool:
        """Export an image as JPEG unless a same-format symlink is safe."""
        if not src.exists():
            logger.warning(f"Source {name} does not exist: {src}")
            return False

        if self.use_symlinks and src.suffix.lower() in {".jpg", ".jpeg"}:
            return self._copy_or_link_file(src, dst, name)

        if not self.use_symlinks and src.suffix.lower() in {".jpg", ".jpeg"}:
            return self._copy_or_link_file(src, dst, name)

        if self.dry_run:
            logger.debug(f"[DRY RUN] Would convert {src} -> {dst}")
            return True

        try:
            from PIL import Image

            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as image:
                image.convert("RGB").save(dst, format="JPEG", quality=95, optimize=True)
            return True
        except Exception as e:
            logger.error(f"Failed to convert {name}: {e}")
            return False

    def _get_sample_id(self, sample: dict[str, Any], index: int | None = None) -> str:
        """Generate unique sample ID."""
        if index is not None:
            return self._sample_ids[index]
        sample_index = self.samples.index(sample)
        return self._sample_ids[sample_index]

    def _get_identity_id(self, sample: dict[str, Any]) -> str:
        """Return raw or anonymized identity ID."""
        identity_id = sample.get("identity_id", "")
        if not identity_id:
            return ""
        if self.anonymize:
            return self._identity_ids.get(str(identity_id), "")
        return str(identity_id)

    def _modality_path(self, modality: str, split: str, sample_id: str, suffix: str, emotion: str | None) -> Path:
        """Build output path for split- and optional emotion-organized files."""
        if self.organize_by_emotion and emotion:
            return self.output_root / modality / split / emotion / f"{sample_id}{suffix}"
        return self.output_root / modality / split / f"{sample_id}{suffix}"

    def _feature_path(self, feature_type: str, split: str, sample_id: str, emotion: str | None) -> Path:
        """Build output path for feature arrays."""
        if self.organize_by_emotion and emotion:
            return self.output_root / "features" / feature_type / split / emotion / f"{sample_id}.npy"
        return self.output_root / "features" / feature_type / split / f"{sample_id}.npy"

    def _get_output_subdir(self, sample: dict[str, Any]) -> Path:
        """Get output subdirectory for sample."""
        base = self.output_root
        split = sample.get("split", "train")

        if self.organize_by_emotion:
            emotion = sample.get("unified_emotion", "unknown")
            return base, split, emotion
        return base, split, None

    def export_samples(self) -> None:
        """Export samples to folder structure."""
        logger.info("Exporting samples to folder structure...")

        success_count = 0
        fail_count = 0

        for idx, sample in enumerate(self.samples):
            sample_id = self._get_sample_id(sample, idx)
            split = sample.get("split", "train")
            emotion = sample.get("unified_emotion", "unknown") if self.organize_by_emotion else None

            # Image
            if sample.get("image_path"):
                img_src = Path(sample["image_path"])
                img_dst = self._modality_path("images", split, sample_id, ".jpg", emotion)

                if self._copy_image_file(img_src, img_dst, "image"):
                    success_count += 1
                else:
                    fail_count += 1

            # Reference image
            if sample.get("reference_image_path"):
                ref_src = Path(sample["reference_image_path"])
                ref_dst = self._modality_path("reference_images", split, sample_id, "_ref.jpg", emotion)

                if self._copy_image_file(ref_src, ref_dst, "reference image"):
                    success_count += 1
                else:
                    fail_count += 1

            # Audio
            if sample.get("audio_path"):
                audio_src = Path(sample["audio_path"])
                audio_dst = self._modality_path("audio", split, sample_id, ".wav", emotion)

                if self._copy_or_link_file(audio_src, audio_dst, "audio"):
                    success_count += 1
                else:
                    fail_count += 1

            # Audio features
            if self.copy_audio_features and sample.get("audio_feature_path"):
                feature_src = Path(sample["audio_feature_path"])
                feature_dst = self._feature_path("audio_features", split, sample_id, emotion)

                if self._copy_or_link_file(feature_src, feature_dst, "audio feature"):
                    success_count += 1
                else:
                    fail_count += 1

            # Identity features, when present in a manifest.
            if self.copy_audio_features and sample.get("identity_feature_path"):
                identity_src = Path(sample["identity_feature_path"])
                identity_dst = self._feature_path("identity_features", split, sample_id, emotion)

                if self._copy_or_link_file(identity_src, identity_dst, "identity feature"):
                    success_count += 1
                else:
                    fail_count += 1

            if (idx + 1) % 1000 == 0:
                logger.info(
                    f"Processed {idx + 1}/{len(self.samples)} samples "
                    f"(success: {success_count}, fail: {fail_count})"
                )

        logger.info(
            f"Export complete: {success_count} successful, {fail_count} failed"
        )

    def export_metadata(self) -> None:
        """Export metadata CSV and JSON."""
        logger.info("Exporting metadata...")

        # Prepare metadata records
        metadata_records = []
        for idx, sample in enumerate(self.samples):
            sample_id = self._get_sample_id(sample, idx)
            split = sample.get("split", "train")
            emotion = sample.get("unified_emotion", "unknown")

            record = {
                "sample_id": sample_id,
                "split": split,
                "emotion": emotion,
                "has_image": sample.get("modality_mask", {}).get("has_image", False),
                "has_audio": sample.get("modality_mask", {}).get("has_audio", False),
                "has_text": sample.get("modality_mask", {}).get("has_text", False),
                "has_reference_image": sample.get("modality_mask", {}).get("has_reference_image", False),
                "text": sample.get("text", ""),
                "raw_text": sample.get("raw_text", ""),
                "valence": sample.get("valence", None),
                "arousal": sample.get("arousal", None),
                "identity_id": self._get_identity_id(sample),
                "alignment_scores": json.dumps(sample.get("alignment_scores", {})),
            }
            alignment_scores = sample.get("alignment_scores", {})
            record["image_text_similarity"] = alignment_scores.get("image_text_similarity", alignment_scores.get("clip_similarity"))
            record["image_audio_similarity"] = alignment_scores.get("image_audio_similarity", alignment_scores.get("imagebind_similarity"))
            record["text_audio_similarity"] = alignment_scores.get("text_audio_similarity")
            record["clip_backend"] = alignment_scores.get("clip_backend")
            record["imagebind_backend"] = alignment_scores.get("imagebind_backend")

            if not self.anonymize:
                record["original_sample_id"] = sample.get("sample_id")
                record["image_source"] = sample.get("dataset_sources", {}).get("image_source")
                record["audio_source"] = sample.get("dataset_sources", {}).get("audio_source")
                record["text_source"] = sample.get("dataset_sources", {}).get("text_source")
                record["image_path"] = sample.get("image_path")
                record["reference_image_path"] = sample.get("reference_image_path")
                record["audio_path"] = sample.get("audio_path")
                record["audio_feature_path"] = sample.get("audio_feature_path")

            metadata_records.append(record)

        # Write CSV
        if not self.dry_run:
            csv_path = self.output_root / "metadata" / "dataset_manifest.csv"
            with csv_path.open("w", newline="") as f:
                if metadata_records:
                    writer = csv.DictWriter(f, fieldnames=metadata_records[0].keys())
                    writer.writeheader()
                    writer.writerows(metadata_records)
            logger.info(f"Wrote metadata CSV: {csv_path}")

            # Write JSON
            json_path = self.output_root / "metadata" / "dataset_manifest.json"
            with json_path.open("w") as f:
                json.dump(metadata_records, f, indent=2)
            logger.info(f"Wrote metadata JSON: {json_path}")

    def export_text_annotations(self) -> None:
        """Export text annotations."""
        logger.info("Exporting text annotations...")

        if self.dry_run:
            logger.info("[DRY RUN] Would export text annotations")
            return

        # JSONL format
        jsonl_path = self.output_root / "text" / "annotations.jsonl"
        with jsonl_path.open("w") as f:
            for idx, sample in enumerate(self.samples):
                sample_id = self._get_sample_id(sample, idx)
                text_record = {
                    "sample_id": sample_id,
                    "split": sample.get("split", "train"),
                    "emotion": sample.get("unified_emotion", "unknown"),
                    "text": sample.get("text", ""),
                    "raw_text": sample.get("raw_text", ""),
                }
                f.write(json.dumps(text_record) + "\n")
        logger.info(f"Wrote text annotations (JSONL): {jsonl_path}")

        # CSV format
        csv_path = self.output_root / "text" / "annotations.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["sample_id", "split", "emotion", "text", "raw_text"]
            )
            writer.writeheader()
            for idx, sample in enumerate(self.samples):
                sample_id = self._get_sample_id(sample, idx)
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "split": sample.get("split", "train"),
                        "emotion": sample.get("unified_emotion", "unknown"),
                        "text": sample.get("text", ""),
                        "raw_text": sample.get("raw_text", ""),
                    }
                )
        logger.info(f"Wrote text annotations (CSV): {csv_path}")

    def export_splits(self) -> None:
        """Export train/val/test split files."""
        logger.info("Exporting split files...")

        if self.dry_run:
            logger.info("[DRY RUN] Would export split files")
            return

        splits = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            sample_id = self._get_sample_id(sample, idx)
            split = sample.get("split", "train")
            splits[split].append(sample_id)

        for split_name, sample_ids in splits.items():
            split_path = self.output_root / "splits" / f"{split_name}_split.txt"
            with split_path.open("w") as f:
                for sample_id in sample_ids:
                    f.write(f"{sample_id}\n")
            logger.info(f"Wrote {split_name} split ({len(sample_ids)} samples): {split_path}")

    def export_statistics(self) -> None:
        """Export dataset statistics."""
        logger.info("Exporting statistics...")

        if self.dry_run:
            logger.info("[DRY RUN] Would export statistics")
            return

        stats = {
            "total_samples": self.statistics["total_samples"],
            "splits": dict(self.statistics["split_counts"]),
            "emotions": dict(self.statistics["emotion_counts"]),
            "modality_coverage": dict(self.statistics["modality_coverage"]),
        }

        stats_path = self.output_root / "metadata" / "statistics.json"
        with stats_path.open("w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Wrote statistics: {stats_path}")
        logger.info(f"Statistics:\n{json.dumps(stats, indent=2)}")

    def export_release_notes(self) -> None:
        """Create a small README in the exported folder."""
        if self.dry_run:
            logger.info("[DRY RUN] Would export README")
            return

        readme_path = self.output_root / "README.md"
        anonymity = (
            "This export uses anonymized sample and identity IDs and omits original source paths."
            if self.anonymize
            else "This export preserves original sample/source metadata."
        )
        lines = [
            "# Controlla Dataset Export",
            "",
            anonymity,
            "",
            "## Structure",
            "",
            "- `images/{split}`: target images as JPEG files.",
            "- `reference_images/{split}`: identity reference images as JPEG files.",
            "- `audio/{split}`: aligned emotional speech WAV files.",
            "- `text/annotations.*`: text prompts and raw text.",
            "- `metadata/dataset_manifest.*`: sample-level metadata.",
            "- `metadata/statistics.json`: split, emotion, and modality counts.",
            "- `splits/*_split.txt`: exported sample IDs per split.",
        ]
        if self.copy_audio_features:
            lines.append("- `features/audio_features/{split}`: precomputed audio features as NPY files.")

        readme_path.write_text("\n".join(lines) + "\n")
        logger.info(f"Wrote README: {readme_path}")

    def run(self) -> None:
        """Execute full export pipeline."""
        logger.info("Starting dataset export...")
        self.load_jsonl()
        self.compute_statistics()
        self.create_directories()
        self.export_samples()
        self.export_metadata()
        self.export_text_annotations()
        self.export_splits()
        self.export_statistics()
        self.export_release_notes()
        logger.info("Dataset export complete!")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    exporter = DatasetExporter(
        input_jsonl=args.input_jsonl,
        output_root=args.output_root,
        use_symlinks=args.use_symlinks,
        use_hardlinks=args.use_hardlinks,
        copy_audio_features=args.copy_audio_features,
        organize_by_emotion=args.organize_by_emotion,
        anonymize=args.anonymize,
        dry_run=args.dry_run,
    )

    exporter.run()


if __name__ == "__main__":
    main()
