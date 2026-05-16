"""Utilities for dataset anonymization and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class DatasetAnonymizer:
    """Anonymize dataset metadata for public release."""

    def __init__(self, metadata_path: Path) -> None:
        """Initialize anonymizer.
        
        Args:
            metadata_path: Path to dataset_manifest.csv or .json
        """
        self.metadata_path = Path(metadata_path)

    def anonymize_metadata(self, output_path: Path | None = None) -> dict[str, Any]:
        """Remove identifying information from metadata.
        
        Removes:
        - original_sample_id
        - dataset_sources (image_source, audio_source, text_source)
        - alignment_scores (optional, can keep for validation)
        - identity_id (can replace with anonymized hash)
        
        Keeps:
        - split (train/val/test)
        - emotion labels
        - text and raw_text
        - VAD scores
        - sample metadata for reproducibility
        """
        import pandas as pd

        df = pd.read_csv(self.metadata_path)

        # Create mapping for anonymization
        id_mapping = {}

        # Anonymize columns
        if "original_sample_id" in df.columns:
            df = df.drop("original_sample_id", axis=1)

        if "image_source" in df.columns:
            df = df.drop("image_source", axis=1)

        if "audio_source" in df.columns:
            df = df.drop("audio_source", axis=1)

        if "text_source" in df.columns:
            df = df.drop("text_source", axis=1)

        # Optionally anonymize identity_id
        if "identity_id" in df.columns:
            unique_ids = df["identity_id"].unique()
            for old_id in unique_ids:
                if pd.isna(old_id):
                    continue
                # Hash-based anonymization
                hash_val = hashlib.md5(str(old_id).encode()).hexdigest()[:8]
                anon_id = f"id_{hash_val}"
                id_mapping[old_id] = anon_id

            df["identity_id"] = df["identity_id"].apply(
                lambda x: id_mapping.get(x, x) if pd.notna(x) else x
            )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".json":
                df.to_json(output_path, orient="records", indent=2)
            else:
                df.to_csv(output_path, index=False)

            logger.info(f"Anonymized metadata saved to: {output_path}")

        return {
            "anonymized_rows": len(df),
            "columns_removed": [
                "original_sample_id",
                "image_source",
                "audio_source",
                "text_source",
            ],
            "id_mapping_count": len(id_mapping),
        }


class DatasetValidator:
    """Validate dataset folder structure and content."""

    def __init__(self, root_path: Path) -> None:
        """Initialize validator.
        
        Args:
            root_path: Root of dataset folder structure
        """
        self.root = Path(root_path)
        self.issues = []
        self.warnings = []
        self.info = []

    def validate_structure(self) -> dict[str, Any]:
        """Check folder structure is correct."""
        required_dirs = [
            "metadata",
            "images/train",
            "images/val",
            "images/test",
            "reference_images/train",
            "reference_images/val",
            "reference_images/test",
            "audio/train",
            "audio/val",
            "audio/test",
            "text",
            "splits",
        ]

        for rel_path in required_dirs:
            full_path = self.root / rel_path
            if not full_path.exists():
                self.issues.append(f"Missing directory: {rel_path}")
            else:
                self.info.append(f"✓ Found directory: {rel_path}")

        return {"issues": len(self.issues), "warnings": len(self.warnings)}

    def validate_metadata(self) -> dict[str, Any]:
        """Validate metadata files exist and are readable."""
        import pandas as pd

        required_meta = [
            "metadata/dataset_manifest.csv",
            "metadata/statistics.json",
        ]

        metadata_stats = {}

        for meta_file in required_meta:
            path = self.root / meta_file
            if not path.exists():
                self.issues.append(f"Missing metadata file: {meta_file}")
                continue

            try:
                if meta_file.endswith(".csv"):
                    df = pd.read_csv(path)
                    metadata_stats[meta_file] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                    }
                    self.info.append(f"✓ {meta_file}: {len(df)} rows, {len(df.columns)} columns")
                elif meta_file.endswith(".json"):
                    with open(path) as f:
                        data = json.load(f)
                    metadata_stats[meta_file] = len(data) if isinstance(data, list) else 1
                    self.info.append(f"✓ {meta_file}: valid JSON")
            except Exception as e:
                self.issues.append(f"Error reading {meta_file}: {e}")

        return metadata_stats

    def validate_splits(self) -> dict[str, Any]:
        """Validate split files."""
        split_stats = {}

        for split in ["train", "val", "test"]:
            split_file = self.root / "splits" / f"{split}_split.txt"
            if not split_file.exists():
                self.issues.append(f"Missing split file: {split}_split.txt")
                continue

            try:
                with split_file.open() as f:
                    sample_ids = [line.strip() for line in f if line.strip()]

                split_stats[split] = len(sample_ids)
                self.info.append(f"✓ {split}_split.txt: {len(sample_ids)} samples")
            except Exception as e:
                self.issues.append(f"Error reading {split}_split.txt: {e}")

        return split_stats

    def validate_modality_coverage(self) -> dict[str, Any]:
        """Check that advertised modalities exist."""
        import pandas as pd

        csv_path = self.root / "metadata" / "dataset_manifest.csv"
        if not csv_path.exists():
            self.warnings.append("Cannot validate modality coverage without manifest CSV")
            return {}

        df = pd.read_csv(csv_path)

        coverage = {
            "has_image": 0,
            "has_audio": 0,
            "has_text": 0,
            "has_reference_image": 0,
        }

        for col in coverage.keys():
            if col in df.columns:
                # Count True values (handle both string and bool)
                count = ((df[col] == True) | (df[col] == "True")).sum()
                coverage[col] = int(count)
            else:
                self.warnings.append(f"Missing column in manifest: {col}")

        self.info.append(f"✓ Modality coverage: {coverage}")
        return coverage

    def check_file_sampling(self) -> dict[str, Any]:
        """Spot-check that some files actually exist."""
        import pandas as pd

        csv_path = self.root / "metadata" / "dataset_manifest.csv"
        if not csv_path.exists():
            return {}

        df = pd.read_csv(csv_path)

        # Check first 5 training samples
        train_samples = df[df["split"] == "train"].head(5)

        sampling_results = {
            "images_checked": 0,
            "images_found": 0,
            "audio_checked": 0,
            "audio_found": 0,
        }

        for idx, row in train_samples.iterrows():
            sample_id = row["sample_id"]

            # Check image
            if row.get("has_image") == "True" or row.get("has_image"):
                img_path = self.root / "images" / "train" / f"{sample_id}.jpg"
                if not img_path.exists():
                    # Try emotion subdirectory
                    img_path = (
                        self.root
                        / "images"
                        / "train"
                        / row.get("emotion", "unknown")
                        / f"{sample_id}.jpg"
                    )

                sampling_results["images_checked"] += 1
                if img_path.exists():
                    sampling_results["images_found"] += 1

            # Check audio
            if row.get("has_audio") == "True" or row.get("has_audio"):
                audio_path = self.root / "audio" / "train" / f"{sample_id}.wav"
                if not audio_path.exists():
                    audio_path = (
                        self.root
                        / "audio"
                        / "train"
                        / row.get("emotion", "unknown")
                        / f"{sample_id}.wav"
                    )

                sampling_results["audio_checked"] += 1
                if audio_path.exists():
                    sampling_results["audio_found"] += 1

        self.info.append(
            f"✓ Spot check: {sampling_results['images_found']}/{sampling_results['images_checked']} "
            f"images, {sampling_results['audio_found']}/{sampling_results['audio_checked']} audio files found"
        )

        return sampling_results

    def generate_report(self) -> dict[str, Any]:
        """Run all validations and generate report."""
        logger.info("Starting dataset validation...")

        report = {
            "timestamp": str(Path.cwd()),
            "structure_validation": self.validate_structure(),
            "metadata_validation": self.validate_metadata(),
            "split_validation": self.validate_splits(),
            "modality_coverage": self.validate_modality_coverage(),
            "sampling_validation": self.check_file_sampling(),
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
        }

        return report

    def print_report(self, report: dict[str, Any]) -> None:
        """Print validation report to console."""
        print("\n" + "=" * 60)
        print("DATASET VALIDATION REPORT")
        print("=" * 60)

        print("\n✓ INFO:")
        for info in report["info"]:
            print(f"  {info}")

        if report["warnings"]:
            print("\n⚠️  WARNINGS:")
            for warning in report["warnings"]:
                print(f"  {warning}")

        if report["issues"]:
            print("\n❌ ISSUES:")
            for issue in report["issues"]:
                print(f"  {issue}")
        else:
            print("\n✓ No issues found!")

        print("\n" + "=" * 60)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Dataset anonymization and validation tools")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Anonymize subcommand
    anon_parser = subparsers.add_parser("anonymize", help="Anonymize metadata for review")
    anon_parser.add_argument("--input", required=True, help="Input metadata file")
    anon_parser.add_argument("--output", required=True, help="Output metadata file")

    # Validate subcommand
    val_parser = subparsers.add_parser("validate", help="Validate dataset structure")
    val_parser.add_argument("--root", required=True, help="Dataset root directory")

    args = parser.parse_args()

    if args.command == "anonymize":
        anonymizer = DatasetAnonymizer(args.input)
        result = anonymizer.anonymize_metadata(args.output)
        print(json.dumps(result, indent=2))

    elif args.command == "validate":
        validator = DatasetValidator(args.root)
        report = validator.generate_report()
        validator.print_report(report)
        with open("validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: validation_report.json")


if __name__ == "__main__":
    main()
