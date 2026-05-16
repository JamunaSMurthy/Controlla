"""Dataset loader for folder-based Controlla dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


class ControllaFolderDataset:
    """Load Controlla dataset from folder structure.
    
    Provides a simple interface to read the folder-organized dataset
    including images, audio, text annotations, and metadata.
    """

    def __init__(self, root_path: str | Path, split: str = "train") -> None:
        """Initialize dataset loader.
        
        Args:
            root_path: Root of exported folder structure
            split: One of 'train', 'val', 'test'
        """
        self.root = Path(root_path)
        self.split = split

        # Load metadata
        self.metadata_df = self._load_metadata()
        self.split_samples = self._load_split_file()

    def _load_metadata(self) -> list[dict[str, Any]]:
        """Load metadata CSV."""
        csv_path = self.root / "metadata" / "dataset_manifest.csv"
        metadata = []
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            metadata = list(reader)
        return metadata

    def _load_split_file(self) -> list[str]:
        """Load sample IDs for this split."""
        split_path = self.root / "splits" / f"{self.split}_split.txt"
        with split_path.open("r") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        """Number of samples in split."""
        return len(self.split_samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get sample by index.
        
        Returns dict with:
            - sample_id
            - image: PIL Image or None
            - reference_image: PIL Image or None
            - audio: numpy array or None
            - text: str
            - emotion: str
            - metadata: dict with all annotation info
        """
        from PIL import Image

        from dataset_builder.preprocess.audio_features import load_audio_waveform

        sample_id = self.split_samples[idx]

        # Find metadata
        meta = next(
            (m for m in self.metadata_df if m["sample_id"] == sample_id),
            None,
        )

        if meta is None:
            raise ValueError(f"Sample {sample_id} not found in metadata")

        result = {
            "sample_id": sample_id,
            "split": self.split,
            "emotion": meta.get("emotion", "unknown"),
            "text": meta.get("text", ""),
            "raw_text": meta.get("raw_text", ""),
            "metadata": meta,
        }

        # Load image
        if meta.get("has_image") == "True" or meta.get("has_image"):
            img_path = self.root / "images" / self.split / f"{sample_id}.jpg"
            if not img_path.exists():
                # Try emotion subdirectory
                emotion = meta.get("emotion", "unknown")
                img_path = (
                    self.root / "images" / self.split / emotion / f"{sample_id}.jpg"
                )

            if img_path.exists():
                result["image"] = Image.open(img_path).convert("RGB")
            else:
                result["image"] = None
        else:
            result["image"] = None

        # Load reference image
        if meta.get("has_reference_image") == "True" or meta.get("has_reference_image"):
            ref_path = self.root / "reference_images" / self.split / f"{sample_id}_ref.jpg"
            if not ref_path.exists():
                emotion = meta.get("emotion", "unknown")
                ref_path = (
                    self.root
                    / "reference_images"
                    / self.split
                    / emotion
                    / f"{sample_id}_ref.jpg"
                )

            if ref_path.exists():
                result["reference_image"] = Image.open(ref_path).convert("RGB")
            else:
                result["reference_image"] = None
        else:
            result["reference_image"] = None

        # Load audio
        if meta.get("has_audio") == "True" or meta.get("has_audio"):
            audio_path = self.root / "audio" / self.split / f"{sample_id}.wav"
            if not audio_path.exists():
                emotion = meta.get("emotion", "unknown")
                audio_path = (
                    self.root / "audio" / self.split / emotion / f"{sample_id}.wav"
                )

            if audio_path.exists():
                sr, audio = load_audio_waveform(audio_path)
                result["audio"] = audio
                result["audio_sample_rate"] = sr
            else:
                result["audio"] = None
        else:
            result["audio"] = None

        return result

    def get_split_stats(self) -> dict[str, Any]:
        """Get statistics for this split."""
        emotion_counts = {}
        modality_counts = {"has_image": 0, "has_audio": 0, "has_text": 0, "has_reference": 0}

        for meta in self.metadata_df:
            if meta["split"] == self.split:
                emotion = meta.get("emotion", "unknown")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                if meta.get("has_image") == "True" or meta.get("has_image"):
                    modality_counts["has_image"] += 1
                if meta.get("has_audio") == "True" or meta.get("has_audio"):
                    modality_counts["has_audio"] += 1
                if meta.get("has_text") == "True" or meta.get("has_text") or meta.get("text"):
                    modality_counts["has_text"] += 1
                if meta.get("has_reference_image") == "True" or meta.get("has_reference_image"):
                    modality_counts["has_reference"] += 1

        return {
            "split": self.split,
            "num_samples": len(self.split_samples),
            "emotions": emotion_counts,
            "modalities": modality_counts,
        }


class ControllaBatchLoader:
    """Batch loader for Controlla folder dataset."""

    def __init__(
        self,
        root_path: str | Path,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> None:
        """Initialize batch loader.
        
        Args:
            root_path: Root of exported folder structure
            split: Dataset split
            batch_size: Batch size
            shuffle: Whether to shuffle
        """
        self.dataset = ControllaFolderDataset(root_path, split)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))

        if shuffle:
            import random

            random.shuffle(self.indices)

    def __len__(self) -> int:
        """Number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches."""
        for batch_idx in range(len(self)):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            batch_indices = self.indices[start:end]

            batch = {
                "sample_id": [],
                "image": [],
                "reference_image": [],
                "audio": [],
                "text": [],
                "emotion": [],
            }

            for idx in batch_indices:
                sample = self.dataset[idx]
                batch["sample_id"].append(sample["sample_id"])
                batch["image"].append(sample.get("image"))
                batch["reference_image"].append(sample.get("reference_image"))
                batch["audio"].append(sample.get("audio"))
                batch["text"].append(sample["text"])
                batch["emotion"].append(sample["emotion"])

            yield batch
