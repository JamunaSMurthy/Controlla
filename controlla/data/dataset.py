"""Dataset definitions for Controlla.

The dataset returns a dictionary containing:

- image: target tensor with shape [3, H, W]
- reference_image: identity/reference tensor with shape [3, H, W]
- prompt: raw text string
- emotion_label: integer label tensor with shape []
- audio_features: tensor with shape [A]
- has_reference and has_audio: float masks with shape []
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import build_image_transform
from utils.manifests import load_manifest_rows, resolve_data_manifest_path


class ControllaDataset(Dataset):
    """Manifest-backed multimodal dataset for Controlla experiments."""

    def __init__(
        self,
        config: dict[str, Any],
        dummy_data: bool = False,
        train: bool = True,
    ) -> None:
        self.config = config
        data_config = config["data"]
        model_config = config["model"]

        self.image_size = int(data_config["image_size"])
        self.image_encoder_backend = str(model_config.get("image_encoder_backend", "cnn"))

        # For CLIP image encoder, use CLIP normalization. For mock/CNN/diffusion,
        # use [-1, 1] normalization.
        transform_mode = "clip" if self.image_encoder_backend == "clip" else "diffusion"
        self.transform = build_image_transform(self.image_size, mode=transform_mode)

        self.audio_dim = int(model_config["audio_dim"])
        self.num_emotions = int(model_config["num_emotions"])

        project_root = Path(config.get("project_root", Path.cwd()))
        self.image_root = self._resolve_base_path(data_config.get("image_root"), project_root)

        self.prompt_column = data_config["prompt_column"]
        self.image_column = data_config["image_column"]
        self.emotion_column = data_config["emotion_column"]
        self.audio_column = data_config["audio_column"]
        self.reference_column = data_config.get("reference_column", self.image_column)

        emotion_taxonomy = [
            str(label).strip().lower()
            for label in data_config.get("emotion_taxonomy", [])
        ]
        self.emotion_to_index = {
            label: index
            for index, label in enumerate(emotion_taxonomy)
        }

        if dummy_data:
            self.rows = self._build_dummy_rows(int(data_config["dummy_num_samples"]))
        else:
            manifest_path = resolve_data_manifest_path(config, train=train)
            self.rows = load_manifest_rows(manifest_path)

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True

        if isinstance(value, float) and math.isnan(value):
            return True

        text = str(value).strip()

        return text == "" or text.lower() in {"nan", "none", "null"}

    def _resolve_base_path(
        self,
        raw_path: str | None,
        project_root: Path,
    ) -> Path | None:
        if self._is_missing(raw_path):
            return None

        candidate = Path(str(raw_path))

        if candidate.exists():
            return candidate

        candidate = (project_root / str(raw_path)).resolve()

        if candidate.exists():
            return candidate

        return Path(str(raw_path))

    def _build_dummy_rows(self, num_samples: int) -> list[dict[str, str]]:
        return [
            {
                self.prompt_column: f"portrait sample {index}",
                self.image_column: "",
                self.reference_column: "",
                self.emotion_column: str(index % self.num_emotions),
                self.audio_column: "",
            }
            for index in range(num_samples)
        ]

    def _encode_emotion_label(self, raw_value: Any) -> int:
        if self._is_missing(raw_value):
            return 0

        if isinstance(raw_value, (int, np.integer)):
            value = int(raw_value)
            if 0 <= value < self.num_emotions:
                return value
            raise ValueError(f"Emotion label index out of range: {value}")

        token = str(raw_value).strip()

        if token.lstrip("-").isdigit():
            value = int(token)
            if 0 <= value < self.num_emotions:
                return value
            raise ValueError(f"Emotion label index out of range: {value}")

        normalized = token.lower()

        if normalized in self.emotion_to_index:
            return self.emotion_to_index[normalized]

        raise ValueError(f"Unsupported emotion label: {raw_value}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, raw_path: Any) -> Path | None:
        if self._is_missing(raw_path):
            return None

        path = Path(str(raw_path))

        if path.exists():
            return path

        if self.image_root is not None:
            candidate = self.image_root / str(raw_path)
            if candidate.exists():
                return candidate

        return None

    def _load_image_or_dummy(
        self,
        image_path: Path | None,
        seed: int,
    ) -> torch.Tensor:
        if image_path is None:
            generator = torch.Generator().manual_seed(seed)
            return torch.rand(
                (3, self.image_size, self.image_size),
                generator=generator,
            ) * 2.0 - 1.0

        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def _load_audio_or_dummy(
        self,
        audio_path: Path | None,
        seed: int,
    ) -> tuple[torch.Tensor, float]:
        if audio_path is None:
            generator = torch.Generator().manual_seed(seed)
            audio = torch.randn(self.audio_dim, generator=generator)
            return audio, 0.0

        try:
            audio = np.load(audio_path)
        except Exception as error:
            raise RuntimeError(f"Failed to load audio feature file: {audio_path}") from error

        audio_tensor = torch.as_tensor(audio, dtype=torch.float32).flatten()

        if audio_tensor.numel() < self.audio_dim:
            pad = torch.zeros(self.audio_dim - audio_tensor.numel())
            audio_tensor = torch.cat([audio_tensor, pad], dim=0)

        audio_tensor = audio_tensor[: self.audio_dim]

        return audio_tensor, 1.0

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]

        image_path = self._resolve_path(row.get(self.image_column, ""))
        reference_path = self._resolve_path(row.get(self.reference_column, ""))
        audio_path = self._resolve_path(row.get(self.audio_column, ""))

        image = self._load_image_or_dummy(image_path, seed=index)

        # If reference is missing, fall back to the target image.
        # has_reference remains 0 if no explicit reference was provided.
        reference_image = self._load_image_or_dummy(
            reference_path if reference_path is not None else image_path,
            seed=index + 1000,
        )

        audio_features, has_audio = self._load_audio_or_dummy(
            audio_path,
            seed=index + 2000,
        )

        return {
            "prompt": str(row.get(self.prompt_column, "portrait")),
            "image": image,
            "reference_image": reference_image,
            "emotion_label": torch.tensor(
                self._encode_emotion_label(row.get(self.emotion_column, 0)),
                dtype=torch.long,
            ),
            "audio_features": audio_features,
            "has_reference": torch.tensor(
                float(reference_path is not None),
                dtype=torch.float32,
            ),
            "has_audio": torch.tensor(
                has_audio,
                dtype=torch.float32,
            ),
        }


class DummyControllaDataset(ControllaDataset):
    """Always-on dummy dataset for tests and smoke runs."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, dummy_data=True)


def build_dataloader(
    config: dict[str, Any],
    train: bool = True,
    dummy_data: bool = False,
) -> DataLoader:
    """Construct a dataloader for train or evaluation runs."""

    dataset = (
        DummyControllaDataset(config)
        if dummy_data
        else ControllaDataset(config, train=train)
    )

    train_config = config["train"]

    return DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=train,
        num_workers=int(train_config["num_workers"]),
        pin_memory=bool(train_config.get("pin_memory", False)),
        drop_last=bool(train_config.get("drop_last", False)),
    )