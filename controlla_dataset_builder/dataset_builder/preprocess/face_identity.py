"""Identity helpers using hard labels or embedding-based fallbacks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_builder.utils.hashing import stable_hash


class ArcFaceIdentityEncoder:
    """ArcFace identity encoder with deterministic fallback support.

    Set `CONTROLLA_USE_REAL_ARCFACE=1` and provide a local InsightFace model
    root through `CONTROLLA_ARCFACE_MODEL_ROOT` to use real ArcFace embeddings.
    Without a configured local model, the encoder falls back to deterministic
    image features so offline tests and dataset plumbing remain reproducible.
    """

    def __init__(
        self,
        image_size: int = 32,
        histogram_bins: int = 16,
        *,
        use_real_arcface: bool | None = None,
        model_name: str | None = None,
        model_root: str | Path | None = None,
    ) -> None:
        self.image_size = image_size
        self.histogram_bins = histogram_bins
        self.backend = "deterministic_fallback"
        self._app = None

        should_use_real = (
            bool(int(os.getenv("CONTROLLA_USE_REAL_ARCFACE", "0")))
            if use_real_arcface is None
            else use_real_arcface
        )
        raw_model_root = model_root or os.getenv("CONTROLLA_ARCFACE_MODEL_ROOT", "")
        if should_use_real and raw_model_root:
            resolved_model_root = Path(raw_model_root).expanduser()
            self._app = self._load_insightface_app(model_name or os.getenv("CONTROLLA_ARCFACE_MODEL", "buffalo_l"), resolved_model_root)
            if self._app is not None:
                self.backend = "arcface"

    def _load_insightface_app(self, model_name: str, model_root: Path):
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=model_name, root=str(model_root), providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception:
            return None

    def encode(self, image_path: str) -> list[float]:
        if self._app is not None:
            embedding = self._encode_with_arcface(image_path)
            if embedding is not None:
                return embedding.astype(np.float32).tolist()
        return self._encode_fallback(image_path)

    def _encode_with_arcface(self, image_path: str) -> np.ndarray | None:
        try:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            faces = self._app.get(image)
            if not faces:
                return None
            face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            if embedding.size == 0:
                return None
            return embedding
        except Exception:
            return None

    def _encode_fallback(self, image_path: str) -> list[float]:
        path = Path(image_path)
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(rgb, dtype=np.float32) / 255.0

        channel_means = array.mean(axis=(0, 1))
        channel_stds = array.std(axis=(0, 1))
        gray = array.mean(axis=2)
        horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
        vertical_edges = np.abs(np.diff(gray, axis=0)).mean()

        histograms = []
        for channel in range(3):
            hist, _ = np.histogram(array[:, :, channel], bins=self.histogram_bins, range=(0.0, 1.0), density=True)
            histograms.append(hist.astype(np.float32))

        feature_vector = np.concatenate(
            [
                channel_means.astype(np.float32),
                channel_stds.astype(np.float32),
                np.array([horizontal_edges, vertical_edges], dtype=np.float32),
                *histograms,
            ]
        )
        return feature_vector.astype(np.float32).tolist()


def infer_identity_hash(embedding: list[float] | np.ndarray) -> str:
    array = np.asarray(embedding, dtype=np.float32)
    rounded = np.round(array, 4)
    return stable_hash("|".join(str(value) for value in rounded[:32]))[:16]


def extract_identity_features(
    image_path: str,
    output_path: str | Path | None = None,
    encoder: ArcFaceIdentityEncoder | None = None,
) -> tuple[str, str]:
    active_encoder = encoder or ArcFaceIdentityEncoder()
    embedding = np.asarray(active_encoder.encode(image_path), dtype=np.float32)
    destination = Path(output_path) if output_path is not None else Path(image_path).with_suffix(".identity.npy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, embedding)
    return str(destination.resolve()), infer_identity_hash(embedding)
