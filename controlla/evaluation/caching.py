"""Disk-backed embedding caches for experiment metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingCache:
    """Simple filesystem cache for expensive embedding computations."""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_root / namespace / f"{digest}.npy"

    def load(self, namespace: str, key: str) -> np.ndarray | None:
        path = self._cache_path(namespace, key)
        if not path.exists():
            return None
        return np.load(path)

    def save(self, namespace: str, key: str, array: np.ndarray) -> None:
        path = self._cache_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)

    def write_metadata(self, namespace: str, payload: dict[str, Any]) -> None:
        path = self.cache_root / namespace / "metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")