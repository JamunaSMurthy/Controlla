"""Stable hashing helpers for deterministic IDs and splits."""

from __future__ import annotations

import hashlib


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()