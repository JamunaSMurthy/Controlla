"""Logging utilities for console-friendly research runs."""

from __future__ import annotations

import logging
from pathlib import Path


def create_logger(name: str, output_dir: str | Path | None = None) -> logging.Logger:
    """Create a logger with optional file logging in the experiment directory."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_path / "controlla.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger