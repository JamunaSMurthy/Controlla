"""Logging utilities for console-friendly research runs."""

from __future__ import annotations

import logging
from pathlib import Path


def create_logger(
    name: str,
    output_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger with optional file logging."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )

    # Avoid duplicate handlers in notebooks/tests.
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        log_path = output_path / "controlla.log"

        existing_file_handlers = [
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
        ]

        if not existing_file_handlers:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False

    return logger