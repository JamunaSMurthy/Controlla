"""Controlla package root.

This project root also acts as the Python package namespace so modules can be
imported as `models.*`, `data.*`, and related subpackages during editable installs.
"""

__all__ = ["data", "models", "losses", "trainers", "pipelines", "utils"]