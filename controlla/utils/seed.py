"""Reproducibility helpers for Controlla."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool | None = None,
) -> None:
    """Set random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed.
        deterministic: Whether to force deterministic CuDNN behavior.
        benchmark: Optional CuDNN benchmark override. If None, benchmark is set
            to the opposite of deterministic.
    """

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = (not deterministic) if benchmark is None else bool(benchmark)