"""Random seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed