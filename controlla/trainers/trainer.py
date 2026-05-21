"""Training loop for Controlla."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast

from controlla.losses.total_loss import ControllaLoss
from controlla.trainers.eval_step import compute_validation_step
from controlla.trainers.train_step import compute_train_step
from controlla.utils.checkpoint import save_checkpoint
from controlla.utils.logger import create_logger


class Trainer:
    """Research trainer for Controlla experiments.

    Supports:
        - epoch-based training,
        - iteration-based training,
        - gradient accumulation,
        - mixed precision,
        - gradient clipping,
        - optional validation,
        - last/best checkpoint saving.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.output_dir = Path(config.get("output_dir", "outputs/train"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(
            "controlla.train",
            output_dir=str(self.output_dir