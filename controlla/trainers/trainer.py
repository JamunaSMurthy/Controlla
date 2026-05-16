"""Training loop for Controlla."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from losses.total_loss import ControllaLoss
from trainers.train_step import compute_train_step
from utils.checkpoint import save_checkpoint
from utils.logger import create_logger


class Trainer:
    """Simple research trainer for Controlla experiments."""

    def __init__(self, model: torch.nn.Module, config: dict[str, Any], device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = create_logger("controlla.train", output_dir=config["output_dir"])
        self.loss_fn = ControllaLoss(config).to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]))

    def fit(self, train_loader: torch.utils.data.DataLoader) -> None:
        """Train the model for the configured number of epochs."""
        self.model.train()
        num_epochs = int(self.config["train"]["num_epochs"])
        log_every = int(self.config["train"]["log_every"])
        save_every = int(self.config["train"]["save_every"])
        max_grad_norm = float(self.config["train"]["max_grad_norm"])

        global_step = 0
        for epoch in range(num_epochs):
            for batch in train_loader:
                loss, metrics, _ = compute_train_step(self.model, self.loss_fn, batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                if global_step % log_every == 0:
                    self.logger.info("step=%s metrics=%s", global_step, metrics)
                global_step += 1

            if save_every > 0 and (epoch + 1) % save_every == 0:
                checkpoint_path = Path(self.config["output_dir"]) / f"checkpoint_epoch_{epoch + 1}.pt"
                save_checkpoint(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "config": self.config,
                        "epoch": epoch + 1,
                    },
                    checkpoint_path,
                )
                self.logger.info("saved checkpoint to %s", checkpoint_path)