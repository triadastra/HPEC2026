"""
Training callbacks for early stopping and checkpointing.
"""

import time
from pathlib import Path
from typing import Optional
import shutil

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Stops training if validation loss doesn't improve for `patience` epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" (for metric to monitor)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric: float) -> bool:
        """
        Check if should stop training.

        Args:
            metric: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == "min":
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class CheckpointManager:
    """
    Manage model checkpoints during training.

    Saves best model(s) and optionally last epoch.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: "min" or "max" (for metric)
            save_best_only: Only save best model
            save_last: Save last epoch checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last

        self.best_score = None

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        score: float,
    ) -> None:
        """
        Save checkpoint if metric improved.

        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            epoch: Current epoch
            score: Current metric value
        """
        is_best = False

        if self.best_score is None:
            is_best = True
        elif self.mode == "min" and score < self.best_score:
            is_best = True
        elif self.mode == "max" and score > self.best_score:
            is_best = True

        if is_best:
            self.best_score = score
            checkpoint_path = self.save_dir / "best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model to {checkpoint_path}")

        if not self.save_best_only and self.save_last:
            checkpoint_path = self.save_dir / "last.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                    "score": score,
                },
                checkpoint_path,
            )
