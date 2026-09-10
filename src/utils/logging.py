"""
Logging utilities.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = False,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file (optional)
        log_level: Logging level (default: INFO)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file

    Returns:
        Configured logger instance
    """
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Logger specifically for training with metrics tracking.
    """

    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (for log file naming)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment-specific log file
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_file = self.log_dir / f"{experiment_name}.log"

        # Setup logging
        self.logger = setup_logging(
            log_file=str(log_file),
            log_to_console=True,
            log_to_file=True,
        )

        self.metrics_history = []

    def log_epoch(self, epoch: int, metrics: dict, phase: str = "train") -> None:
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
            phase: Phase (train/val/test)
        """
        metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} [{phase}] - {metric_str}")

        # Store in history
        self.metrics_history.append({
            "epoch": epoch,
            "phase": phase,
            **metrics,
        })

    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def save_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Save metrics history to JSON file.

        Args:
            save_path: Path to save metrics (default: log_dir/metrics.json)
        """
        import json

        if save_path is None:
            save_path = self.log_dir / "metrics.json"

        with open(save_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
