"""
Loss functions for time series forecasting.

Only the two losses the Trainer actually constructs live here: ``MSELoss``
(flat path) and ``MaskedMSELoss`` (combo path). MAE/Huber/weighted variants
were removed as dead code — nothing selected them, and reintroducing one
means wiring it into the Trainer's criterion choice, not just adding a class.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Mean Squared Error loss."""

    def __init__(self, reduction: str = "mean"):
        """
        Initialize MSE loss.

        Args:
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss.

        Args:
            predictions: (batch_size, 2) - [value, weight]
            targets: (batch_size, 2) - [value, weight]

        Returns:
            Loss value
        """
        return self.mse(predictions, targets)


class MaskedMSELoss(nn.Module):
    """Group-masked MSE loss for combo-window models.

    Combo models output ``(B, G, 2)``: one prediction per (state,
    commodity, flow) combination at each batch element. Not every combo
    exists at every target time — the combo dataloader marks invalid
    slots in ``group_mask`` ``(B, G)`` (1=valid, 0=padded).

    This loss ignores padded slots so the optimizer doesn't chase zeros
    for missing series. Output scale matches per-series MSE: total
    squared error over valid (B*G) slots divided by the count of valid
    slots, then averaged over the 2 targets.

    Matches the masking convention in
    ``paper included/{LSTM,GRU}/inference_*.py`` walk-forward eval.
    """

    def forward(
        self,
        predictions: torch.Tensor,        # (B, G, 2)
        target_value: torch.Tensor,       # (B, G)
        target_weight: torch.Tensor,      # (B, G)
        group_mask: torch.Tensor,         # (B, G), 1=valid, 0=padded
    ) -> torch.Tensor:
        targets = torch.stack([target_value, target_weight], dim=-1)  # (B, G, 2)
        sq_err = (predictions - targets) ** 2                          # (B, G, 2)
        mask = group_mask.unsqueeze(-1).to(sq_err.dtype)               # (B, G, 1)
        masked = sq_err * mask
        denom = mask.sum().clamp_min(1.0) * sq_err.shape[-1]            # valid_slots * 2_targets
        return masked.sum() / denom
