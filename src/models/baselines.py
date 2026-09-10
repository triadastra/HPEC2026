"""
Non-learned forecasting baselines.

Three parameter-free models, all registered with ``ModelFactory``:

================  =====================  ===================================
Name              Class                  Rule (next-step, per series)
================  =====================  ===================================
seasonal_naive    SeasonalNaiveModel     y_t = y_{t-12}
random_walk       RandomWalkModel        y_t = y_{t-1}
moving_average    MovingAverageModel     y_t = mean(y_{t-k .. t-1}), k=12
================  =====================  ===================================

``moving_average`` is ported from
``paper included/moving_average/inference_ma.py`` (``rolling(k).mean().shift(1)``,
default window 12). The other two are textbook naive baselines.

These have NO trainable parameters and NO categorical encoding — they read
the numeric [Value, Weight] channels straight out of the input window. They
do not go through ``scripts/train.py``. ``scripts/evaluate.py`` scores them on
the same Census test loader and raw-unit metric path as the learned models.
The predecessor-only runner has been removed from the repository.

Feature-column convention (from
``TradeDataPipeline.get_feature_columns``): index 0 = Value, 1 = Weight,
followed by lags and sin/cos. The window passed to ``forward`` is the RAW
numeric tensor (B, L, F) before any encoding.
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseModel, ModelFactory
from .encodings import EncodingStrategy


# Default target-channel positions for TradeDataPipeline, whose feat_cols are
# [Value, Weight, lags..., sin, cos]. They are NOT universal: the census
# 9-channel panel puts agg_value at channel 0 and agg_weight at channel 5
# (sidecar target_channels [0, 5]), with the four mode-value channels in
# between, so scoring a census baseline at index 1 would compare against the
# air-value channel instead of weight.
# Callers pass value_idx / weight_idx (CensusLattice exposes them as
# ``target_ch``); these constants are only the WCTR default. (F12)
VALUE_IDX = 0   # feat_cols[0] == "Value"
WEIGHT_IDX = 1  # feat_cols[1] == "Weight"


class _NullEncoding(EncodingStrategy):
    """Identity 'encoder' — baselines ignore categorical features and keep
    the model parameter-free. ``output_dim`` is just the numeric width."""

    def __init__(self):
        super().__init__()

    def encode(self, x_numeric, state_ids, comm_ids, flow_ids):
        return x_numeric

    def output_dim(self, numeric_input_dim: int) -> int:
        return numeric_input_dim


class _BaselineModel(BaseModel):
    """Shared base: no encoder params, no trainable layers."""

    requires_combo_loader = False
    requires_training = False  # marker; baselines bypass the Trainer entirely

    def _create_encoder(self, variant: str, **kwargs) -> EncodingStrategy:
        return _NullEncoding()

    def _build_model(self, value_idx: int = VALUE_IDX,
                     weight_idx: int = WEIGHT_IDX, **kwargs) -> None:
        # No parameters. Only the target-channel positions, which differ
        # between the WCTR feature layout and the census 9-channel panel. (F12)
        self.value_idx = int(value_idx)
        self.weight_idx = int(weight_idx)

    def _vw(self, x_numeric: torch.Tensor, t_index: int) -> torch.Tensor:
        """Pick [value, weight] at window position ``t_index`` -> (B, 2)."""
        return x_numeric[:, t_index][:, [self.value_idx, self.weight_idx]]


class SeasonalNaiveModel(_BaselineModel):
    """y_t = y_{t-12}. Target is at window_end+1, so y_{t-12} sits at window
    index -12 (requires input_len >= 12; the paper uses 36)."""

    def forward(self, x_numeric, state_ids=None, comm_ids=None,
                flow_ids=None, group_mask: Optional[torch.Tensor] = None):
        if x_numeric.size(1) < 12:
            raise ValueError(
                f"SeasonalNaive needs input_len >= 12, got {x_numeric.size(1)}."
            )
        return self._vw(x_numeric, -12)


class RandomWalkModel(_BaselineModel):
    """y_t = y_{t-1} = last observed step of the input window."""

    def forward(self, x_numeric, state_ids=None, comm_ids=None,
                flow_ids=None, group_mask: Optional[torch.Tensor] = None):
        return self._vw(x_numeric, -1)


class MovingAverageModel(_BaselineModel):
    """y_t = mean(y_{t-k .. t-1}); mean of the input window's last ``window``
    steps. Mirrors paper rolling(window).mean().shift(1), default k=12."""

    def _build_model(self, window: int = 12, **kwargs) -> None:
        super()._build_model(**kwargs)
        self.window = window

    def forward(self, x_numeric, state_ids=None, comm_ids=None,
                flow_ids=None, group_mask: Optional[torch.Tensor] = None):
        k = min(self.window, x_numeric.size(1))
        vw = x_numeric[:, -k:][:, :, [self.value_idx, self.weight_idx]]  # (B, k, 2)
        return vw.mean(dim=1)                                   # (B, 2)


def _make_baseline(cls):
    def factory(variant: str, **kwargs) -> BaseModel:
        # variant is irrelevant (no encoding), accept any for a uniform CLI.
        return cls(variant=variant, **kwargs)
    return factory


ModelFactory.register("seasonal_naive", _make_baseline(SeasonalNaiveModel))
ModelFactory.register("random_walk", _make_baseline(RandomWalkModel))
ModelFactory.register("moving_average", _make_baseline(MovingAverageModel))
