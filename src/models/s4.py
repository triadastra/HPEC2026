"""Real S4 model for trade forecasting (per-series window head).

Wraps the GENUINE S4 layer -- ``S4Block`` (FFTConv + DPLR SSM kernel, the
default ``mode='dplr'``) -- loaded straight from the vendored standalone at
``external/s4/models/s4/s4.py`` (state-spaces/s4). This is the SAME S4
implementation benchmarked in the per-combo Exp 2 runs, NOT a simplified
github-side stand-in (contrast the gated diagonal RNN in ``dss.py``). The
standalone falls back to ``cauchy_naive`` when pykeops / the CUDA extension are
absent, so there is zero compiled-kernel build dependency.

It exists so the real S4 can also run through ``scripts/train.py``'s aggregate
path (Exp 1), which the upstream Hydra framework does not cover.
"""
import importlib.util
import os
import sys

import torch.nn as nn

from .base import BaseModel, ModelFactory

# The standalone S4 is a single self-contained file (no relative imports and no
# package __init__.py), so load it directly from its path -- avoids sys.path
# shadowing of the generic ``models`` name and needs no package scaffolding.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_S4_FILE = os.path.join(_REPO, "external", "s4", "models", "s4", "s4.py")
_spec = importlib.util.spec_from_file_location("_real_s4_standalone", _S4_FILE)
_s4mod = importlib.util.module_from_spec(_spec)
sys.modules["_real_s4_standalone"] = _s4mod
_spec.loader.exec_module(_s4mod)
S4Block = _s4mod.S4Block


class S4Model(BaseModel):
    """Stacked pre-norm real-S4 (``S4Block``) layers over the input window; the
    prediction is read from the last timestep, mirroring the LSTM/GRU/Mamba heads."""

    requires_combo_loader = False

    def _build_model(self, d_model: int = 128, n_layers: int = 4,
                     d_state: int = 64, dropout: float = 0.1, **kwargs) -> None:
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.blocks = nn.ModuleList([
            S4Block(d_model, d_state=d_state, transposed=False, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x_numeric, state_ids, comm_ids, flow_ids, group_mask=None):
        x = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)  # (B, L, input_dim)
        x = self.input_proj(x)                                              # (B, L, d_model)
        for blk, norm in zip(self.blocks, self.norms):
            y, _ = blk(norm(x))            # S4Block returns (output, state)
            x = x + self.dropout(y)         # pre-norm residual
        return self.output_proj(x[:, -1, :])                                # (B, 2)


def make_s4(variant: str, **kwargs) -> BaseModel:
    if variant in ("onehot", "embeddings"):
        return S4Model(variant=variant, **kwargs)
    raise ValueError(
        f"Unsupported S4 variant: {variant!r}. S4 supports: onehot, embeddings."
    )


ModelFactory.register("s4", make_s4)
