"""Mamba-2 model for per-series trade forecasting.

Runs Mamba-2 straight from the vendored source in ``external/mamba/``
(cloned state-spaces/mamba), NOT a pip package -- mirroring how the S4
family uses ``external/s4/``. The SSD scan is Triton (compiled at runtime,
no build). Two compiled CUDA kernels are deliberately bypassed so there is
zero build dependency:

  * ``selective_scan_cuda`` -- Mamba-1's kernel, never used by Mamba-2.
  * ``causal_conv1d``       -- optional; Mamba-2 falls back to a plain
                              PyTorch causal conv (``use_mem_eff_path=False``).

Replaces the LSTM-stub "mamba" formerly registered by ssm_models.py.
"""
import os
import sys
import types

import torch.nn as nn

# Use the vendored clone, shadowing any installed mamba_ssm.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MAMBA_SRC = os.path.join(_REPO, "external", "mamba")
if os.path.isdir(_MAMBA_SRC) and _MAMBA_SRC not in sys.path:
    sys.path.insert(0, _MAMBA_SRC)

# Bypass both compiled kernels: stub Mamba-1's (unused), force the conv fallback.
sys.modules.setdefault("selective_scan_cuda", types.ModuleType("selective_scan_cuda"))
sys.modules["causal_conv1d"] = None  # -> import fails -> Mamba2 uses PyTorch conv

from mamba_ssm import Mamba2  # noqa: E402

from .base import BaseModel, ModelFactory


class MambaModel(BaseModel):
    """Stacked pre-norm Mamba-2 blocks over per-series windows; prediction is
    read from the last timestep, mirroring the LSTM/GRU heads."""

    requires_combo_loader = False

    def _build_model(self, d_model: int = 128, n_layers: int = 4,
                     d_state: int = 64, d_conv: int = 4, expand: int = 2,
                     headdim: int = 32, dropout: float = 0.1, **kwargs) -> None:
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv,
                   expand=expand, headdim=headdim, use_mem_eff_path=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x_numeric, state_ids, comm_ids, flow_ids, group_mask=None):
        x = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)
        x = self.input_proj(x)
        for blk, norm in zip(self.blocks, self.norms):
            x = x + self.dropout(blk(norm(x)))   # pre-norm residual
        return self.output_proj(x[:, -1, :])


def make_mamba(variant: str, **kwargs) -> BaseModel:
    if variant in ("onehot", "embeddings"):
        return MambaModel(variant=variant, **kwargs)
    # Axial-CA / CaFA hybrids (Test 6): attention grid mixing + Mamba-2
    # temporal backbone. Lazy import keeps the flat path importable everywhere.
    from .mamba_axial import make_mamba_axial
    return make_mamba_axial("mamba2", variant, **kwargs)


# ``mamba2`` is the honest name (this wrapper IS Mamba-2 / SSD); ``mamba`` stays
# registered as a back-compat alias for the earlier runs/checkpoints.
ModelFactory.register("mamba2", make_mamba)
ModelFactory.register("mamba", make_mamba)
