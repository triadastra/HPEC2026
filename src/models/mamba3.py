"""Mamba-3 model for per-series trade forecasting.

Runs Mamba-3 (Dao AI Lab/Goombalab, ICLR 2026 -- exponential-trapezoidal
discretization, complex/RoPE-rotated state, optional MIMO) straight from the
vendored source in ``external/mamba/`` -- the SAME clone the Mamba-2 wrapper
uses. The required SSD scan is the Triton ``mamba3_siso_combined`` kernel
(runtime-compiled, no prebuilt CUDA); the optional tilelang MIMO kernel is left
off, so this is the SISO Mamba-3. Needs triton/GPU (the torch-2.13 toolchain
that ships mamba-ssm >= 2.3.2, which is why the whole benchmark is pinned there).

Mirror of ``mamba.py`` (Mamba-2) with the ``Mamba3`` mixer swapped in; same
d_model / n_layers / d_state as every other backbone so it stays comparable.
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

# Bypass Mamba-1's (unused) compiled kernel + force the PyTorch causal-conv path.
sys.modules.setdefault("selective_scan_cuda", types.ModuleType("selective_scan_cuda"))
sys.modules["causal_conv1d"] = None

from mamba_ssm.modules.mamba3 import Mamba3  # noqa: E402

from .base import BaseModel, ModelFactory


class Mamba3Model(BaseModel):
    """Stacked pre-norm Mamba-3 blocks over per-series windows; prediction is
    read from the last timestep, mirroring the LSTM/GRU/Mamba-2/S4 heads."""

    requires_combo_loader = False

    def _build_model(self, d_model: int = 128, n_layers: int = 4,
                     d_state: int = 64, headdim: int = 64, expand: int = 2,
                     dropout: float = 0.1, **kwargs) -> None:
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.blocks = nn.ModuleList([
            Mamba3(d_model=d_model, d_state=d_state, headdim=headdim, expand=expand)
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


def make_mamba3(variant: str, **kwargs) -> BaseModel:
    if variant in ("onehot", "embeddings"):
        return Mamba3Model(variant=variant, **kwargs)
    # Axial-CA / CaFA hybrids (Test 6): attention grid mixing + Mamba-3
    # temporal backbone. Lazy import keeps the flat path importable everywhere.
    from .mamba_axial import make_mamba_axial
    return make_mamba_axial("mamba3", variant, **kwargs)


ModelFactory.register("mamba3", make_mamba3)
