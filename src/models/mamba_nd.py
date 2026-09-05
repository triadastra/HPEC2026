"""Mamba-ND grid model for the multidimensional trade benchmark.

Implements the Mamba-ND method (Li et al., ECCV 2024; vendored under
``external/Mamba-ND``): scan the data with a 1-D selective SSM separately
along each grid axis, alternating the axis (and direction) layer by layer.
This is the Mamba analogue of S4ND's separable per-axis convolutions.

Two faithful adaptations vs. the upstream repo (see external/Mamba-ND/
VENDORING_NOTE.md):
  * Mixer = **Mamba-2 (SSD)** from ``external/mamba`` rather than Mamba-1.
    The box has only the Triton SSD path; Mamba-1's ``selective_scan_cuda``
    kernel is unavailable. Mamba-ND's scan scheme is mixer-agnostic, so this
    is a drop-in (and uses the same block as the benchmark-winning Mamba-2).
  * The per-axis scan (rearrange axis -> sequence, optional flip, mix,
    rearrange back) follows ``external/Mamba-ND/.../Block.forward`` but drops
    its mmcv ``build_dropout``/FFN wrappers so no mmcv dependency is needed.

Grid axes over the dense (State, Commodity, Flow) lattice + time:
    grid_2d -> scan Time + State                        (cf. s4nd grid_2d)
    grid_3d -> scan Time + State + Commodity            (cf. s4nd grid_3d)
    grid_4d -> scan Time + State + Commodity + Flow     (cf. s4nd grid_4d)

Combos are scattered onto the dense lattice (padded cells masked/zeroed),
scanned, then gathered back; prediction is the last-timestep readout per
kept combo, like the other combo models. GPU/Triton only.
"""

import os
import sys

import torch
import torch.nn as nn
from typing import Optional

# Use the vendored Mamba-2 source, shadowing any installed mamba_ssm, and
# bypass the optional causal_conv1d kernel (same setup as src/models/mamba.py).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAMBA_SRC = os.path.join(_REPO, "external", "mamba")
if os.path.isdir(_MAMBA_SRC) and _MAMBA_SRC not in sys.path:
    sys.path.insert(0, _MAMBA_SRC)
sys.modules.setdefault("causal_conv1d", None)

from mamba_ssm import Mamba2  # noqa: E402

from .base import BaseModel, ModelFactory  # noqa: E402
from .combo_attention import LeftoverEncoder  # noqa: E402
from .scan_schedule import VARIANT_CAT_AXES, build_scan_schedule  # noqa: E402

# Scan-axis map lives in scan_schedule (Triton-free, so it is unit-testable).
_VARIANT_CAT_AXES = VARIANT_CAT_AXES


class MambaND(nn.Module):
    """Alternating per-axis Mamba-2 scans over the (Time, State, Commodity,
    Flow) lattice. Returns (B, L, G, H) for the kept combos.

    ``scan_schedule`` picks which axis each layer sweeps and in which
    direction; see ``scan_schedule.build_scan_schedule``. The default
    ``"cyclic"`` replaces the original schedule, which phase-locked axis order
    against direction and pinned every axis to one direction for the 2d and 4d
    variants. Pass ``scan_schedule="legacy"`` to reproduce runs recorded before
    that fix.
    """

    def __init__(self, features_per_group, d_model, lattice_dims, combo_coords,
                 cat_axes, n_layers=4, d_state=64, d_conv=4, expand=2,
                 headdim=32, dropout=0.1, scan_schedule="cyclic",
                 bidirectional=True):
        super().__init__()
        self.S, self.C, self.Fl = (int(v) for v in lattice_dims)
        self.N = self.S * self.C * self.Fl
        # Max folded-batch rows per Mamba-2 call (keeps the Triton kernel within
        # grid limits when scanning a short axis folds many rows into batch).
        self._scan_chunk = 8192

        coords = torch.as_tensor(combo_coords, dtype=torch.long)            # (G,3)
        flat = coords[:, 0] * (self.C * self.Fl) + coords[:, 1] * self.Fl + coords[:, 2]
        self.register_buffer("flat_idx", flat)                              # (G,)
        cell_valid = torch.zeros(self.N)
        cell_valid[flat] = 1.0
        self.register_buffer("vmask6",
                             cell_valid.view(1, 1, self.S, self.C, self.Fl, 1))

        self.in_proj = nn.Linear(features_per_group, d_model)
        # Scan axes (tensor positions): Time (1) always + the variant's categorical axes.
        self.scan_positions = [1] + list(cat_axes)
        self.scan_schedule = scan_schedule
        self.layer_pos, self.layer_reverse = build_scan_schedule(
            self.scan_positions, n_layers, schedule=scan_schedule,
            bidirectional=bidirectional,
        )
        self.mixers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv,
                   expand=expand, headdim=headdim, use_mem_eff_path=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def _scan_axis(self, x, pos, mixer, norm, reverse):
        """Pre-norm residual Mamba-2 scan along tensor axis ``pos`` of the dense
        grid x = (B, T, S, C, Fl, H). All other grid axes fold into the batch."""
        grid_axes = [1, 2, 3, 4]
        others = [a for a in grid_axes if a != pos]
        perm = [0] + others + [pos, 5]                # (B, o1, o2, o3, A, H)
        d = x.permute(*perm).contiguous()
        shp = d.shape
        A, H = shp[-2], shp[-1]
        seq = d.reshape(-1, A, H)                      # (M=B*o1*o2*o3, A, H)
        h = norm(seq)
        if reverse:
            h = h.flip(1)
        # Chunk the folded batch through the mixer: scanning a short axis (e.g.
        # Flow=2) folds every other axis into M, which can reach tens of
        # thousands and overrun the Mamba-2 Triton kernel's grid limits
        # (illegal memory access). Process in slices to stay within bounds.
        M = h.shape[0]
        if M <= self._scan_chunk:
            h = mixer(h)
        else:
            h = torch.cat([mixer(h[i:i + self._scan_chunk])
                           for i in range(0, M, self._scan_chunk)], dim=0)
        if reverse:
            h = h.flip(1)
        seq = seq + self.dropout(h)                    # pre-norm residual
        out = seq.reshape(*shp)
        inv = [0] * 6
        for new_pos, old in enumerate(perm):
            inv[old] = new_pos
        return out.permute(*inv).contiguous()

    def forward(self, x):                              # x: (B, L, G, F)
        B, L, G, _ = x.shape
        h = self.in_proj(x)                            # (B, L, G, H)
        H = h.shape[-1]
        dense = x.new_zeros(B, L, self.N, H)
        dense[:, :, self.flat_idx, :] = h
        dense = dense.view(B, L, self.S, self.C, self.Fl, H)
        for pos, mixer, norm, rev in zip(
                self.layer_pos, self.mixers, self.norms, self.layer_reverse):
            dense = self._scan_axis(dense, pos, mixer, norm, rev) * self.vmask6
        dense = dense.view(B, L, self.N, H)
        return dense[:, :, self.flat_idx, :]           # (B, L, G, H)


class MambaNDModel(BaseModel):
    """Mamba-ND over the combo lattice; last-timestep readout per combo."""

    requires_combo_loader = True

    def _build_model(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        dropout: float = 0.1,
        dims: int = 4,
        scan_schedule: str = "cyclic",
        bidirectional: bool = True,
        num_combos: Optional[int] = None,
        features_per_group: Optional[int] = None,
        combo_coords=None,
        lattice_dims=None,
        combo_encoder: str = "embeddings",
        state_embed_dim: int = 7,
        comm_embed_dim: int = 88,
        flow_embed_dim: int = 2,
        **kwargs,
    ) -> None:
        assert (
            num_combos is not None and features_per_group is not None
            and combo_coords is not None and lattice_dims is not None
        ), "MambaNDModel needs combo metadata from the combo dataloader."
        cat_axes = {2: (2,), 3: (2, 3), 4: (2, 3, 4)}[dims]
        self.encoder = nn.Identity()
        self.num_groups = num_combos
        # Encoder x dimensionality wiring (PLAN.md §3), same as every other
        # combo model. A scan only makes an axis addressable if it is SCANNED:
        # the un-promoted categorical axes are folded into the batch here
        # exactly as they are in AxialComboSA, so without this the 2d variant
        # cannot tell commodities apart and 3d cannot tell flows apart, while
        # the gru/lstm/transformer/s4nd combos it is compared against can.
        # No-op at dims=4 (out_dim == 0). (F2)
        self.leftover = LeftoverEncoder(
            dims, lattice_dims, combo_coords, encoder=combo_encoder,
            state_embed_dim=state_embed_dim, comm_embed_dim=comm_embed_dim,
            flow_embed_dim=flow_embed_dim,
        )
        feat_dim = features_per_group + self.leftover.out_dim
        self.features_per_group = feat_dim
        self.grid = MambaND(
            feat_dim, d_model, lattice_dims, combo_coords, cat_axes,
            n_layers=n_layers, d_state=d_state, d_conv=d_conv, expand=expand,
            headdim=headdim, dropout=dropout, scan_schedule=scan_schedule,
            bidirectional=bidirectional,
        )
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor = None,
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc = self.leftover()                           # (G, E) or None
        if enc is not None:
            B, L, G, _ = x_numeric.shape
            x_numeric = torch.cat(
                [x_numeric, enc.to(x_numeric.dtype)[None, None].expand(B, L, G, -1)],
                dim=-1,
            )
        h = self.grid(x_numeric)                        # (B, L, G, H)
        return self.output_proj(h[:, -1, :])            # (B, G, 2)


# Grid tags. These models have NO attention -- S4ND mixes with separable DPLR
# kernels and Mamba-ND with an ordered scan -- so the old
# ``cross_attention_{2,3,4}d`` tag named a mechanism they do not contain. The
# tag only ever selected how many lattice axes are promoted. Legacy spellings
# still resolve.
_AXIAL_DIMS = {f"grid_{d}d": d for d in (2, 3, 4)}
_LEGACY_GRID = {
    **{f"cross_attention_{d}d": f"grid_{d}d" for d in (2, 3, 4)},
    **{f"asa_{d}d": f"grid_{d}d" for d in (2, 3, 4)},
}


def make_mamba_nd(variant: str, **kwargs) -> BaseModel:
    variant = _LEGACY_GRID.get(variant, variant)
    if variant not in _AXIAL_DIMS:
        raise ValueError(
            f"mamba_nd supports only grid_{{2d,3d,4d}} tags (got {variant!r})."
        )
    return MambaNDModel(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)


ModelFactory.register("mamba_nd", make_mamba_nd)
