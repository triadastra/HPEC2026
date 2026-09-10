"""Genuine S4ND over the masked (Time, State, Commodity, Flow) lattice.

S4ND (Nguyen et al., NeurIPS 2022) extends S4 to N dimensions by exploiting
LTI-ness: each axis gets its own 1-D SSM convolution kernel and the N-D kernel
is their OUTER PRODUCT — one native multidimensional linear operator (the
"factorized shared-kernel" cell of the structure taxonomy; CaFA is its
attention analogue). Because the joint kernel is separable (channels=1,
rank-1), the N-D FFT convolution factorizes EXACTLY into sequential per-axis
1-D FFT convolutions — mathematically identical to materializing the
outer-product kernel, at a fraction of the memory.

Kernels are the REAL DPLR SSM kernels (``SSMKernelDPLR``) from the vendored
standalone ``external/s4/models/s4/s4.py`` — the same file the flat ``s4``
model wraps — NOT the package-bound ``src/models/sequence/modules/s4nd.py``
(whose ``src.*`` imports collide with this repo's own ``src`` package).
Bidirectional two-sided kernels per axis follow ``FFTConv.forward``'s padding
recipe verbatim; bidirectionality over the Time axis is safe because the whole
36-month window strictly precedes the target month (the transformer likewise
attends bidirectionally within the window).

Caveat this model exists to TEST (PLAN.md taxonomy): an LTI kernel along an
axis presumes translation structure. State / HS6-commodity / flow are
unordered categoricals, and the ~20% invalid cells are zeroed between layers
but enter each layer's convolutions carrying the pre-norm LayerNorm bias
(kernels smear across masked cells; outputs are re-masked after every layer). If S4ND ties or loses here, the translation-equivariance
ill-posedness argument stands empirically.

Grid variants reuse the ``cross_attention_{2,3,4}d`` slots as grid tags
(precedent: mamba_nd) — Time is always convolved, plus State / +Commodity /
+Flow. No ``cafa_*`` variants: S4ND is natively factorized.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, ModelFactory
from .combo_attention import LeftoverEncoder
# Reuse the already-loaded standalone (module attribute of s4.py). If that
# import fails (e.g. the base/2.13 env's torchvision circularity), this module
# fails too and src/models/__init__ skips registration defensively.
from .s4 import _s4mod

_DPLR_KERNEL = _s4mod.kernel_registry["dplr"]           # SSMKernelDPLR


class S4NDLayer(nn.Module):
    """One S4ND layer: separable N-D SSM convolution + D-skip + GELU + Linear.

    Applies the per-axis bidirectional DPLR kernels sequentially (exact
    separable N-D conv), then the S4Block-style position-wise tail.
    """

    def __init__(self, d_model, axis_lens, d_state=64, dropout=0.1):
        super().__init__()
        self.axis_lens = list(axis_lens)
        # One bidirectional kernel per axis: channels=2 -> (2, H, L), split
        # into forward/backward halves exactly like FFTConv(bidirectional=True).
        self.kernels = nn.ModuleList([
            _DPLR_KERNEL(d_model=d_model, l_max=int(L), channels=2, d_state=d_state)
            for L in self.axis_lens
        ])
        self.D = nn.Parameter(torch.randn(d_model))     # skip term (channels=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)

    def _axis_conv(self, x, pos, kernel, L):
        """Two-sided FFT convolution along tensor axis ``pos``.

        x: (B, T, S, C, Fl, H); returns same shape.
        """
        xm = x.movedim(pos, -1).contiguous()            # (B, ..., H, L)
        lead = xm.shape[:-1]
        xm2 = xm.reshape(-1, xm.shape[-2], L)           # (M, H, L) — H stays adjacent to L
        k, _ = kernel(L=L)                              # (2, H, L)
        k = F.pad(k[0], (0, L)) + F.pad(k[1].flip(-1), (L, 0))   # (H, 2L) two-sided
        k_f = torch.fft.rfft(k, n=2 * L)                # (H, Lf)
        x_f = torch.fft.rfft(xm2, n=2 * L)              # (M, H, Lf)
        y = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]  # (M, H, L)
        return y.reshape(*lead, L).movedim(-1, pos)

    def forward(self, x, axis_positions):
        """x: (B, T, S, C, Fl, H); conv along each axis in ``axis_positions``."""
        y = x
        for pos, kernel, L in zip(axis_positions, self.kernels, self.axis_lens):
            y = self._axis_conv(y, pos, kernel, L)
        y = y + x * self.D                              # skip (per-feature)
        y = self.drop(self.act(y))
        return self.out_linear(y)


class S4NDModel(BaseModel):
    """S4ND grid model over the combo lattice; last-timestep readout per combo.

    Composition mirrors mamba_nd (the grid-native slot): leftover encoder +
    in_proj -> scatter to the dense lattice -> stacked pre-norm residual S4ND
    layers (each layer = one full separable N-D conv over Time + the promoted
    axes) with invalid cells re-masked after every layer -> gather -> readout.
    """

    requires_combo_loader = True

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        d_state: int = 64,
        dropout: float = 0.1,
        dims: int = 4,
        input_len: int = 36,
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
        ), "S4NDModel needs combo metadata from the combo dataloader."
        self.encoder = nn.Identity()
        self.num_groups = num_combos
        self.S, self.C, self.Fl = (int(v) for v in lattice_dims)
        self.N = self.S * self.C * self.Fl
        self.hidden_size = hidden_size
        self.input_len = input_len

        coords = torch.as_tensor(combo_coords, dtype=torch.long)          # (G,3)
        flat = coords[:, 0] * (self.C * self.Fl) + coords[:, 1] * self.Fl + coords[:, 2]
        self.register_buffer("flat_idx", flat)
        cell_valid = torch.zeros(self.N)
        cell_valid[flat] = 1.0
        self.register_buffer("vmask6",
                             cell_valid.view(1, 1, self.S, self.C, self.Fl, 1))

        # Leftover categoricals (same wiring as every other combo model).
        self.leftover = LeftoverEncoder(
            dims, lattice_dims, combo_coords, encoder=combo_encoder,
            state_embed_dim=state_embed_dim, comm_embed_dim=comm_embed_dim,
            flow_embed_dim=flow_embed_dim,
        )
        feat_dim = features_per_group + self.leftover.out_dim
        self.features_per_group = feat_dim
        self.in_proj = nn.Linear(feat_dim, hidden_size)

        # Tensor axes of the dense grid (B, T, S, C, Fl, H): Time=1 always +
        # the promoted categorical axes (2d: State; 3d: +Commodity; 4d: +Flow).
        cat_positions = {2: (2,), 3: (2, 3), 4: (2, 3, 4)}[dims]
        self.axis_positions = [1] + list(cat_positions)
        grid_lens = {1: input_len, 2: self.S, 3: self.C, 4: self.Fl}
        axis_lens = [grid_lens[p] for p in self.axis_positions]

        self.layers = nn.ModuleList([
            S4NDLayer(hidden_size, axis_lens, d_state=d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, 2)

    def forward(
        self,
        x_numeric: torch.Tensor,             # (B, L, G, F)
        state_ids: torch.Tensor = None,
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc = self.leftover()
        if enc is not None:
            B, L, G, _ = x_numeric.shape
            x_numeric = torch.cat(
                [x_numeric, enc.to(x_numeric.dtype)[None, None].expand(B, L, G, -1)],
                dim=-1,
            )
        B, L, G, _ = x_numeric.shape
        h = self.in_proj(x_numeric)                     # (B, L, G, H)
        dense = h.new_zeros(B, L, self.N, h.shape[-1])
        dense[:, :, self.flat_idx, :] = h
        dense = dense.view(B, L, self.S, self.C, self.Fl, -1)
        for layer, norm in zip(self.layers, self.norms):
            dense = (dense + self.dropout(layer(norm(dense), self.axis_positions))) \
                * self.vmask6                            # pre-norm residual + re-mask
        dense = dense.reshape(B, L, self.N, -1)
        h = dense[:, :, self.flat_idx, :]               # (B, L, G, H)
        return self.output_proj(h[:, -1, :])            # (B, G, 2)


# Grid tags. These models have NO attention -- S4ND mixes with separable DPLR
# kernels and Mamba-ND with an ordered scan -- so the old
# ``cross_attention_{2,3,4}d`` tag named a mechanism they do not contain. The
# tag only ever selected how many lattice axes are promoted. Legacy spellings
# still resolve.
_GRID_DIMS = {f"grid_{d}d": d for d in (2, 3, 4)}
_LEGACY_GRID = {
    **{f"cross_attention_{d}d": f"grid_{d}d" for d in (2, 3, 4)},
    **{f"asa_{d}d": f"grid_{d}d" for d in (2, 3, 4)},
}


def make_s4nd(variant: str, **kwargs) -> BaseModel:
    variant = _LEGACY_GRID.get(variant, variant)
    if variant not in _GRID_DIMS:
        raise ValueError(
            f"s4nd supports only grid_{{2d,3d,4d}} tags (got {variant!r})."
        )
    return S4NDModel(variant=variant, dims=_GRID_DIMS[variant], **kwargs)


ModelFactory.register("s4nd", make_s4nd)
