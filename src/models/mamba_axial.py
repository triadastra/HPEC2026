"""Axial/CaFA x Mamba hybrids: attention grid mixing + SSM temporal backbone.

Tier-1 (same-harness) combo models for the mamba2/mamba3 rosters: the EXACT
LeftoverEncoder + AxialComboSA / LocalFactorizedAttention instances the GRU/LSTM/
Transformer combos use (src/models/combo_attention.py, cafa.py), the same
residual + post-norm + per-group temporal head composition as
``gru.py::GRUComboAxial`` — with only the temporal operator swapped from
``nn.GRU`` to the stacked pre-norm Mamba-2 / Mamba-3 blocks of the flat
wrappers (mamba.py / mamba3.py). This fills the mamba-ACA / mamba-CaFA cells
of the structure x backbone factorial (Test 6, roster extension 2026-07-14):
only the temporal operator differs from the gru/lstm/transformer combos, and
only the grid-mixing operator differs between the _cross_attention_ and
_cafa_ variants.

The per-group flatten folds G=28,292 sequences into the batch; the Mamba
Triton kernels overrun their grid limits somewhere above ~10k folded rows
(illegal memory access — see mamba_nd.py), so the temporal stack is fed in
chunks of 8,192 rows, exactly like ``MambaND._scan_chunk``.

GPU/Triton only (base torch-2.13 env) — imported lazily from the mamba.py /
mamba3.py factory dispatchers so the flat models keep working where this
cannot load.
"""
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseModel
from .fa_local import LocalFactorizedAttention
from .aca import AxialCrossAttention
from .fa import FactorizedAttention
from .combo_attention import AxialComboSA, LeftoverEncoder, canonical_variant


class MambaComboAxial(BaseModel):
    """Axial/CaFA grid mixing + per-group Mamba temporal stack.

    Pipeline (mirrors GRUComboAxial line by line):
        1. LeftoverEncoder on the not-promoted categoricals, concat to features
        2. AxialComboSA / LocalFactorizedAttention over the (S,C,Fl) lattice -> (B,L,G,H)
        3. Residual from in_proj(x) + LayerNorm
        4. Permute to (B,G,L,H), reshape to (B*G, L, H), per-group pre-norm
           residual Mamba-2/Mamba-3 stack (chunked through the Triton kernel)
        5. Linear(H -> 2) on the last timestep

    Output: (B, G, 2)
    """

    requires_combo_loader = True

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        dims: int = 4,
        factorized: bool = False,
        upstream_fa: bool = False,
        cross_axis: bool = False,
        axis_identity: bool = False,
        fa_heads: int = None,
        fa_dim_head: int = None,
        fa_kernel_multiplier: int = None,
        fa_qk_norm: bool = None,
        fa_kernel_softmax: bool = False,
        authors_cafa_heads: int = 4,
        authors_cafa_dim_head: int = None,
        authors_cafa_kernel_multiplier: int = 2,
        authors_cafa_qk_norm: bool = True,
        mixer: str = "mamba2",
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: Optional[int] = None,
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
        ), (
            "MambaComboAxial needs num_combos, features_per_group, combo_coords "
            "and lattice_dims from the combo dataloader (wired by train.py)."
        )
        self.encoder = nn.Identity()  # combo input is raw (B,L,G,F)
        self.num_groups = num_combos
        # Same encoder-x-dims wiring as every other combo model (PLAN.md §3).
        self.leftover = LeftoverEncoder(
            dims, lattice_dims, combo_coords, encoder=combo_encoder,
            state_embed_dim=state_embed_dim, comm_embed_dim=comm_embed_dim,
            flow_embed_dim=flow_embed_dim,
        )
        feat_dim = features_per_group + self.leftover.out_dim
        self.features_per_group = feat_dim
        if sum(map(bool, (factorized, upstream_fa, cross_axis))) > 1:
            raise ValueError(
                "Pick exactly one grid backend: local FA, authors' FA, or ACA"
            )
        axial_cls = (
            AxialCrossAttention if cross_axis
            else FactorizedAttention if upstream_fa
            else LocalFactorizedAttention if factorized
            else AxialComboSA
        )
        axial_kwargs = {}
        if upstream_fa:
            axial_kwargs.update(
                fa_heads=fa_heads,
                fa_dim_head=fa_dim_head,
                fa_kernel_multiplier=fa_kernel_multiplier,
                fa_qk_norm=fa_qk_norm,
                fa_kernel_softmax=fa_kernel_softmax,
                authors_cafa_heads=authors_cafa_heads,
                authors_cafa_dim_head=authors_cafa_dim_head,
                authors_cafa_kernel_multiplier=authors_cafa_kernel_multiplier,
                authors_cafa_qk_norm=authors_cafa_qk_norm,
            )
        self.axial = axial_cls(
            feat_dim, hidden_size, lattice_dims, combo_coords,
            dims=dims, attn_dropout=attn_dropout, axis_identity=axis_identity,
            **axial_kwargs,
        )
        self.hidden_size = hidden_size
        self.in_proj = nn.Linear(feat_dim, hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        # Temporal SSM stack — identical block config to the flat wrappers
        # (mamba.py: headdim 32 / mamba3.py: headdim 64), pre-norm residual.
        if mixer == "mamba2":
            from .mamba import Mamba2
            hd = 32 if headdim is None else headdim
            def _block():
                return Mamba2(d_model=hidden_size, d_state=d_state, d_conv=d_conv,
                              expand=expand, headdim=hd, use_mem_eff_path=False)
        elif mixer == "mamba3":
            from .mamba3 import Mamba3
            hd = 64 if headdim is None else headdim
            def _block():
                return Mamba3(d_model=hidden_size, d_state=d_state, headdim=hd,
                              expand=expand)
        else:
            raise ValueError(f"MambaComboAxial: unknown mixer {mixer!r}")
        self.blocks = nn.ModuleList([_block() for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, 2)
        self._chunk = 8192  # folded-batch ceiling for the Triton kernels

    def forward(
        self,
        x_numeric: torch.Tensor,             # (B, L, G, F)
        state_ids: torch.Tensor = None,      # ignored for combo path
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc = self.leftover()                              # (G, E) or None
        if enc is not None:
            B, L, G, _ = x_numeric.shape
            x_numeric = torch.cat(
                [x_numeric, enc.to(x_numeric.dtype)[None, None].expand(B, L, G, -1)],
                dim=-1,
            )
        h_attn = self.axial(x_numeric)                     # (B, L, G, H)
        B, L, G, _ = x_numeric.shape
        shortcut = self.in_proj(x_numeric)                 # (B, L, G, H)
        h = self.post_norm(h_attn + shortcut)
        # Groups fold into the batch (see gru.py::_combo_temporal_head for why
        # the permute must precede the reshape), then chunk through the mixer.
        h = h.permute(0, 2, 1, 3).reshape(B * G, L, self.hidden_size)
        last = []
        for i in range(0, h.shape[0], self._chunk):
            hh = h[i:i + self._chunk]
            for blk, norm in zip(self.blocks, self.norms):
                hh = hh + self.dropout(blk(norm(hh)))      # pre-norm residual
            last.append(hh[:, -1, :])
        out = torch.cat(last, dim=0)                       # (B*G, H)
        return self.output_proj(out).reshape(B, G, -1)


_AXIAL_DIMS = {f"asa_{d}d": d for d in (2, 3, 4)}      # axial SELF-attention
_CAFA_DIMS = {f"fa_local_{d}d": d for d in (2, 3, 4)}  # local FA reimplementation
_FA_DIMS = {f"fa_{d}d": d for d in (2, 3, 4)}          # authors' FA operator
_FA_SM_DIMS = {f"fa_sm_{d}d": d for d in (2, 3, 4)}   # ... on their softmax switch
_ACA_DIMS = {f"aca_{d}d": d for d in (2, 3, 4)}        # axial CROSS-attention
_AUTHORS_CAFA_DIMS = {
    "authors_cafa_2d": 2, "authors_cafa_3d": 3, "authors_cafa_4d": 4,
}


def make_mamba_axial(mixer: str, variant: str, **kwargs) -> BaseModel:
    """Factory helper used by mamba.py / mamba3.py for the axial/FA variants."""
    # Accept legacy spellings (cafa_*, cross_attention_*d). See LEGACY_VARIANTS.
    variant = canonical_variant(variant)
    if variant in _AXIAL_DIMS:
        return MambaComboAxial(variant=variant, dims=_AXIAL_DIMS[variant],
                               mixer=mixer, **kwargs)
    if variant in _CAFA_DIMS:
        return MambaComboAxial(variant=variant, dims=_CAFA_DIMS[variant],
                               factorized=True, mixer=mixer, **kwargs)
    if variant in _FA_SM_DIMS:
        # The variant name is the contract: fa_sm_* IS the softmax-kernel
        # arm (Exp 7), so force the flag instead of trusting the composed
        # config to carry it. Silently dropped, it would make this arm a
        # byte-identical duplicate of fa_* and the comparison vacuous.
        kwargs["fa_kernel_softmax"] = True
        return MambaComboAxial(variant=variant, dims=_FA_SM_DIMS[variant],
                               upstream_fa=True, mixer=mixer, **kwargs)
    if variant in _FA_DIMS:
        return MambaComboAxial(variant=variant, dims=_FA_DIMS[variant],
                               upstream_fa=True, mixer=mixer, **kwargs)
    if variant in _ACA_DIMS:
        return MambaComboAxial(variant=variant, dims=_ACA_DIMS[variant],
                               cross_axis=True, mixer=mixer, **kwargs)
    if variant in _AUTHORS_CAFA_DIMS:
        return MambaComboAxial(
            variant=variant, dims=_AUTHORS_CAFA_DIMS[variant],
            upstream_fa=True, mixer=mixer, **kwargs,
        )
    raise ValueError(
        f"Unknown {mixer} combo variant: {variant!r}. Expected "
        "asa_{2d,3d,4d}, aca_{2d,3d,4d}, fa_local_{2d,3d,4d}, or fa_{2d,3d,4d}."
    )
