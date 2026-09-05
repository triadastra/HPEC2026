"""
LSTM models for time series forecasting.

Mirror of ``gru.py`` with ``nn.LSTM`` in place of ``nn.GRU``. Restored into the
HPEC roster (2026-07) alongside GRU as a recurrent baseline — it was dropped in
the initial sequel scaffold, but the variant configs still reference
``make_lstm`` and it is a cheap, informative anchor. Same four paths as GRU:

================  ===========================  ================
Variant flag      Class                        Input shape
================  ===========================  ================
onehot            LSTMModel                    (B, L, F)
embeddings        LSTMModel                    (B, L, F)
cross_attention   LSTMComboCAOnly              (B, L, G, F)
film_attention    LSTMComboCAFiLM              (B, L, G, F)
cross_attention_{2,3,4}d / cafa_{2,3,4}d   LSTMComboAxial   (B, L, G, F)
================  ===========================  ================

``nn.LSTM`` returns ``(output, (h_n, c_n))``; we unpack ``out, _`` exactly as the
GRU path does, so the cell state is carried internally and the interface is
byte-for-byte the same as ``gru.py``. See ``gru.py`` for the per-group flatten
correctness note in ``_combo_temporal_head``.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseModel, ModelFactory
from .fa_local import LocalFactorizedAttention
from .aca import AxialCrossAttention
from .fa import FactorizedAttention
from .combo_attention import (
    AxialComboSA,
    LeftoverEncoder,
    canonical_variant,
)


# ---------------------------------------------------------------------------
# Variant 1: normal 1D LSTM (onehot, embeddings)
# ---------------------------------------------------------------------------

class LSTMModel(BaseModel):
    """Per-series LSTM (mirror of ``GRUModel``)."""

    requires_combo_loader = False

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.input_proj = nn.Linear(self.input_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.output_proj = nn.Linear(out_dim, 2)

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,  # ignored; uniform signature
    ) -> torch.Tensor:
        x = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        return self.output_proj(out[:, -1, :])


# ---------------------------------------------------------------------------
# Shared backbone helpers (mirror gru.py; attribute is `lstm`)
# ---------------------------------------------------------------------------

def _build_combo_backbone(
    module: nn.Module,
    features_per_group: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> None:
    module.hidden_size = hidden_size
    module.in_proj = nn.Linear(features_per_group, hidden_size)
    module.post_norm = nn.LayerNorm(hidden_size)
    module.lstm = nn.LSTM(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0.0,
        batch_first=True,
    )
    module.output_proj = nn.Linear(hidden_size, 2)


def _combo_temporal_head(
    module: nn.Module,
    h_modulated: torch.Tensor,  # (B, L, G, H)
    x_raw: torch.Tensor,        # (B, L, G, F)
) -> torch.Tensor:
    """Residual + per-group LSTM + output projection. Returns (B, G, 2)."""
    B, L, G, _ = x_raw.shape
    shortcut = module.in_proj(x_raw)
    h = module.post_norm(h_modulated + shortcut)
    # Move group axis next to batch BEFORE flattening (see gru.py note).
    h = h.permute(0, 2, 1, 3).reshape(B * G, L, module.hidden_size)
    out, _ = module.lstm(h)
    return module.output_proj(out[:, -1, :]).reshape(B, G, -1)


# ---------------------------------------------------------------------------
# Variant 2: cross-attention only
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Variant 3: cross-attention + FiLM
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Variant 4: axial cross-attention (S4ND-style 2d/3d/4d) + CaFA (factorized)
# ---------------------------------------------------------------------------

class LSTMComboAxial(BaseModel):
    """LSTM over groups with axial cross-attention (or factorized CaFA)."""

    requires_combo_loader = True

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
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
            "LSTMComboAxial needs num_combos, features_per_group, combo_coords "
            "and lattice_dims from the combo dataloader (wired by train.py)."
        )
        self.encoder = nn.Identity()
        self.num_groups = num_combos
        # Encode categoricals not promoted to an axial axis (2d: comm+flow, 3d:
        # flow, 4d: none); append to per-group feats -- encoder x dims (PLAN.md §3).
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
        _build_combo_backbone(self, feat_dim, hidden_size, num_layers, dropout)

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor = None,
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
        h_attn = self.axial(x_numeric)
        return _combo_temporal_head(self, h_attn, x_numeric)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

_AXIAL_DIMS = {f"asa_{d}d": d for d in (2, 3, 4)}      # axial SELF-attention
_CAFA_DIMS = {f"fa_local_{d}d": d for d in (2, 3, 4)}  # local FA reimplementation
_FA_DIMS = {f"fa_{d}d": d for d in (2, 3, 4)}          # authors' FA operator
_FA_SM_DIMS = {f"fa_sm_{d}d": d for d in (2, 3, 4)}   # ... on their softmax switch
_ACA_DIMS = {f"aca_{d}d": d for d in (2, 3, 4)}        # axial CROSS-attention
_AUTHORS_CAFA_DIMS = {
    "authors_cafa_2d": 2, "authors_cafa_3d": 3, "authors_cafa_4d": 4,
}


def make_lstm(variant: str, **kwargs) -> BaseModel:
    """Pick the right LSTM class for the requested variant (mirrors make_gru)."""
    # Accept legacy spellings (cafa_*, cross_attention_*d) so old
    # manifests and checkpoints keep resolving. See LEGACY_VARIANTS.
    variant = canonical_variant(variant)
    if variant in ("onehot", "embeddings"):
        return LSTMModel(variant=variant, **kwargs)
    if variant in _AXIAL_DIMS:
        return LSTMComboAxial(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)
    if variant in _CAFA_DIMS:
        return LSTMComboAxial(variant=variant, dims=_CAFA_DIMS[variant],
                              factorized=True, **kwargs)
    if variant in _FA_SM_DIMS:
        # The variant name is the contract: fa_sm_* IS the softmax-kernel
        # arm (Exp 7), so force the flag instead of trusting the composed
        # config to carry it. Silently dropped, it would make this arm a
        # byte-identical duplicate of fa_* and the comparison vacuous.
        kwargs["fa_kernel_softmax"] = True
        return LSTMComboAxial(variant=variant, dims=_FA_SM_DIMS[variant],
                              upstream_fa=True, **kwargs)
    if variant in _FA_DIMS:
        return LSTMComboAxial(variant=variant, dims=_FA_DIMS[variant],
                              upstream_fa=True, **kwargs)
    if variant in _ACA_DIMS:
        return LSTMComboAxial(variant=variant, dims=_ACA_DIMS[variant],
                     cross_axis=True, **kwargs)
    if variant in _AUTHORS_CAFA_DIMS:
        return LSTMComboAxial(variant=variant, dims=_AUTHORS_CAFA_DIMS[variant],
                              upstream_fa=True, **kwargs)
    raise ValueError(
        f"Unknown / unsupported LSTM variant: {variant!r}. Expected onehot, "
        "embeddings, or asa/aca/fa/fa_local_{2d,3d,4d}."
    )


ModelFactory.register("lstm", make_lstm)
