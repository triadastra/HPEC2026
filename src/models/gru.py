"""
GRU models for time series forecasting.

Three concrete classes, all in this one file, registered as ``"gru"``:

================  ===========================  ================  ============================
Variant flag      Class                        Input shape       Paper-included file
================  ===========================  ================  ============================
onehot            GRUModel                     (B, L, F)         GRU/model.py::GRU
embeddings        GRUModel                     (B, L, F)         GRU/model.py::GRU
cross_attention   GRUComboCAOnly               (B, L, G, F)      GRU/model.py::GRUWithCrossAttentionOnly
film_attention    GRUComboCAFiLM               (B, L, G, F)      GRU/model.py::GRUWithCrossAttention
================  ===========================  ================  ============================

All architectures follow ``code/paper included/GRU/model.py`` (no imports
across the repo boundary). The two combo classes share a small backbone
helper to remove the duplication that existed in the paper code. The math
is unchanged EXCEPT for one correctness fix: the per-group flatten in
``_combo_temporal_head`` now permutes ``(B, L, G, H) -> (B, G, L, H)``
before reshaping to ``(B*G, L, H)``. The paper code reshaped directly,
which interleaved group and timestep and fed the recurrence temporally-
scrambled sequences. See the comment in ``_combo_temporal_head``.

A single registration ``ModelFactory.register("gru", make_gru)``
dispatches to the right class based on the ``variant`` argument.

Combo variants advertise ``requires_combo_loader = True`` so that
``scripts/train.py`` knows to ask ``TradeDataPipeline`` for the
``(B, L, G, F)`` combo-window dataloader.

The cross-attention / FiLM modules live in ``src/models/combo_attention.py``
and are shared between ``lstm.py`` and ``gru.py`` — paper had identical
copies in both files.
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
# Variant 1: normal 1D GRU
# Paper class: `GRU` in paper included/GRU/model.py
# Used by variants: onehot, embeddings
# ---------------------------------------------------------------------------

class GRUModel(BaseModel):
    """Per-series GRU (paper-equivalent base block).

        input_proj : Linear(input_dim -> hidden_size)
        gru        : GRU(hidden_size -> hidden_size, num_layers, dropout)
        output_proj: Linear(hidden_size -> 2)

    Prediction taken from ``gru_out[:, -1, :]`` — last timestep of the
    full output sequence. For a unidirectional single-batch GRU this
    equals ``h_n[-1]``, but staying with the paper's expression preserves
    state-dict key parity.
    """

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
        self.gru = nn.GRU(
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
        group_mask: Optional[torch.Tensor] = None,  # ignored; kept for uniform call signature
    ) -> torch.Tensor:
        x = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)
        x = self.input_proj(x)
        out, _ = self.gru(x)
        return self.output_proj(out[:, -1, :])


# ---------------------------------------------------------------------------
# Shared backbone helpers (one of each kept in this file rather than a
# global module, because both helpers reference attribute name `gru` —
# keeping it local makes the file self-contained).
# ---------------------------------------------------------------------------

def _build_combo_backbone(
    module: nn.Module,
    features_per_group: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> None:
    """Attach in_proj, post_norm, per-group GRU, output_proj to ``module``."""
    module.hidden_size = hidden_size
    module.in_proj = nn.Linear(features_per_group, hidden_size)
    module.post_norm = nn.LayerNorm(hidden_size)
    module.gru = nn.GRU(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0.0,
        batch_first=True,
    )
    module.output_proj = nn.Linear(hidden_size, 2)


def _combo_temporal_head(
    module: nn.Module,
    h_modulated: torch.Tensor,  # (B, L, G, H) — already attended (+ optionally FiLM'd)
    x_raw: torch.Tensor,        # (B, L, G, F) — for residual shortcut
) -> torch.Tensor:
    """Apply residual + per-group GRU + output projection. Returns (B, G, 2)."""
    B, L, G, _ = x_raw.shape
    shortcut = module.in_proj(x_raw)                                # (B, L, G, H)
    h = module.post_norm(h_modulated + shortcut)                    # (B, L, G, H)
    # Flatten groups into the batch so each row is ONE group's full time
    # sequence before the GRU. The group axis must be moved next to batch
    # first: a bare `reshape(B*G, L, H)` on a (B, L, G, H) tensor interleaves
    # group and timestep, feeding the GRU temporally-scrambled sequences.
    # (This corrects the paper-included code, which omitted the permute.)
    h = h.permute(0, 2, 1, 3).reshape(B * G, L, module.hidden_size)  # (B*G, L, H)
    out, _ = module.gru(h)
    # out[:, -1] is the last timestep per group; reshape back to (B, G, ·)
    # in the same b*G+g order the rows were created in.
    return module.output_proj(out[:, -1, :]).reshape(B, G, -1)


# ---------------------------------------------------------------------------
# Variant 2: cross-attention ONLY (no FiLM)
# Paper class: `GRUWithCrossAttentionOnly` in paper included/GRU/model.py
# Used by variant: cross_attention
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Variant 3: cross-attention + FiLM
# Paper class: `GRUWithCrossAttention` in paper included/GRU/model.py
# Used by variant: film_attention
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

class GRUComboAxial(BaseModel):
    """GRU over groups with AXIAL cross-attention (S4ND-style 2d/3d/4d).

    Instead of one flat cross-attention over all combos, attend separately
    along each lattice axis (State / +Commodity / +Flow), mirroring S4ND's
    separable grid mixing. Then the same per-group GRU over time.

    Pipeline:
        1. AxialComboSA over the (S,C,Flow) lattice -> (B, L, G, H)
        2. Residual from in_proj(x) + LayerNorm
        3. Permute to (B,G,L,H), reshape to (B*G, L, H), per-group GRU
        4. Linear(H -> 2) on last timestep

    Output: (B, G, 2)
    """

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
            "GRUComboAxial needs num_combos, features_per_group, combo_coords "
            "and lattice_dims from the combo dataloader (wired by train.py)."
        )
        self.encoder = nn.Identity()  # combo input is raw (B,L,G,F); see GRUComboCAOnly
        self.num_groups = num_combos
        # Encode the categoricals NOT promoted to an axial axis (2d: comm+flow,
        # 3d: flow, 4d: none) and append to the per-group features -- this is the
        # encoder x dimensionality wiring (PLAN.md §3). Test 2 -> combo_encoder=
        # onehot ; Test 3 -> embeddings. onehot@2d is memory-heavy (commodity).
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


_AXIAL_DIMS = {f"asa_{d}d": d for d in (2, 3, 4)}      # axial SELF-attention
_CAFA_DIMS = {f"fa_local_{d}d": d for d in (2, 3, 4)}  # local FA reimplementation
_FA_DIMS = {f"fa_{d}d": d for d in (2, 3, 4)}          # authors' FA operator
_FA_SM_DIMS = {f"fa_sm_{d}d": d for d in (2, 3, 4)}   # ... on their softmax switch
_ACA_DIMS = {f"aca_{d}d": d for d in (2, 3, 4)}        # axial CROSS-attention
_AUTHORS_CAFA_DIMS = {
    "authors_cafa_2d": 2, "authors_cafa_3d": 3, "authors_cafa_4d": 4,
}


def make_gru(variant: str, **kwargs) -> BaseModel:
    """Pick the right GRU class for the requested variant."""
    # Accept legacy spellings (cafa_*, cross_attention_*d) so old
    # manifests and checkpoints keep resolving. See LEGACY_VARIANTS.
    variant = canonical_variant(variant)
    if variant in ("onehot", "embeddings"):
        return GRUModel(variant=variant, **kwargs)
    if variant in _AXIAL_DIMS:
        return GRUComboAxial(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)
    if variant in _CAFA_DIMS:
        return GRUComboAxial(variant=variant, dims=_CAFA_DIMS[variant],
                             factorized=True, **kwargs)
    if variant in _FA_SM_DIMS:
        # The variant name is the contract: fa_sm_* IS the softmax-kernel
        # arm (Exp 7), so force the flag instead of trusting the composed
        # config to carry it. Silently dropped, it would make this arm a
        # byte-identical duplicate of fa_* and the comparison vacuous.
        kwargs["fa_kernel_softmax"] = True
        return GRUComboAxial(variant=variant, dims=_FA_SM_DIMS[variant],
                             upstream_fa=True, **kwargs)
    if variant in _FA_DIMS:
        return GRUComboAxial(variant=variant, dims=_FA_DIMS[variant],
                             upstream_fa=True, **kwargs)
    if variant in _ACA_DIMS:
        return GRUComboAxial(variant=variant, dims=_ACA_DIMS[variant],
                     cross_axis=True, **kwargs)
    if variant in _AUTHORS_CAFA_DIMS:
        return GRUComboAxial(variant=variant, dims=_AUTHORS_CAFA_DIMS[variant],
                             upstream_fa=True, **kwargs)
    raise ValueError(
        f"Unknown / unsupported GRU variant: {variant!r}. Expected onehot, "
        "embeddings, or asa/aca/fa/fa_local_{2d,3d,4d}."
    )


ModelFactory.register("gru", make_gru)
