"""
GPT (decoder-only causal Transformer) for time series forecasting.

One concrete class registered as ``"gpt"``:

================  ===========  ================  =============================
Variant flag      Class        Input shape       Paper-included file
================  ===========  ================  =============================
onehot            GPTModel     (B, L, F)         GPT/onehot.py  (FlatGPT)
embeddings        GPTModel     (B, L, F)         GPT/gpt_flat.py (FlatGPT)
================  ===========  ================  =============================

"GPT" in this paper is a **decoder-only, causal-masked Transformer** that
pools the LAST token and regresses the next single step. It is NOT an
autoregressive multi-step decoder — there is no token-by-token generation
loop, temperature, or top-k/top-p sampling (the old placeholder config's
"Generation" knobs never existed in the paper code). Documented here so
nobody mistakes it for a generative LM.

The blocks (``PositionalEncoding`` / ``CausalSelfAttention`` / ``GPTBlock``
/ ``FlatGPT``) are copy-pasted verbatim from
``code/paper included/GPT/gpt_model.py``. The onehot and embeddings paper
files share this identical model and differ only in how categoricals are
encoded; that distinction is handled here by the shared ``EncodingFactory``
(via ``BaseModel.encode_features``), so one ``GPTModel`` covers both. GPT
has no combo variant in the paper; ``GPTComboAxial`` is a repository extension.

NOTE: This file copies its own PositionalEncoding (no dropout, sinusoidal)
verbatim from the paper rather than reusing src/models/attention.py, to
preserve exact state-dict / numeric parity with paper included/.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseModel, ModelFactory
from .combo_attention import AxialComboSA, LeftoverEncoder, canonical_variant


# ---------------------------------------------------------------------------
# Blocks (verbatim from paper included/GPT/gpt_model.py)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1), :]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.register_buffer("causal_mask", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build causal mask lazily to match sequence length
        L = x.size(1)
        if (self.causal_mask is None) or (self.causal_mask.size(0) < L):
            mask = torch.full((L, L), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            self.causal_mask = mask
        y, _ = self.attn(x, x, x, attn_mask=self.causal_mask[:L, :L])
        return y


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# GPT model wrapped in the BaseModel interface
# Paper class: FlatGPT in paper included/GPT/gpt_model.py
# Used by variants: onehot, embeddings
# ---------------------------------------------------------------------------

class GPTModel(BaseModel):
    """Decoder-only causal Transformer, last-token next-step head.

        in_proj : Linear(input_dim -> d_model)
        pos     : sinusoidal PositionalEncoding
        blocks  : num_layers × GPTBlock (causal self-attn + MLP, pre-norm)
        norm    : LayerNorm(d_model)   (applied to the last token)
        head    : MLP(d_model -> d_model -> 2)

    ``input_dim`` already includes the categorical encoding (embeddings or
    one-hot) chosen by ``variant``.
    """

    requires_combo_loader = False

    def _build_model(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 3.0,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.in_proj = nn.Linear(self.input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,  # ignored; uniform signature
    ) -> torch.Tensor:
        x = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)  # (B, L, input_dim)
        h = self.in_proj(x)
        h = self.pos(h)
        for blk in self.blocks:
            h = blk(h)
        h_last = self.norm(h[:, -1])
        return self.head(h_last)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

class GPTComboAxial(BaseModel):
    """GPT (causal Transformer) with S4ND-style AXIAL cross-attention across
    the combo lattice, then per-group causal GPT blocks over time.

    Pipeline:
        1. Encode categorical axes not promoted into axial attention
        2. AxialComboSA over the (S,C,Flow) lattice -> (B, L, G, D)
        3. Residual from in_proj(x_with_leftovers) + LayerNorm
        4. Permute to (B,G,L,D), reshape to (B*G, L, D), per-group GPT blocks
        5. LayerNorm + MLP head on the last timestep -> (B, G, 2)
    """

    requires_combo_loader = True

    def _build_model(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        dims: int = 4,
        axis_identity: bool = False,
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
        ), "GPTComboAxial needs combo metadata from the combo dataloader."
        self.encoder = nn.Identity()
        self.d_model = d_model
        self.num_groups = num_combos
        # Axes not promoted into axial attention are folded into the batch.
        # Preserve their identities explicitly so 2-D/3-D groups with the
        # same numeric history do not become indistinguishable.
        self.leftover = LeftoverEncoder(
            dims, lattice_dims, combo_coords, encoder=combo_encoder,
            state_embed_dim=state_embed_dim, comm_embed_dim=comm_embed_dim,
            flow_embed_dim=flow_embed_dim,
        )
        feat_dim = features_per_group + self.leftover.out_dim
        self.axial = AxialComboSA(
            feat_dim, d_model, lattice_dims, combo_coords,
            dims=dims, attn_dropout=attn_dropout, axis_identity=axis_identity,
        )
        self.in_proj_raw = nn.Linear(feat_dim, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor = None,
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc = self.leftover()                                # (G, E) or None
        if enc is not None:
            B, L, G, _ = x_numeric.shape
            x_numeric = torch.cat(
                [x_numeric, enc.to(x_numeric.dtype)[None, None].expand(B, L, G, -1)],
                dim=-1,
            )
        h = self.axial(x_numeric)                            # (B, L, G, D)
        h = self.pre_norm(h + self.in_proj_raw(x_numeric))   # (B, L, G, D)
        B, L, G, D = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B * G, L, D)       # per-group sequence
        h = self.pos(h)
        for blk in self.blocks:
            h = blk(h)
        h_last = self.norm(h[:, -1])                          # (B*G, D)
        return self.head(h_last).reshape(B, G, -1)


_AXIAL_DIMS = {f"asa_{d}d": d for d in (2, 3, 4)}      # axial SELF-attention


def make_gpt(variant: str, **kwargs) -> BaseModel:
    """Flat onehot/embeddings -> GPTModel; axial combo variants ->
    GPTComboAxial (S4ND-style cross_attention_{2d,3d,4d})."""
    # Accept legacy spellings (cafa_*, cross_attention_*d) so old
    # manifests and checkpoints keep resolving. See LEGACY_VARIANTS.
    variant = canonical_variant(variant)
    if variant in ("onehot", "embeddings"):
        return GPTModel(variant=variant, **kwargs)
    if variant in _AXIAL_DIMS:
        return GPTComboAxial(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)
    raise ValueError(
        f"Unknown / unsupported GPT variant: {variant!r}. Expected onehot, "
        "embeddings, or cross_attention_{2d,3d,4d}."
    )


ModelFactory.register("gpt", make_gpt)
