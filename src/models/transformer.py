"""
Transformer model for time series forecasting.

Implements a standard transformer encoder with configurable encoding strategy.

Note: This is a simplified implementation. For full functionality, complete
the transformer architecture based on your specific requirements.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseModel, ModelFactory
from .attention import PositionalEncoding
from .fa_local import LocalFactorizedAttention
from .aca import AxialCrossAttention
from .fa import FactorizedAttention
from .combo_attention import AxialComboSA, LeftoverEncoder, canonical_variant


class TransformerModel(BaseModel):
    """
    Transformer model with configurable encoding strategy.

    Architecture:
    1. Encode categorical features
    2. Add positional encoding
    3. Process through transformer encoder layers
    4. Output layer for prediction
    """

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 512,
        activation: str = "gelu",
        max_seq_len: int = 1000,
        positional_encoding: str = "sinusoidal",
        **kwargs
    ) -> None:
        """
        Build transformer model.

        Args:
            hidden_size: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            d_ff: Feed-forward dimension
            activation: Activation function
            max_seq_len: Maximum sequence length
            positional_encoding: Type of positional encoding
            **kwargs: Additional arguments
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Project input to model dimension
        self.input_projection = nn.Linear(self.input_dim, hidden_size)

        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.pos_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
            self.use_learned_pos = False
        else:
            self.pos_encoding = nn.Embedding(max_seq_len, hidden_size)
            self.use_learned_pos = True

        # Transformer encoder. norm_first=True (pre-norm) is far more
        # stable than the default post-norm without learning-rate warmup;
        # post-norm + lr=1e-4 still diverged to NaN on this dataset
        # within epoch 0. Pre-norm trains cleanly with the same lr.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_numeric: (batch_size, seq_len, num_features)
            state_ids: (batch_size,)
            comm_ids: (batch_size,)
            flow_ids: (batch_size,)

        Returns:
            predictions: (batch_size, 2)
        """
        # Encode features
        x_encoded = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)

        # Project to model dimension
        x = self.input_projection(x_encoded)  # (B, seq_len, hidden_size)

        # Add positional encoding
        if self.use_learned_pos:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.pos_encoding(positions)
        else:
            x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, seq_len, hidden_size)

        # Use last time step for prediction
        x_last = x[:, -1, :]  # (B, hidden_size)

        # Output layer
        predictions = self.output_layer(x_last)  # (B, 2)

        return predictions


# Register model
class TransformerComboAxial(BaseModel):
    """Transformer with S4ND-style AXIAL cross-attention across the combo
    lattice, then a per-group Transformer encoder over time.

    Pipeline:
        1. AxialComboSA over the (S,C,Flow) lattice -> (B, L, G, H)
        2. Residual from in_proj(x_raw) + LayerNorm
        3. Permute to (B,G,L,H), reshape to (B*G, L, H), per-group Transformer
        4. Output MLP on the last timestep -> (B, G, 2)
    """

    requires_combo_loader = True

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 512,
        activation: str = "gelu",
        max_seq_len: int = 1000,
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
        ), "TransformerComboAxial needs combo metadata from the combo dataloader."
        self.encoder = nn.Identity()
        self.hidden_size = hidden_size
        self.num_groups = num_combos
        # Encode categoricals not promoted to an axial axis; append to per-group
        # feats (encoder x dims, PLAN.md §3). Grows feat_dim for BOTH the axial
        # module and the raw residual projection.
        self.leftover = LeftoverEncoder(
            dims, lattice_dims, combo_coords, encoder=combo_encoder,
            state_embed_dim=state_embed_dim, comm_embed_dim=comm_embed_dim,
            flow_embed_dim=flow_embed_dim,
        )
        feat_dim = features_per_group + self.leftover.out_dim
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
        self.in_proj_raw = nn.Linear(feat_dim, hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, activation=activation, batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor = None,
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc = self.leftover()                                  # (G, E) or None
        if enc is not None:
            B, L, G, _ = x_numeric.shape
            x_numeric = torch.cat(
                [x_numeric, enc.to(x_numeric.dtype)[None, None].expand(B, L, G, -1)],
                dim=-1,
            )
        h = self.axial(x_numeric)                              # (B, L, G, H)
        h = self.post_norm(h + self.in_proj_raw(x_numeric))   # (B, L, G, H)
        B, L, G, H = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B * G, L, H)        # per-group sequence
        h = self.pos_encoding(h)
        h = self.transformer_encoder(h)                       # (B*G, L, H)
        return self.output_layer(h[:, -1, :]).reshape(B, G, -1)


_AXIAL_DIMS = {f"asa_{d}d": d for d in (2, 3, 4)}      # axial SELF-attention
_CAFA_DIMS = {f"fa_local_{d}d": d for d in (2, 3, 4)}  # local FA reimplementation
_FA_DIMS = {f"fa_{d}d": d for d in (2, 3, 4)}          # authors' FA operator
_FA_SM_DIMS = {f"fa_sm_{d}d": d for d in (2, 3, 4)}   # ... on their softmax switch
_ACA_DIMS = {f"aca_{d}d": d for d in (2, 3, 4)}        # axial CROSS-attention
_AUTHORS_CAFA_DIMS = {
    "authors_cafa_2d": 2, "authors_cafa_3d": 3, "authors_cafa_4d": 4,
}


def make_transformer(variant: str, **kwargs) -> BaseModel:
    """Flat onehot/embeddings -> TransformerModel; axial combo variants ->
    TransformerComboAxial (S4ND-style cross_attention_{2d,3d,4d})."""
    # Accept legacy spellings (cafa_*, cross_attention_*d) so old
    # manifests and checkpoints keep resolving. See LEGACY_VARIANTS.
    variant = canonical_variant(variant)
    if variant in ("onehot", "embeddings"):
        return TransformerModel(variant=variant, **kwargs)
    if variant in _AXIAL_DIMS:
        return TransformerComboAxial(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)
    if variant in _CAFA_DIMS:
        return TransformerComboAxial(variant=variant, dims=_CAFA_DIMS[variant],
                                     factorized=True, **kwargs)
    if variant in _FA_SM_DIMS:
        # The variant name is the contract: fa_sm_* IS the softmax-kernel
        # arm (Exp 7), so force the flag instead of trusting the composed
        # config to carry it. Silently dropped, it would make this arm a
        # byte-identical duplicate of fa_* and the comparison vacuous.
        kwargs["fa_kernel_softmax"] = True
        return TransformerComboAxial(variant=variant, dims=_FA_SM_DIMS[variant],
                                     upstream_fa=True, **kwargs)
    if variant in _FA_DIMS:
        return TransformerComboAxial(variant=variant, dims=_FA_DIMS[variant],
                                     upstream_fa=True, **kwargs)
    if variant in _ACA_DIMS:
        return TransformerComboAxial(variant=variant, dims=_ACA_DIMS[variant],
                     cross_axis=True, **kwargs)
    if variant in _AUTHORS_CAFA_DIMS:
        return TransformerComboAxial(
            variant=variant, dims=_AUTHORS_CAFA_DIMS[variant],
            upstream_fa=True, **kwargs,
        )
    # No catch-all: an unrecognised variant must fail loudly. Falling back to
    # the flat model meant `--variant grid_2d` (an SSM-only grid tag) silently
    # trained a per-series transformer that the run name claimed was axial.
    raise ValueError(
        f"Unknown / unsupported transformer variant: {variant!r}. Expected "
        "onehot, embeddings, or asa/aca/fa/fa_local_{2d,3d,4d}."
    )


ModelFactory.register("transformer", make_transformer)
