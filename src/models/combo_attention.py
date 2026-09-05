"""
Shared cross-attention and FiLM modules for combo-window models.

Copy-pasted verbatim from ``paper included/{LSTM,GRU}/model.py`` (those
two files defined identical CA, LatentCA, and FiLM classes). Pulled into
this single module so ``src/models/lstm.py`` and ``src/models/gru.py``
share one implementation rather than carrying duplicate copies.

The math is unchanged from the paper. If you spot a divergence vs
``paper included``, fix it here.

NOTE: these are intentionally NOT in ``src/models/attention.py``. That file
holds the generic ``MultiHeadAttention`` / ``PositionalEncoding`` blocks (still
used by ``transformer.py``) plus ``CrossAttentionModule`` / ``FiLMModule``
(used by ``encodings.py``); those are per-series building blocks, whereas
everything here operates across the combo lattice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# Compatibility shim copied from paper. Some PyTorch builds lack nn.init.full_.
if not hasattr(nn.init, "full_"):
    def _full_(tensor, fill_value):
        return nn.init.constant_(tensor, float(fill_value))
    nn.init.full_ = _full_


class GroupSelfAttention(nn.Module):
    """Single-head scaled dot-product SELF-attention across groups, per timestep.

    Q, K and V are all projections of the same input ``x``, which makes this
    self-attention by definition. It was called ``CrossAttention`` because the
    predecessor code meant "attention ACROSS groups" -- a collision with the
    standard term, in which cross-attention draws Q from one source and K/V
    from another. The upstream CaFA kernel keeps that distinction too
    (``LowRankKernel`` defaults ``u_y = u_x``; its ``CABlock`` is the genuinely
    cross-attentive one). ``LatentCrossAttention`` below IS cross-attention.

    Input:
        x: (B, L, G, Fin)
        mask_g (optional): (B, G) or (B, L, G), 1=valid, 0=padded

    Output:
        h: (B, L, G, H)
        attn_weights: (B, L, G, G)
    """

    def __init__(self, feature_dim: int, hidden_dim: int, attn_dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask_g: Optional[torch.Tensor] = None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.einsum("blgh,blkh->blgk", Q, K) * self.scale

        if mask_g is not None:
            if mask_g.dim() == 2:
                mask_g = mask_g[:, None, :]
            key_mask = (mask_g[:, :, None, :] > 0).to(scores.dtype)
            scores = scores.masked_fill(key_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        h = torch.einsum("blgk,blkh->blgh", attn, V)
        h = self.out_norm(h)
        return h, attn
class AxialComboSA(nn.Module):
    """Axial SELF-attention applied separately along each chosen lattice axis
    (axial attention, Ho et al. 2019). The flat combo set is scattered onto
    the dense ``(State, Commodity, Flow)`` lattice, cross-attention is run
    independently along each selected axis (padded/non-kept cells masked
    out), then the kept-combo cells are gathered back. The number of axes
    mirrors the S4ND grid variants:

        dims=2 -> attend along State                  (cf. s4nd_2d)
        dims=3 -> attend along State, then Commodity  (cf. s4nd_3d)
        dims=4 -> attend along State, Commodity, Flow (cf. s4nd_4d)

    Attention along each axis is self-attention (see ``GroupSelfAttention``);
    nothing here draws queries from a second source.

    ``axis_identity=True`` adds a learned identity embedding per PROMOTED
    axis to each cell's features after ``in_proj`` — the categorical analogue
    of the per-axis positional encodings that axial-attention models always
    carry (Ho et al. 2019; Axial-DeepLab; CaFA's positional/distance
    encoding). Without it the module is exactly permutation-equivariant along
    the attended axes: attention can key on a cell's current VALUES but not on
    WHICH state/commodity/flow it is. (The non-promoted axes are covered by
    ``LeftoverEncoder`` instead.) Default **False**: the benchmark's Tests 2/3
    ran identity-blind, so OFF preserves their semantics; Test 4 opts in via
    ``--axis-identity`` for the fair identity-aware comparison arm.

    Input:  x  (B, L, G, F)
    Output: h  (B, L, G, H)
    """

    _AXES = {2: (0,), 3: (0, 1), 4: (0, 1, 2)}

    def __init__(self, features_per_group, hidden_dim, lattice_dims,
                 combo_coords, dims, attn_dropout: float = 0.0,
                 axis_identity: bool = False):
        super().__init__()
        assert dims in self._AXES, f"dims must be 2, 3 or 4 (got {dims})"
        self.S, self.C, self.Fl = (int(v) for v in lattice_dims)
        self.N = self.S * self.C * self.Fl
        self.dims = dims
        self.axes = self._AXES[dims]

        coords = torch.as_tensor(combo_coords, dtype=torch.long)            # (G, 3)
        flat = coords[:, 0] * (self.C * self.Fl) + coords[:, 1] * self.Fl + coords[:, 2]
        self.register_buffer("flat_idx", flat)                              # (G,)
        self.register_buffer("axis_coords", coords, persistent=False)       # (G, 3)
        cell_valid = torch.zeros(self.N)
        cell_valid[flat] = 1.0
        self.register_buffer("cell_valid", cell_valid.view(self.S, self.C, self.Fl))
        self.register_buffer("vmask6",
                             cell_valid.view(1, 1, self.S, self.C, self.Fl, 1))

        self.in_proj = nn.Linear(features_per_group, hidden_dim)
        if axis_identity:
            sizes = (self.S, self.C, self.Fl)
            self.axis_embeds = nn.ModuleList()
            for ax in self.axes:
                emb = nn.Embedding(sizes[ax], hidden_dim)
                nn.init.normal_(emb.weight, std=0.02)
                self.axis_embeds.append(emb)
        else:
            self.axis_embeds = None
        # Subclasses (fa_local, fa) replace this backend by reassigning
        # ``self.sa`` to an empty ModuleList. Do NOT keep a second attribute
        # pointing at the same list: rebinding one name would leave the other
        # holding the parent's projections, silently re-registering unused
        # parameters and inflating every params/FLOPs figure in the cost panel.
        self.sa = nn.ModuleList(
            [GroupSelfAttention(hidden_dim, hidden_dim, attn_dropout)
             for _ in self.axes]
        )

    def _dense_features(self, x):
        """in_proj + promoted-axis identity embeddings + scatter onto the
        dense lattice. Returns (B, L, S, C, Fl, H); non-kept cells are zero."""
        B, L, G, _ = x.shape
        h = self.in_proj(x)                                # (B, L, G, H)
        if self.axis_embeds is not None:
            for ax, emb in zip(self.axes, self.axis_embeds):
                h = h + emb(self.axis_coords[:, ax])       # (G, H) broadcast
        dense = x.new_zeros(B, L, self.N, h.shape[-1])
        dense[:, :, self.flat_idx, :] = h
        return dense.view(B, L, self.S, self.C, self.Fl, h.shape[-1])

    def _axis_sa(self, dense, ax, sa):
        """Self-attend along lattice axis ``ax`` in {0:State,1:Commodity,2:Flow}.

        ``dense`` is (B, L, S, C, Fl, H). We fold the two non-attended lattice
        axes into the batch so CrossAttention attends across the chosen axis.
        """
        B, L, S, C, Fl, H = dense.shape
        sizes = (S, C, Fl)
        pos = 2 + ax
        others = [2 + i for i in range(3) if i != ax]
        perm = [0] + others + [1, pos, 5]                  # (B, o1, o2, L, A, H)
        d = dense.permute(*perm).contiguous()
        o1, o2 = (sizes[i] for i in range(3) if i != ax)
        A = sizes[ax]
        d = d.view(B * o1 * o2, L, A, H)
        # validity mask for this axis: cell_valid -> (o1, o2, A) -> (B*o1*o2, A)
        vm = self.cell_valid.permute(*[i for i in range(3) if i != ax], ax).contiguous()
        vm = vm.view(1, o1 * o2, A).expand(B, o1 * o2, A).reshape(B * o1 * o2, A)
        # Lines with NO valid cell (e.g. a (commodity,flow) column with zero kept
        # states) would mask every key -> softmax over all -inf -> NaN in both
        # forward and backward. Give such dead lines an all-valid mask (their
        # output is zeroed by vmask6 afterwards), so no query row is ever fully
        # masked and gradients stay finite.
        dead = vm.sum(dim=-1, keepdim=True) == 0
        vm = torch.where(dead, torch.ones_like(vm), vm)
        out, _ = sa(d, mask_g=vm)                          # (B*o1*o2, L, A, H)
        # No nan_to_num here. The dead-line fix above is the actual guarantee
        # that no query row is fully masked, so this tensor is NaN-free by
        # construction; scrubbing it again cost a full out-of-place pass over
        # the dense lattice on every axis of every layer (~3.9 ms/step on an
        # H20 at benchmark scale) to change nothing. Worse, it made genuine
        # divergence finite, which is how a diverged run used to reach the
        # leaderboard looking merely bad. Divergence is now caught loudly at
        # the loss instead -- see Trainer.NonFiniteLossError.
        out = out.view(B, o1, o2, L, A, H)
        inv = [0] * 6
        for new_pos, old in enumerate(perm):
            inv[old] = new_pos
        # Deliberately NOT .contiguous(): the caller immediately evaluates
        # (dense + this) * vmask6, and that elementwise add materialises a
        # contiguous result anyway. Forcing it here copied the lattice twice.
        # Verified bitwise identical end-to-end.
        return out.permute(*inv)

    def forward(self, x):                                  # (B, L, G, F)
        B, L, G, _ = x.shape
        dense = self._dense_features(x)                    # (B, L, S, C, Fl, H)
        for ax, sa in zip(self.axes, self.sa):
            dense = (dense + self._axis_sa(dense, ax, sa)) * self.vmask6
        dense = dense.view(B, L, self.N, -1)
        return dense[:, :, self.flat_idx, :]               # (B, L, G, H)


class LeftoverEncoder(nn.Module):
    """Encode the categoricals NOT promoted to an axial axis, per group.

    Axial ``dims`` 2/3/4 promote {State} / {State,Commodity} /
    {State,Commodity,Flow} to attention axes (see ``AxialComboSA._AXES``).
    Whatever is left must be encoded, or the axial model cannot tell groups
    apart along the un-attended axis (they are merely folded into the batch).
    Emits ``(G, out_dim)`` to concat onto the per-group features BEFORE the
    axial ``in_proj`` — this is the encoder x dimensionality wiring (PLAN.md §3):

        dims=2 -> encode Commodity + Flow    dims=3 -> encode Flow    dims=4 -> none

        encoder='embeddings' : one nn.Embedding per leftover categorical (cheap;
                               2d -> commodity(88)+flow(2)=90 dims).
        encoder='onehot'     : one-hot per leftover categorical. E can be huge —
                               2d includes commodity -> C dims (C=1263 => ~20 GB
                               per combo batch); fine at 3d(flow=2)/4d(none).

    ``combo_coords`` is ``(G, 3)`` in (state, commodity, flow) = axis (0,1,2) order.
    """

    def __init__(self, dims, lattice_dims, combo_coords, encoder="embeddings",
                 state_embed_dim=7, comm_embed_dim=88, flow_embed_dim=2):
        super().__init__()
        card = tuple(int(v) for v in lattice_dims)          # (S, C, Fl)
        edim = (state_embed_dim, comm_embed_dim, flow_embed_dim)
        promoted = AxialComboSA._AXES[dims]
        self.leftover = [a for a in (0, 1, 2) if a not in promoted]
        self.encoder = encoder
        coords = torch.as_tensor(combo_coords, dtype=torch.long)   # (G, 3)
        self.out_dim = 0
        if encoder == "embeddings":
            self.embeds = nn.ModuleList()
            for a in self.leftover:
                emb = nn.Embedding(card[a], edim[a])
                nn.init.normal_(emb.weight, std=0.02)
                self.embeds.append(emb)
                self.out_dim += edim[a]
            self.register_buffer("coords", coords, persistent=False)
        elif encoder == "onehot":
            parts = [F.one_hot(coords[:, a], card[a]).float() for a in self.leftover]
            oh = torch.cat(parts, dim=1) if parts else coords.new_zeros((coords.shape[0], 0)).float()
            self.register_buffer("onehot", oh, persistent=False)
            self.out_dim = int(oh.shape[1])
        else:
            raise ValueError(f"LeftoverEncoder: unknown encoder {encoder!r}")

    def forward(self):                                      # -> (G, out_dim) or None
        if self.out_dim == 0:
            return None
        if self.encoder == "embeddings":
            return torch.cat(
                [emb(self.coords[:, a]) for emb, a in zip(self.embeds, self.leftover)],
                dim=1,
            )
        return self.onehot

# Legacy variant spellings -> current ones. "cafa_*" claimed the authors'
# MODEL name (CaFA = ForeCasting with Factorized Attention) for a local
# reimplementation of their FA operator; "cross_attention_*" described
# self-attention. Old manifests and checkpoints still resolve through here.
LEGACY_VARIANTS = {
    **{f"cafa_{d}d": f"fa_local_{d}d" for d in (2, 3, 4)},
    **{f"cross_attention_{d}d": f"asa_{d}d" for d in (2, 3, 4)},
}


def canonical_variant(variant: str) -> str:
    """Map a legacy variant spelling onto its current name."""
    return LEGACY_VARIANTS.get(variant, variant)


# Legacy names. The operators are self-attention; these aliases exist so old
# checkpoints, manifests and third-party imports keep resolving.
CrossAttention = GroupSelfAttention
AxialComboCA = AxialComboSA
