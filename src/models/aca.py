"""ACA: axial CROSS-attention over the categorical combo lattice.

Everything else in this roster is axial SELF-attention. ``AxialComboSA`` runs a
full attention along each promoted axis, and ``fa`` / ``fa_local`` factorize the
same self-attention into shared per-axis kernels. In all of them Q and K are
projections of the SAME source, so no arm answers the question "does letting one
categorical axis condition on a DIFFERENT one buy anything?".

Provenance, stated plainly
--------------------------
There is no canonical "axial cross-attention" paper to clone, and it is worth
being precise about why:

* Ho et al. 2019 (arXiv:1912.12180), the origin of axial attention, describes it
  as "a simple generalization of self-attention" -- attention along one axis
  with the others folded into the batch. That is ``AxialComboSA``, not this.
* Axial-DeepLab (Wang et al. 2020, arXiv:2003.07853) adds position-sensitive
  relative encodings to Q/K/V. Still self-attention; the categorical analogue of
  its positional term is this repo's ``axis_identity`` (Test 4).
* The closest published pattern is axial-centric cross-plane attention for 3D
  medical imaging (arXiv:2602.21636): the primary plane supplies queries, the
  complementary planes supply keys and values, with "directional cross-plane
  fusion" and ablations over "axial-centric querying" and "QKV allocation".

This module adapts that pattern to a categorical (State, Commodity, Flow)
lattice. It is OUR construction, not a reproduction of any released code, and
the paper should describe it that way.

Mechanism
---------
For each promoted axis A, with M = the flattened COMPLEMENT of A in the lattice:

    U_A     = masked mean of the lattice over every axis except A   -> (B,L,|A|,H)
    U_M     = masked mean of the lattice over A, flattened          -> (B,L,M,H)
    Q = W_q U_A ;  K = W_k U_M ;  V = W_v U_M
    attn    = softmax(Q Kᵀ / sqrt(H))                               -> (B,L,|A|,M)
    ctx[a]  = sum_m attn[a,m] V[m]                                  -> (B,L,|A|,H)
    out     = (lattice + broadcast(ctx along A)) * valid-cell mask

Q and K come from different sources, which is what makes this cross-attention
rather than a renamed self-attention. The hop is directional (A <- complement),
matching "directional cross-plane fusion".

Cost is comparable to ``fa_local``: one |A| x M kernel per (batch, timestep, axis)
instead of a per-line |A| x |A| stack. Sparsity is handled the same way -- masked
mean pooling by valid-cell counts, and complement positions with no valid cell at
all are removed from the softmax so a dead column cannot absorb mass.

Interface-compatible with ``AxialComboSA``: (B, L, G, F) -> (B, L, G, H).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .combo_attention import AxialComboSA


class AxialCrossAttention(AxialComboSA):
    """Directional cross-axis attention: each promoted axis queries the rest."""

    def __init__(self, features_per_group, hidden_dim, lattice_dims,
                 combo_coords, dims, attn_dropout: float = 0.0,
                 axis_identity: bool = False):
        super().__init__(features_per_group, hidden_dim, lattice_dims,
                         combo_coords, dims, attn_dropout=attn_dropout,
                         axis_identity=axis_identity)
        # Drop the parent's per-axis SELF-attention backend. Reassign the same
        # attribute rather than adding another name to it: two names on one
        # ModuleList would keep these projections registered and silently
        # inflate every parameter/FLOP figure.
        self.sa = nn.ModuleList()

        H = int(hidden_dim)
        self.q_projs = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in self.axes])
        self.k_projs = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in self.axes])
        self.v_projs = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in self.axes])
        self.scale = H ** -0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.out_norm = nn.LayerNorm(H)

        # Masked-mean denominators. The scattered lattice is exactly zero on
        # absent cells, so a plain sum over these counts IS the masked mean.
        for ax in self.axes:
            others = [i for i in range(3) if i != ax]
            axis_count = self.cell_valid.sum(dim=tuple(others)).clamp_min(1.0)
            self.register_buffer(f"axis_count{ax}", axis_count)          # (|A|,)

            comp_count = self.cell_valid.sum(dim=ax)                      # (o1, o2)
            self.register_buffer(f"comp_count{ax}", comp_count.clamp_min(1.0))
            # A complement position with no valid cell carries no information;
            # leaving it in the softmax would hand it attention mass anyway.
            self.register_buffer(f"comp_valid{ax}", (comp_count > 0).reshape(-1))

    def _cross_axis(self, dense, ax, q_proj, k_proj, v_proj):
        """One directional hop: axis ``ax`` attends over its complement."""
        B, L, S, C, Fl, H = dense.shape
        others = [i for i in range(3) if i != ax]

        # Query source: the axis being updated.
        pool_dims = tuple(2 + o for o in others)
        u_axis = dense.sum(dim=pool_dims) / getattr(
            self, f"axis_count{ax}").view(1, 1, -1, 1)                    # (B,L,|A|,H)

        # Key/value source: everything EXCEPT that axis. This is the whole
        # point -- Q and K do not come from the same tensor.
        u_comp = dense.sum(dim=2 + ax) / getattr(
            self, f"comp_count{ax}").view(1, 1, *getattr(self, f"comp_count{ax}").shape, 1)
        u_comp = u_comp.reshape(B, L, -1, H)                              # (B,L,M,H)

        q = q_proj(u_axis)
        k = k_proj(u_comp)
        v = v_proj(u_comp)
        scores = torch.einsum("blad,blmd->blam", q, k) * self.scale
        comp_valid = getattr(self, f"comp_valid{ax}").view(1, 1, 1, -1)
        scores = scores.masked_fill(~comp_valid, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.einsum("blam,blmh->blah", attn, v)                    # (B,L,|A|,H)

        # Broadcast the per-axis context back over the lattice.
        shape = [B, L, 1, 1, 1, H]
        shape[2 + ax] = ctx.shape[2]
        return ctx.reshape(shape)

    def forward(self, x):                                     # (B, L, G, F)
        B, L, G, _ = x.shape
        dense = self._dense_features(x)                       # (B,L,S,C,Fl,H)
        for ax, q_proj, k_proj, v_proj in zip(
                self.axes, self.q_projs, self.k_projs, self.v_projs):
            ctx = self._cross_axis(dense, ax, q_proj, k_proj, v_proj)
            dense = (dense + ctx) * self.vmask6
        dense = self.out_norm(dense) * self.vmask6
        dense = dense.view(B, L, self.N, -1)
        return dense[:, :, self.flat_idx, :]                  # (B, L, G, H)


__all__ = ["AxialCrossAttention"]
