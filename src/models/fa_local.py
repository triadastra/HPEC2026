"""Local reimplementation of CaFA's factorized attention (FA) operator.

NAMING. "CaFA" is the authors' WEATHER MODEL -- ForeCasting with
Factorized Attention (Li et al. 2024). The operator inside it is FA, and
their released code calls it ``FABlockS2``. This module is a local
reimplementation of that operator from the paper, so its variants are
``fa_local_*``; ``src/models/fa.py`` runs the authors' actual components
and owns the unqualified ``fa_*`` name. Neither is CaFA: this benchmark
does not run their weather model.

It is SELF-attention, not cross-attention: Q and K are both projections of
the same pooled tensor U.

``AxialComboSA`` (combo_attention.py) runs a FULL single-head self-attention
sequentially along each lattice axis: every hop has its own Q/K/V projections
and LayerNorm, each hop re-attends over the output of the previous hop, and —
crucially — every line of the lattice gets its OWN attention matrix (the two
non-attended axes are folded into the batch), so the score tensors are
O(N * S_ax) per axis in both compute and memory.

``LocalFactorizedAttention`` is the factorized counterpart (Li et al. 2024,
arXiv:2405.07395). Per attended axis it follows the paper's two-step recipe:

    1. Axial projection (paper Eq. 4): masked-mean-pool the features over all
       OTHER lattice axes (uniform mesh weights — the lattice is categorical)
       and pass through a pointwise two-layer MLP ``gamma``, giving ONE
       1-D function U_ax of shape (B, L, S_ax, H) per axis.
    2. Axial kernel (paper Eq. 5): Q/K from U_ax only, so each axis has ONE
       shared S_ax x S_ax kernel per (batch, timestep) — NOT one per line.

A single shared V projection is then contracted by each kernel along its own
axis in turn (paper Eq. 6). On the dense sublattice the joint operator is a
true Kronecker product of the per-axis kernels:

    A[(s,c,f),(s',c',f')]  =  A_S[s,s'] * A_C[c,c'] * A_F[f,f']

    out = (A_S (x) A_C (x) A_F) . V

This is what makes the kernel cost quadratic in the AXIAL size only: score
tensors are (B, L, S_ax, S_ax) instead of AxialComboSA's
(B * prod(other axes), L, S_ax, S_ax) — e.g. an S*Fl-fold memory reduction on
the commodity axis, which is what OOMed fa_local_4d under the old per-line
implementation.

Deviations from the paper, on purpose:
  - No Bessel distance encoding / spherical positional encoding / cos-latitude
    mesh weights: the (State, Commodity, Flow) axes are categorical, there is
    no metric or quadrature on them.
  - Kernels are softmax-normalised (paper uses LeakyReLU on the raw modulated
    scores). Softmax is kept to match the AxialComboSA baseline so the
    cafa-vs-axial comparison isolates the factorization itself.
  - Sparse lattice: the paper's grid is dense; here only kept combos are valid
    cells. V is exactly zero on invalid cells (bias-free projection onto a
    zero-scattered tensor), and after each contraction the output is
    renormalised per line by the softmax mass that fell on VALID keys, so each
    output stays a convex combination of valid values. This per-line scalar
    rescale is the only departure from a strict Kronecker product and costs
    O(N * S_ax) elementwise work, not O(N * S_ax) score memory.

Parameter cost per attended axis: 4*H^2 (gamma MLP + Q + K) versus 3*H^2 +
LayerNorm for a full ``CrossAttention`` hop, plus ONE shared V (H^2) and ONE
output LayerNorm for the whole operator. dims=2/3/4 selects the attended axes
exactly like ``AxialComboSA`` (2: State, 3: +Commodity, 4: +Flow).

Interface-compatible with ``AxialComboSA``:  input (B, L, G, F) -> (B, L, G, H);
non-kept lattice cells are masked out of every kernel renormalisation and
zeroed on output.

STATUS: implemented + syntax-checked; smoke-tested on CPU
(``python -m pytest tests/test_fa_local_smoke.py``). Not yet trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .combo_attention import AxialComboSA


class LocalFactorizedAttention(AxialComboSA):
    """Kronecker-factorized axial SELF-attention, local build. See module docstring."""

    # (B, L, S, C, Fl, H): dims to pool out for each attended axis (paper Eq. 4)
    _POOL_DIMS = {0: (3, 4), 1: (2, 4), 2: (2, 3)}
    # per-axis einsums: contract shared kernel (B,L,q,k) with the running value
    # tensor (paper Eq. 6), and with cell_valid for the per-line renormaliser
    _CONTRACT = {
        0: ("blqs,blscfh->blqcfh", "blqs,scf->blqcf"),
        1: ("blqc,blscfh->blsqfh", "blqc,scf->blsqf"),
        2: ("blqf,blscfh->blscqh", "blqf,scf->blscq"),
    }

    def __init__(self, features_per_group, hidden_dim, lattice_dims,
                 combo_coords, dims, attn_dropout: float = 0.0,
                 axis_identity: bool = False):
        super().__init__(features_per_group, hidden_dim, lattice_dims,
                         combo_coords, dims, attn_dropout=attn_dropout,
                         axis_identity=axis_identity)
        # Drop the parent's full per-axis GroupSelfAttention stacks (3*H^2 + LN each);
        # replace with the paper's projection (gamma) + factorized per-axis Q/K
        # + ONE shared V + ONE output norm.
        self.sa = nn.ModuleList()
        H = hidden_dim
        self.gammas = nn.ModuleList([
            nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))
            for _ in self.axes
        ])
        self.q_projs = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in self.axes])
        self.k_projs = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in self.axes])
        self.v_proj = nn.Linear(H, H, bias=False)
        self.scale = H ** -0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.out_norm = nn.LayerNorm(H)
        # Valid-cell counts per axial position, for masked-mean pooling. The
        # scattered feature tensor is zero on invalid cells, so a plain sum
        # divided by these counts IS the masked mean.
        counts = (
            self.cell_valid.sum(dim=(1, 2)),   # (S,)
            self.cell_valid.sum(dim=(0, 2)),   # (C,)
            self.cell_valid.sum(dim=(0, 1)),   # (Fl,)
        )
        for ax in self.axes:
            self.register_buffer(f"pool_count{ax}", counts[ax].clamp_min(1.0))

    def _axial_kernel(self, dense_h, ax, gamma, q_proj, k_proj):
        """ONE shared kernel per (batch, timestep) for lattice axis ``ax``.

        Paper Eq. 4 (projection: masked mean over the other axes + gamma MLP)
        followed by Eq. 5 (Q/K from the projected 1-D function only).
        """
        count = getattr(self, f"pool_count{ax}")
        U = dense_h.sum(dim=self._POOL_DIMS[ax]) / count.view(1, 1, -1, 1)
        U = gamma(U)                                          # (B, L, A, H)
        Q = q_proj(U)
        K = k_proj(U)
        scores = torch.einsum("blqh,blkh->blqk", Q, K) * self.scale
        attn = F.softmax(scores, dim=-1)                      # (B, L, A, A)
        return self.dropout(attn)

    # -- forward ------------------------------------------------------------
    def forward(self, x):                                     # (B, L, G, F)
        B, L, G, _ = x.shape
        # in_proj + promoted-axis identity embeddings (see AxialComboSA) +
        # scatter. Identity enters BOTH the pooled kernels and V.
        dense_h = self._dense_features(x)                     # (B, L, S, C, Fl, H)

        out = self.v_proj(dense_h)          # shared V, once; zero on invalid cells
        for ax, gamma, q_proj, k_proj in zip(
                self.axes, self.gammas, self.q_projs, self.k_projs):
            attn = self._axial_kernel(dense_h, ax, gamma, q_proj, k_proj)
            eq_out, eq_den = self._CONTRACT[ax]
            # sequential contraction by shared per-axis kernels == (⊗_ax A_ax) . V
            num = torch.einsum(eq_out, attn, out)
            # softmax mass that landed on VALID keys, per line: renormalise so
            # every output is a convex combination of valid values only. Dead
            # lines (zero valid keys) have num == 0 and clamp keeps them finite;
            # vmask6 zeroes them at the end.
            den = torch.einsum(eq_den, attn, self.cell_valid)
            # clamp_min IS the guard -- it is what keeps dead lines (num == 0,
            # den == 0) finite, exactly as the comment above says. A following
            # nan_to_num therefore only re-scrubbed a tensor that is already
            # NaN-free, at the price of a full pass per axis per layer, and it
            # laundered real divergence into finite numbers. Divergence is
            # caught at the loss now -- see Trainer.NonFiniteLossError.
            out = num / den.clamp_min(1e-6).unsqueeze(-1)
            # Re-zero invalid cells before the NEXT axis. A contraction leaves
            # them holding a weighted average of valid cells -- legitimate
            # scratch, but nonzero. Carried into the next axis, that scratch
            # enters the numerator while ``den`` still counts only originally
            # valid keys, so the output stops being a convex combination and
            # inflates. Only matters for dims>=3 (two or more contractions).
            out = out * self.vmask6

        out = self.out_norm(out) * self.vmask6                # zero non-kept cells
        out = out.view(B, L, self.N, -1)
        return out[:, :, self.flat_idx, :]                    # (B, L, G, H)


# Legacy name kept so old manifests and imports resolve.
FactorizedAxialCA = LocalFactorizedAttention
