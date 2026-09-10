"""Geometry-free Factorized Attention (FA) for categorical tensor lattices.

The unmodified upstream repository lives at ``external/cafa-authors`` as a git
submodule pinned to BaratiLab/CaFA.  Its released ``FABlockS2`` is specifically
a two-axis latitude/longitude operator, whereas this benchmark's historical
``2d/3d/4d`` labels promote one/two/three categorical axes.  Consequently the
upstream block cannot be used verbatim for the complete experiment grid.

``FactorizedAttention`` isolates the important, domain-independent FA operator
from CaFA; it is not a claim that this benchmark is the authors' weather model.
It imports and uses
their actual, unchanged implementations of:

* ``PoolingReducer`` for global axial projection;
* ``LowRankKernel`` for multi-head axial Q/K kernels;
* ``MLP`` and ``GroupNorm`` for the channel mixer and head merge.

It also preserves the authors' default non-softmax kernel path with LeakyReLU
gating and Q/K RMS normalization.  Only domain glue is local: scattering the
sparse State x Commodity x Flow lattice, applying uniform categorical mesh
weights, supporting one to three promoted axes, and gathering kept cells.

The existing ``src.models.cafa.FactorizedAxialCA`` remains available under the
``cafa_*`` variants.  This implementation is selected by ``fa_*``; the former
``authors_cafa_*`` names remain compatibility aliases for old run manifests.
"""

from functools import lru_cache
import importlib
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .combo_attention import AxialComboSA


_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "external" / "cafa-authors"


@lru_cache(maxsize=1)
def _load_upstream_modules():
    """Load the authors' unchanged modules without making them a dependency of
    ordinary model imports.  Their repository uses top-level ``libs.*`` imports,
    so its root must briefly be on ``sys.path``.
    """
    expected = _VENDOR_ROOT / "libs" / "factorization_module.py"
    if not expected.exists():
        raise ImportError(
            "Authors' CaFA submodule is missing. Run "
            "`git submodule update --init external/cafa-authors`."
        )

    existing = sys.modules.get("libs")
    if existing is not None:
        module_file = Path(getattr(existing, "__file__", "")).resolve()
        if _VENDOR_ROOT.resolve() not in module_file.parents:
            raise ImportError(
                "Cannot load authors' CaFA because an unrelated top-level "
                f"`libs` package is already loaded from {module_file}."
            )

    root = str(_VENDOR_ROOT)
    sys.path.insert(0, root)
    try:
        factor = importlib.import_module("libs.factorization_module")
        attention = importlib.import_module("libs.attention")
        basics = importlib.import_module("libs.basics")
    finally:
        try:
            sys.path.remove(root)
        except ValueError:
            pass

    loaded = Path(factor.__file__).resolve()
    if _VENDOR_ROOT.resolve() not in loaded.parents:
        raise ImportError(f"Loaded the wrong CaFA factorization module: {loaded}")
    return factor.PoolingReducer, attention.LowRankKernel, basics.MLP, basics.GroupNorm


class FactorizedAttention(AxialComboSA):
    """CaFA-style factorized attention for a sparse categorical lattice.

    Input/output match :class:`AxialComboSA`: ``(B,L,G,F) -> (B,L,G,H)``.
    ``dims=2/3/4`` selects State / +Commodity / +Flow, preserving the existing
    experiment labels and leftover-encoder contract.
    """

    _CONTRACT = {
        0: "blhqs,blhscfd->blhqcfd",
        1: "blhqc,blhscfd->blhsqfd",
        2: "blhqf,blhscfd->blhscqd",
    }
    # Softmax-kernel path only: the share of each row's softmax mass that landed
    # on VALID keys, laid out to broadcast over the head dim of _CONTRACT's
    # output. Dividing by it restores the convex combination that the mass
    # falling on masked (zero-value) cells would otherwise shrink.
    _VALID_MASS = {
        0: "blhqs,scf->blhqcf",
        1: "blhqc,scf->blhsqf",
        2: "blhqf,scf->blhscq",
    }

    def __init__(
        self,
        features_per_group,
        hidden_dim,
        lattice_dims,
        combo_coords,
        dims,
        attn_dropout: float = 0.0,
        axis_identity: bool = False,
        fa_heads: int = None,
        fa_dim_head: int = None,
        fa_kernel_multiplier: int = None,
        fa_qk_norm: bool = None,
        fa_kernel_softmax: bool = False,
        # Deprecated configuration aliases, retained so old manifests load.
        authors_cafa_heads: int = None,
        authors_cafa_dim_head: int = None,
        authors_cafa_kernel_multiplier: int = None,
        authors_cafa_qk_norm: bool = None,
    ):
        super().__init__(
            features_per_group,
            hidden_dim,
            lattice_dims,
            combo_coords,
            dims,
            attn_dropout=attn_dropout,
            axis_identity=axis_identity,
        )
        # The parent creates dense per-axis CrossAttention modules; this backend
        # replaces them with the authors' low-rank kernels.
        self.sa = nn.ModuleList()
        PoolingReducer, LowRankKernel, MLP, GroupNorm = _load_upstream_modules()

        heads = int(
            fa_heads if fa_heads is not None
            else authors_cafa_heads if authors_cafa_heads is not None
            else 4
        )
        if heads <= 0:
            raise ValueError("fa_heads must be positive")
        configured_dim_head = (
            fa_dim_head if fa_dim_head is not None else authors_cafa_dim_head
        )
        dim_head = (
            max(1, hidden_dim // heads)
            if configured_dim_head is None
            else int(configured_dim_head)
        )
        if dim_head <= 0:
            raise ValueError("fa_dim_head must be positive")
        configured_multiplier = (
            fa_kernel_multiplier
            if fa_kernel_multiplier is not None
            else authors_cafa_kernel_multiplier
        )
        kernel_multiplier = int(
            2 if configured_multiplier is None else configured_multiplier
        )
        if kernel_multiplier <= 0:
            raise ValueError("fa_kernel_multiplier must be positive")
        qk_norm = (
            fa_qk_norm if fa_qk_norm is not None
            else authors_cafa_qk_norm if authors_cafa_qk_norm is not None
            else True
        )

        # Kernel nonlinearity. False (default) = the authors' released path:
        # LowRankKernel(softmax=False) + LeakyReLU gating + uniform categorical
        # quadrature. True = their OWN softmax switch, which is what fa_local
        # deliberately used to hold the nonlinearity fixed against the axial
        # baseline. Flipping it also removes the outer LeakyReLU and replaces
        # the quadrature divisor with a valid-mass renormaliser -- see
        # ``_kernel`` and ``forward``; a softmax kernel is already normalised,
        # so keeping either would rescale the output by ~1/line_count and make
        # the arm measure the rescale rather than the nonlinearity.
        self.kernel_softmax = bool(fa_kernel_softmax)

        self.heads = heads
        self.dim_head = dim_head
        self.hidden_dim = int(hidden_dim)

        # This is the channel mixer used by upstream FABlockS2.  It produces a
        # multi-head value, the features used for axial projection, and a skip.
        self.channel_mixer = MLP(
            [hidden_dim, hidden_dim * 6, heads * dim_head + hidden_dim * 2],
            nn.GELU(),
        )
        self.reducers = nn.ModuleList(
            [PoolingReducer(hidden_dim, hidden_dim, hidden_dim) for _ in self.axes]
        )
        self.kernels = nn.ModuleList(
            [
                LowRankKernel(
                    hidden_dim,
                    dim_head * kernel_multiplier,
                    heads,
                    residual=False,
                    softmax=self.kernel_softmax,
                    # Upstream FABlockS2 does not hold scaling fixed across the
                    # switch: it uses scaling_factor (default 1.0, which is what
                    # the LeakyReLU arm runs at) unless kernel_multiplier > 4 OR
                    # softmax is on, in which case the kernel is tempered by
                    # 1/sqrt(dim_head * kernel_multiplier). Reproduce the whole
                    # condition. Under softmax the scaling IS the temperature,
                    # so keeping 1.0 would run the arm at 8x the authors' sharpness
                    # at this geometry (dim_head 32 x multiplier 2 -> 1/8) and the
                    # result would describe neither released path.
                    scaling=((dim_head * kernel_multiplier) ** -0.5
                             if kernel_multiplier > 4 or self.kernel_softmax
                             else 1.0),
                    dropout=attn_dropout,
                    qk_norm=bool(qk_norm),
                )
                for _ in self.axes
            ]
        )
        self.merge_head = nn.Sequential(
            GroupNorm(heads, dim_head * heads),
            nn.Linear(dim_head * heads, hidden_dim, bias=False),
        )
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Upstream operates on a dense grid.  Pre-scaling by total/count makes
        # its bias-free PoolingReducer's ordinary mean equal a masked mean on
        # this sparse lattice.  Per-line counts provide uniform categorical
        # quadrature during the value contractions.
        sizes = (self.S, self.C, self.Fl)
        for ax in self.axes:
            other = [i for i in range(3) if i != ax]
            axial_count = self.cell_valid.sum(dim=tuple(other)).clamp_min(1.0)
            total_other = sizes[other[0]] * sizes[other[1]]
            self.register_buffer(f"pool_scale{ax}", total_other / axial_count)

            line_count = self.cell_valid.sum(dim=ax).clamp_min(1.0)
            shape = [1, 1, 1, self.S, self.C, self.Fl, 1]
            shape[3 + ax] = 1
            self.register_buffer(f"line_count{ax}", line_count.reshape(shape))

    def _project_axis(self, u, ax, reducer):
        """Run upstream PoolingReducer with the selected axis in slot 1."""
        B, L, S, C, Fl, H = u.shape
        lattice_sizes = (S, C, Fl)
        others = [i for i in range(3) if i != ax]
        perm = [0, 1, 2 + ax, 2 + others[0], 2 + others[1], 5]
        x = u.permute(*perm).contiguous()
        A = lattice_sizes[ax]
        x = x.view(B * L, A, lattice_sizes[others[0]], lattice_sizes[others[1]], H)
        scale = getattr(self, f"pool_scale{ax}").view(1, A, 1, 1, 1)
        return reducer(x * scale).view(B, L, A, H)

    def _kernel(self, u, ax, reducer, kernel):
        projected = self._project_axis(u, ax, reducer)
        B, L, A, H = projected.shape
        # Actual upstream LowRankKernel; categorical axes intentionally have no
        # angular distance modulation.  The authors' non-softmax LeakyReLU path
        # is retained exactly.
        k = kernel(projected.view(B * L, A, H))
        if not self.kernel_softmax:
            k = F.leaky_relu(k, negative_slope=0.2)
        return k.view(B, L, self.heads, A, A)

    def forward(self, x):
        B, L, _, _ = x.shape
        dense = self._dense_features(x)
        mixed = self.channel_mixer(dense)
        v_width = self.heads * self.dim_head
        value, kernel_input, skip = torch.split(
            mixed, [v_width, self.hidden_dim, self.hidden_dim], dim=-1
        )
        out = value.view(
            B, L, self.S, self.C, self.Fl, self.heads, self.dim_head
        ).permute(0, 1, 5, 2, 3, 4, 6).contiguous()
        mask7 = self.vmask6.unsqueeze(2)

        for ax, reducer, kernel in zip(self.axes, self.reducers, self.kernels):
            out = out * mask7
            axial_kernel = self._kernel(kernel_input, ax, reducer, kernel)
            den = (torch.einsum(self._VALID_MASS[ax], axial_kernel,
                                self.cell_valid.to(axial_kernel.dtype))
                   if self.kernel_softmax else None)
            out = torch.einsum(self._CONTRACT[ax], axial_kernel, out)
            if self.kernel_softmax:
                # The kernel is already row-normalised, so the quadrature
                # divisor below would rescale by ~1/line_count on top of it.
                # Renormalise by valid mass instead. clamp_min is the guard for
                # dead lines (no valid keys): there num == 0, so the quotient
                # stays finite and mask7 zeroes it on the next line anyway.
                out = out / den.clamp_min(1e-6).unsqueeze(-1)
            else:
                # line_count is registered with .clamp_min(1.0), so this divisor
                # is never zero and the quotient cannot be NaN by construction.
                # The nan_to_num that used to wrap it was therefore a no-op
                # costing a full pass per axis per layer, and it turned genuine
                # divergence into finite numbers. Divergence is caught at the
                # loss now -- see Trainer.NonFiniteLossError.
                out = out / getattr(self, f"line_count{ax}")
            out = out * mask7

        out = out.permute(0, 1, 3, 4, 5, 2, 6).reshape(
            B, L, self.S, self.C, self.Fl, v_width
        )
        out = self.merge_head(out)
        out = self.to_out(out + skip) * self.vmask6
        out = out.view(B, L, self.N, self.hidden_dim)
        return out[:, :, self.flat_idx, :]


__all__ = ["FactorizedAttention"]
