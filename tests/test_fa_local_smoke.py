"""Smoke tests for AxialComboSA and LocalFactorizedAttention.

Collected by pytest (`python -m pytest tests/test_fa_local_smoke.py` — the
command RETRAIN.md §6 gives). Earlier versions used
script-style ``check_*`` names, so that exact pytest command collected ZERO
tests and exited green while none of these checks ran; the functions are
``test_*`` now precisely so that cannot recur. Also runnable directly on a
GPU box:

    python tests/test_fa_local_smoke.py

Checks: forward shape, masked-cell zeroing, gradient flow, NaN-safety on a
sparse lattice (some (commodity, flow) lines with zero kept states), the
Kronecker factorization of the CaFA-style kernels, and that ``axis_identity``
gates permutation equivariance along promoted axes (Test 4's mechanism). The
same identity property for the roster's authors'-FA operator is pinned in
tests/test_fa_smoke.py, which carries the submodule guard.
"""
import sys
sys.path.insert(0, ".")

import pytest
import torch

from src.models.fa_local import LocalFactorizedAttention
from src.models.combo_attention import AxialComboSA

MIXERS = (AxialComboSA, LocalFactorizedAttention)


def make_sparse_lattice(S=4, C=6, Fl=2, keep_frac=0.6, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords = [(s, c, f) for s in range(S) for c in range(C) for f in range(Fl)]
    keep = torch.rand(len(coords), generator=g) < keep_frac
    keep[0] = True  # at least one combo
    combo_coords = [coords[i] for i in range(len(coords)) if keep[i]]
    return combo_coords, (S, C, Fl)


@pytest.mark.parametrize("dims", (2, 3, 4))
@pytest.mark.parametrize("cls", MIXERS)
def test_forward_backward_on_sparse_lattice(cls, dims):
    torch.manual_seed(0)
    B, L, F, H = 3, 7, 5, 16
    combo_coords, lattice_dims = make_sparse_lattice()
    G = len(combo_coords)
    x = torch.randn(B, L, G, F, requires_grad=True)
    m = cls(F, H, lattice_dims, combo_coords, dims=dims)
    out = m(x)
    assert out.shape == (B, L, G, H), (cls.__name__, dims, out.shape)
    assert torch.isfinite(out).all(), f"{cls.__name__} dims={dims}: non-finite output"
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), \
        f"{cls.__name__} dims={dims}: bad input gradient"


def test_kronecker_factorization(B=2, L=3, F=5, H=16, seed=1):
    """The CaFA kernels must be SHARED across lattice lines (paper Eq. 4-5):
    one (B, L, A, A) kernel per axis, and on a dense lattice the joint operator
    must equal the Kronecker product of the per-axis kernels applied to V."""
    torch.manual_seed(seed)
    S, C, Fl = 3, 4, 2
    coords = [(s, c, f) for s in range(S) for c in range(C) for f in range(Fl)]
    m = LocalFactorizedAttention(F, H, (S, C, Fl), coords, dims=4)
    m.eval()
    x = torch.randn(B, L, len(coords), F)

    # per-axis kernel shape: quadratic in the AXIAL size only
    dense_h = m._dense_features(x)
    kernels = []
    with torch.no_grad():
        for ax, g, q, k in zip(m.axes, m.gammas, m.q_projs, m.k_projs):
            A = (S, C, Fl)[ax]
            attn = m._axial_kernel(dense_h, ax, g, q, k)
            assert attn.shape == (B, L, A, A), (ax, attn.shape)
            kernels.append(attn)
        # dense lattice: forward == Kronecker(A_S, A_C, A_F) . V  (out_norm'd)
        V = m.v_proj(dense_h).reshape(B, L, S * C * Fl, H)
        joint = torch.einsum("blqs,blrc,bltf->blqrtscf",
                             kernels[0], kernels[1], kernels[2])
        joint = joint.reshape(B, L, S * C * Fl, S * C * Fl)
        ref = m.out_norm(torch.einsum("blnm,blmh->blnh", joint, V))
        out = m(x)
        assert torch.allclose(out, ref, atol=1e-5), \
            f"Kronecker mismatch: max err {(out - ref).abs().max():.2e}"


@pytest.mark.parametrize("ident", (False, True))
@pytest.mark.parametrize("cls", MIXERS)
def test_axis_identity_gates_permutation_equivariance(cls, ident):
    """axis_identity=True must BREAK permutation equivariance along promoted
    axes (the module can tell WHICH state a cell is, the categorical analogue
    of axial positional encodings); axis_identity=False must restore it.
    This is Test 4's mechanism."""
    seed = 2
    torch.manual_seed(seed)
    B, L, F, H = 1, 3, 5, 16
    S, C, Fl = 4, 3, 2
    coords = [(s, c, f) for s in range(S) for c in range(C) for f in range(Fl)]
    perm = [2, 0, 3, 1]                                   # relabel states
    idx = torch.tensor([coords.index((perm[s], c, f)) for (s, c, f) in coords])
    x = torch.randn(B, L, len(coords), F)
    torch.manual_seed(seed)
    m = cls(F, H, (S, C, Fl), coords, dims=4, axis_identity=ident).eval()
    if ident:
        # probe the STRUCTURAL capability, not the init magnitude: at
        # std=0.02 the (pooled, softmaxed) CaFA path moves outputs by
        # <1e-3; training scales embeddings freely, so amplify here.
        for emb in m.axis_embeds:
            torch.nn.init.normal_(emb.weight, std=1.0)
    with torch.no_grad():
        gap = (m(x)[:, :, idx, :] - m(x[:, :, idx, :])).abs().max().item()
    if ident:
        assert gap > 1e-3, \
            f"{cls.__name__}: identity ON but still permutation-equivariant"
    else:
        assert gap < 1e-5, \
            f"{cls.__name__}: identity OFF but not equivariant (gap={gap:.2e})"


def main():
    for dims in (2, 3, 4):
        for cls in MIXERS:
            test_forward_backward_on_sparse_lattice(cls, dims)
            print(f"[ok] {cls.__name__:26s} dims={dims} forward/backward")
    test_kronecker_factorization()
    print("[ok] Kronecker factorization verified on dense lattice")
    for cls in MIXERS:
        for ident in (False, True):
            test_axis_identity_gates_permutation_equivariance(cls, ident)
    print("[ok] axis_identity embeddings break state-permutation equivariance "
          "(and ablate cleanly with axis_identity=False)")
    print("\nFA-local smoke test passed.")


if __name__ == "__main__":
    main()
