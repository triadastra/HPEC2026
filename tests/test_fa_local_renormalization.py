"""The convex-combination invariant CaFA's sparse renormalization claims.

    python tests/test_cafa_renormalization.py

``cafa.py`` states that after each axial contraction "the output is renormalised
per line by the softmax mass that fell on VALID keys, so each output stays a
convex combination of valid values". This checks that claim rather than
assuming it.

The test is a constant input: if every valid cell carries the same value v,
then any convex combination of valid values is exactly v. Anything else means
mass entered the numerator that the denominator did not count.

Before the fix this passed for fa_local_2d and failed for fa_local_3d / fa_local_4d, since
a single contraction has no second axis to carry stale values into.
"""
import sys

sys.path.insert(0, ".")

import torch

from src.models.fa_local import LocalFactorizedAttention

S, C, FL, H, F = 2, 3, 2, 8, 4
DIMS = {2: "fa_local_2d", 3: "fa_local_3d", 4: "fa_local_4d"}


def sparse_lattice():
    """A lattice with several combinations absent."""
    coords = [(s, c, f) for s in range(S) for c in range(C) for f in range(FL)]
    absent = {(0, 1, 0), (1, 2, 1), (0, 2, 0), (1, 0, 1)}
    return [c for c in coords if c not in absent]


def contract(m, x):
    """The forward pass up to (but not including) out_norm."""
    dense_h = m._dense_features(x)
    out = m.v_proj(dense_h)
    for ax, gamma, q_proj, k_proj in zip(m.axes, m.gammas, m.q_projs, m.k_projs):
        attn = m._axial_kernel(dense_h, ax, gamma, q_proj, k_proj)
        eq_out, eq_den = m._CONTRACT[ax]
        num = torch.einsum(eq_out, attn, out)
        den = torch.einsum(eq_den, attn, m.cell_valid)
        out = torch.nan_to_num(num / den.clamp_min(1e-6).unsqueeze(-1)) * m.vmask6
    return out


def check_convex_combination(dims):
    """A constant field must survive the contraction unchanged."""
    keep = sparse_lattice()
    torch.manual_seed(0)
    m = LocalFactorizedAttention(F, H, (S, C, FL), keep, dims=dims).eval()

    x = torch.ones(1, 1, len(keep), F)  # constant -> V constant on valid cells
    with torch.no_grad():
        v = m.v_proj(m._dense_features(x)).reshape(-1, H)
        valid = m.cell_valid.reshape(-1).bool()
        const = v[valid][0]
        assert torch.allclose(v[valid], const.expand_as(v[valid]), atol=1e-6), (
            "test precondition failed: V is not constant across valid cells"
        )
        got = contract(m, x).reshape(-1, H)[valid]

    err = (got - const).abs().max().item()
    assert err < 1e-5, (
        f"{DIMS[dims]}: output is not a convex combination of valid values "
        f"(max deviation {err:.4f}). Absent cells must be re-zeroed between "
        f"contractions, or their scratch values enter the next numerator while "
        f"the denominator still counts only originally valid keys."
    )
    print(f"{DIMS[dims]}: convex combination holds (max deviation {err:.2e})")


def check_absent_cells_are_zero(dims):
    keep = sparse_lattice()
    torch.manual_seed(0)
    m = LocalFactorizedAttention(F, H, (S, C, FL), keep, dims=dims).eval()
    with torch.no_grad():
        out = contract(m, torch.randn(2, 3, len(keep), F))
    # vmask6 broadcasts over batch and time; index with it rather than
    # flattening, which would fold those dims into the cell axis.
    absent = out.masked_select(~m.vmask6.expand_as(out).bool())
    assert absent.abs().max().item() == 0.0, f"{DIMS[dims]}: absent cells are not zero"
    print(f"{DIMS[dims]}: absent cells stay exactly zero")


def check_dense_lattice_is_unaffected(dims):
    """With every cell present the renormalizer is 1 and the fix is a no-op --
    so this cannot change any dense-grid result."""
    dense = [(s, c, f) for s in range(S) for c in range(C) for f in range(FL)]
    torch.manual_seed(0)
    m = LocalFactorizedAttention(F, H, (S, C, FL), dense, dims=dims).eval()
    x = torch.ones(1, 1, len(dense), F)
    with torch.no_grad():
        const = m.v_proj(m._dense_features(x)).reshape(-1, H)[0]
        got = contract(m, x).reshape(-1, H)
    assert torch.allclose(got, const.expand_as(got), atol=1e-5)
    print(f"{DIMS[dims]}: dense lattice unchanged")


def main():
    for dims in (2, 3, 4):
        check_convex_combination(dims)
        check_absent_cells_are_zero(dims)
        check_dense_lattice_is_unaffected(dims)
    print("\nCaFA renormalization tests passed.")


def test_convex_combination_2d():
    check_convex_combination(2)


def test_convex_combination_3d():
    check_convex_combination(3)


def test_convex_combination_4d():
    check_convex_combination(4)


def test_absent_cells_zero():
    for d in (2, 3, 4):
        check_absent_cells_are_zero(d)


def test_dense_unaffected():
    for d in (2, 3, 4):
        check_dense_lattice_is_unaffected(d)


if __name__ == "__main__":
    main()
