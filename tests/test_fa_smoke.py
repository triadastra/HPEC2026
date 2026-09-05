"""Smoke, provenance, and compatibility tests for Factorized Attention."""

import inspect
from pathlib import Path

import pytest
import torch

from src.models import create_model
from src.models.fa import (
    FactorizedAttention,
    _VENDOR_ROOT,
    _load_upstream_modules,
)
from src.models.fa_local import LocalFactorizedAttention
from src.utils import compose_config, model_kwargs_from_config


# The FA adapter imports the authors' unchanged modules from a git submodule.
# Without it every test here fails with the same ImportError, which makes
# `pytest tests/` red on a fresh clone and hides real regressions in the noise.
# Skip instead — RETRAIN.md §6 tells the operator to run this file, so the
# reason has to be legible.
pytestmark = pytest.mark.skipif(
    not (_VENDOR_ROOT / "libs" / "factorization_module.py").exists(),
    reason=("authors' CaFA submodule not initialized; run "
            "`git submodule update --init external/cafa-authors`"),
)


S, C, FL, FIN, H = 2, 3, 2, 5, 16
COORDS = [
    (s, c, f)
    for s in range(S)
    for c in range(C)
    for f in range(FL)
    if (s, c, f) != (0, 1, 0)
]


def fa_kwargs():
    return dict(
        fa_heads=4,
        fa_dim_head=4,
        fa_kernel_multiplier=2,
        fa_qk_norm=True,
    )


def model_kwargs():
    return dict(
        num_numeric_features=FIN,
        num_states=S,
        num_commodities=C,
        num_flows=FL,
        hidden_size=H,
        num_layers=1,
        num_heads=4,
        d_ff=32,
        dropout=0.0,
        num_combos=len(COORDS),
        features_per_group=FIN,
        combo_coords=COORDS,
        lattice_dims=(S, C, FL),
        combo_encoder="embeddings",
        state_embed_dim=2,
        comm_embed_dim=3,
        flow_embed_dim=2,
        **fa_kwargs(),
    )


def test_imported_components_come_from_pinned_upstream_submodule():
    PoolingReducer, LowRankKernel, MLP, GroupNorm = _load_upstream_modules()
    root = _VENDOR_ROOT.resolve()
    for cls in (PoolingReducer, LowRankKernel, MLP, GroupNorm):
        source = Path(inspect.getsourcefile(cls)).resolve()
        assert root in source.parents, (cls, source)


@pytest.mark.parametrize("dims", (2, 3, 4))
def test_adapter_forward_backward_on_sparse_lattice(dims):
    torch.manual_seed(0)
    module = FactorizedAttention(
        FIN, H, (S, C, FL), COORDS, dims=dims, **fa_kwargs()
    )
    x = torch.randn(2, 3, len(COORDS), FIN, requires_grad=True)
    y = module(x)
    assert y.shape == (2, 3, len(COORDS), H)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in module.parameters() if p.requires_grad)


@pytest.mark.parametrize("model", ("gru", "lstm", "transformer"))
@pytest.mark.parametrize("dims", (2, 3, 4))
def test_host_factories_expose_distinct_fa_variants(model, dims):
    variant = f"fa_{dims}d"
    built = create_model(model, variant, **model_kwargs()).eval()
    assert isinstance(built.axial, FactorizedAttention)
    x = torch.randn(1, 3, len(COORDS), FIN)
    with torch.no_grad():
        out = built(x)
    assert out.shape == (1, len(COORDS), 2)


def test_existing_local_cafa_variant_is_unchanged_and_distinct():
    built = create_model("gru", "fa_local_2d", **model_kwargs())
    assert isinstance(built.axial, LocalFactorizedAttention)
    assert not isinstance(built.axial, FactorizedAttention)


def test_old_authors_cafa_variant_remains_a_compatible_alias():
    built = create_model("gru", "authors_cafa_2d", **model_kwargs())
    assert isinstance(built.axial, FactorizedAttention)


def test_fa_yaml_uses_geometry_free_parameter_names():
    kwargs = model_kwargs_from_config(compose_config("gru", "fa_2d"))
    assert kwargs["fa_heads"] == 4
    assert kwargs["fa_dim_head"] == 32
    assert kwargs["fa_kernel_multiplier"] == 2
    assert kwargs["fa_qk_norm"] is True


# --------------------------------------------------------------------------
# The masked-mean assumption
# --------------------------------------------------------------------------
#
# fa.py pre-scales its input by total/count so that upstream's PoolingReducer,
# which pools an ordinary (dense) mean, produces a MASKED mean on this sparse
# lattice. That identity holds only if the reducer means BEFORE any bias or
# nonlinearity -- i.e. it is MLP(mean(x)), not mean(MLP(x)). If upstream ever
# reorders those, the comment stays true-looking and every FA run silently
# pools absent lattice cells as real zeros.
#
# The module cannot be read on a machine without the submodule, so this asserts
# the property instead of the implementation.

def _fa_module(dims=3):
    return FactorizedAttention(
        features_per_group=FIN, hidden_dim=H,
        lattice_dims=(S, C, FL), combo_coords=COORDS, dims=dims,
    )


@pytest.mark.parametrize("ax", [0, 1, 2])
def test_pooling_reducer_sees_a_masked_mean_not_a_zero_padded_one(ax):
    """Feeding the scaled sparse tensor must equal feeding a tensor that is
    already the masked mean everywhere -- which is what "the pre-scaling makes
    its mean a masked mean" actually claims."""
    torch.manual_seed(0)
    fa = _fa_module()
    if ax not in fa.axes:
        pytest.skip(f"axis {ax} not attended at dims={len(fa.axes)}")
    reducer = fa.reducers[fa.axes.index(ax)]

    B, L = 2, 3
    u = torch.randn(B, L, S, C, FL, H)
    u = u * fa.vmask6                      # zero on absent cells, as in forward

    got = fa._project_axis(u, ax, reducer)

    # The masked mean over the two non-attended axes, broadcast back so an
    # ordinary dense mean reproduces it exactly.
    others = [i for i in range(3) if i != ax]
    counts = fa.cell_valid.sum(dim=tuple(others)).clamp_min(1.0)
    shape = [1, 1, 1, 1, 1, 1]
    shape[2 + ax] = -1
    masked_mean = u.sum(dim=[2 + o for o in others], keepdim=True) / \
        counts.reshape(shape)
    dense = masked_mean.expand_as(u)

    lattice = (S, C, FL)
    perm = [0, 1, 2 + ax, 2 + others[0], 2 + others[1], 5]
    x = dense.permute(*perm).contiguous().view(
        B * L, lattice[ax], lattice[others[0]], lattice[others[1]], H)
    expected = reducer(x).view(B, L, lattice[ax], H)

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_absent_cells_do_not_drag_the_projection_toward_zero():
    """The failure mode if the identity breaks: absent cells pool as real
    zeros, so an axial position with few valid cells is biased toward 0."""
    torch.manual_seed(0)
    fa = _fa_module()
    ax = fa.axes[0]
    reducer = fa.reducers[0]

    # Constant over every VALID cell: a masked mean must return that constant's
    # image, identically for every axial position, however many cells are
    # absent. A zero-padded mean would differ where cells are missing.
    u = torch.ones(1, 1, S, C, FL, H) * fa.vmask6
    projected = fa._project_axis(u, ax, reducer)
    first = projected[:, :, :1, :]
    torch.testing.assert_close(
        projected, first.expand_as(projected), rtol=1e-5, atol=1e-5)


def test_channel_mixer_maps_absent_cells_to_exact_zero():
    """The other half of the masked-mean identity. The forward pass feeds
    ``channel_mixer(dense)`` with absent cells at exactly zero and pools the
    resulting kernel_input WITHOUT re-masking it, so the pre-scaled pooling
    only equals a masked mean if ``channel_mixer(0) == 0`` — i.e. upstream's
    MLP is bias-free (``no_bias=True``) and its GELU maps 0 to 0. If upstream
    ever gained biases, every absent lattice cell would enter the axial
    pooling as a real value and no other test here would notice."""
    torch.manual_seed(0)
    fa = _fa_module()
    z = torch.zeros(2, 3, S, C, FL, fa.hidden_dim)
    out = fa.channel_mixer(z)
    assert (out == 0).all(), (
        "channel_mixer(0) != 0: the upstream MLP is no longer bias-free, so "
        "absent cells contaminate the FA pooling"
    )


def test_axis_identity_gates_permutation_equivariance_for_fa():
    """Test 4 runs 54 of its 108 cells on THIS operator, so the identity
    mechanism has to be pinned here too, not only on AxialComboSA/fa_local
    (tests/test_fa_local_smoke.py): axis_identity=True must break permutation
    equivariance along promoted axes, axis_identity=False must restore it."""
    seed = 2
    S2, C2, FL2 = 4, 3, 2
    coords = [(s, c, f) for s in range(S2) for c in range(C2) for f in range(FL2)]
    perm = [2, 0, 3, 1]                                   # relabel states
    idx = torch.tensor([coords.index((perm[s], c, f)) for (s, c, f) in coords])
    torch.manual_seed(seed)
    x = torch.randn(1, 3, len(coords), FIN)
    for ident in (False, True):
        torch.manual_seed(seed)
        m = FactorizedAttention(
            FIN, H, (S2, C2, FL2), coords, dims=4,
            axis_identity=ident, **fa_kwargs()
        ).eval()
        if ident:
            # Probe the structural capability, not the init magnitude (cf. the
            # same amplification in test_fa_local_smoke.py).
            for emb in m.axis_embeds:
                torch.nn.init.normal_(emb.weight, std=1.0)
        with torch.no_grad():
            gap = (m(x)[:, :, idx, :] - m(x[:, :, idx, :])).abs().max().item()
        if ident:
            assert gap > 1e-3, \
                "FactorizedAttention: identity ON but still permutation-equivariant"
        else:
            assert gap < 1e-5, \
                f"FactorizedAttention: identity OFF but not equivariant (gap={gap:.2e})"
