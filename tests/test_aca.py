"""Axial CROSS-attention (ACA): the property that distinguishes it, and the wiring.

The whole point of this arm is that Q and K come from DIFFERENT sources. If it
ever degrades into another self-attention it becomes a duplicate of `asa_*` that
costs 144 GPU runs, so the discriminating test is the important one here.
"""

import re
from pathlib import Path

import pytest
import torch

from scripts.sweep import ALL_MECHS, MECHS, build_matrix, DEFAULT_SEEDS, FLAT_MODELS
from src.models import create_model
from src.models.aca import AxialCrossAttention
from src.models.combo_attention import AxialComboSA

S, C, FL, FIN, H = 3, 4, 2, 5, 16
DENSE = [(s, c, f) for s in range(S) for c in range(C) for f in range(FL)]
SPARSE = [cell for cell in DENSE if cell not in {(0, 1, 0), (2, 3, 1)}]


def _host_kwargs(coords):
    return dict(num_numeric_features=FIN, num_states=S, num_commodities=C,
                num_flows=FL, num_combos=len(coords), features_per_group=FIN,
                hidden_size=H, num_layers=1, combo_coords=coords,
                lattice_dims=(S, C, FL), combo_encoder="embeddings",
                state_embed_dim=2, comm_embed_dim=2, flow_embed_dim=2)


# --------------------------------------------------------------------------
# The defining property
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dims", (2, 3, 4))
def test_aca_draws_from_outside_its_own_axis_line_and_asa_does_not(dims):
    """Self-attention along an axis can only mix within that axis' line.

    With dims=2 only State is promoted, so ``asa`` updating cell (0,0,0) can
    depend on (1,0,0) -- same (commodity, flow) column -- but NEVER on (1,1,1).
    ``aca`` queries the pooled COMPLEMENT of the axis, so it must.

    NB: the readout has to be a non-degenerate projection. Summing all H
    components of a LayerNorm output cancels by construction and would make
    every gradient here look like zero.
    """
    target = DENSE.index((0, 0, 0))
    off_line = DENSE.index((1, 1, 1))
    torch.manual_seed(1)
    readout = torch.randn(H)

    grads = {}
    for name, cls in (("asa", AxialComboSA), ("aca", AxialCrossAttention)):
        torch.manual_seed(0)
        module = cls(FIN, H, (S, C, FL), DENSE, dims=dims)
        x = torch.randn(1, 1, len(DENSE), FIN, requires_grad=True)
        (module(x)[0, 0, target] * readout).sum().backward()
        grads[name] = x.grad[0, 0, off_line].abs().sum().item()

    assert grads["aca"] > 1e-6, "ACA is not reaching off its own axis line"
    if dims == 2:
        # Only State is promoted, so a self-attention hop provably cannot see
        # another (commodity, flow) column. At dims>=3 more axes are promoted
        # and ASA reaches further, so the contrast is only exact here.
        assert grads["asa"] == 0.0


@pytest.mark.parametrize("dims", (2, 3, 4))
def test_forward_backward_finite_on_a_sparse_lattice(dims):
    torch.manual_seed(0)
    module = AxialCrossAttention(FIN, H, (S, C, FL), SPARSE, dims=dims)
    x = torch.randn(2, 3, len(SPARSE), FIN, requires_grad=True)
    out = module(x)
    assert out.shape == (2, 3, len(SPARSE), H)
    assert torch.isfinite(out).all()
    out.pow(2).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_absent_lattice_cells_never_leave_the_operator():
    """Only kept cells are gathered, and dead complement columns are masked
    out of the softmax rather than being handed attention mass."""
    torch.manual_seed(0)
    module = AxialCrossAttention(FIN, H, (S, C, FL), SPARSE, dims=4)
    assert module.flat_idx.numel() == len(SPARSE)
    for ax in module.axes:
        valid = getattr(module, f"comp_valid{ax}")
        counts = getattr(module, f"comp_count{ax}")
        assert valid.numel() == counts.numel()
        assert valid.any(), "every complement column masked out"


def test_backend_swap_leaves_no_parent_projections_registered():
    module = AxialCrossAttention(FIN, H, (S, C, FL), SPARSE, dims=3)
    groups = {name.split(".")[0] for name, _ in module.named_parameters()}
    assert "sa" not in groups and "ca" not in groups


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

@pytest.mark.parametrize("host", ("gru", "lstm", "transformer"))
@pytest.mark.parametrize("dims", (2, 3, 4))
def test_every_axial_host_builds_and_runs_aca(host, dims):
    model = create_model(host, f"aca_{dims}d", **_host_kwargs(SPARSE))
    assert isinstance(model.axial, AxialCrossAttention)
    with torch.no_grad():
        out = model(torch.randn(1, 3, len(SPARSE), FIN))
    assert out.shape == (1, len(SPARSE), 2)


@pytest.mark.parametrize("dims", (2, 3, 4))
def test_variant_config_exists_and_names_itself(dims):
    text = Path(f"config/variants/aca_{dims}d.yaml").read_text()
    assert f"variant: aca_{dims}d" in text


def test_only_one_grid_backend_may_be_selected():
    from src.models.gru import GRUComboAxial

    with pytest.raises(ValueError):
        GRUComboAxial(variant="aca_2d", dims=2, factorized=True, cross_axis=True,
                      **_host_kwargs(SPARSE))


# --------------------------------------------------------------------------
# Budget: ACA is opt-in, because it is 144 more GPU runs.
# --------------------------------------------------------------------------

def _matrix(mechs=None):
    return build_matrix({"1", "2", "3", "4", "6"}, DEFAULT_SEEDS, 2048, 1,
                        accum=14, effective_batch_size=28292, mechs=mechs)


def test_aca_is_absent_from_the_default_matrix():
    names = [name for name, _ in _matrix()]
    assert "aca" not in MECHS
    assert not any(re.search(r"_aca_\dd", name) for name in names)
    assert len(names) == 369   # two mixers since fa_local left the roster


def test_aca_joins_the_matrix_on_request_at_a_known_cost():
    base = len(_matrix())
    opted = [name for name, _ in _matrix(MECHS + ["aca"])]
    assert len([n for n in opted if re.search(r"_aca_\dd", n)]) == 144
    assert len(opted) == base + 144


def test_dropped_local_arm_is_off_roster_but_still_reachable():
    # fa_local left the declared matrix so every FA claim rests on the authors'
    # components, but the code and its sparse-renormalization fix stay, and a
    # reviewer can rerun the arm on request.
    assert "fa_local" not in MECHS
    assert "fa_local" in ALL_MECHS
    names = [name for name, _ in _matrix(MECHS + ["fa_local"])]
    assert len([n for n in names if re.search(r"_fa_local_\dd", n)]) == 144
