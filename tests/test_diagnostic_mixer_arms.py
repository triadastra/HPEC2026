"""The two diagnostic mixer arms: Exp 7 (``fa_local_*``) and Exp 8 (``fa_sm_*``).

The arm only answers anything if it differs from ``fa_*`` in exactly one way.
Every test here pins one way that could stop being true silently:

* the flag could be swallowed by ``**kwargs`` (the F4 failure mode), leaving a
  byte-identical duplicate of ``fa_*`` under a different run name;
* the geometry could drift apart from ``fa_*.yaml``, making the pair a
  two-variable comparison that still looks single-variable;
* the outer LeakyReLU or the quadrature divisor could survive the switch, so
  the arm would measure a ~1/line_count rescale instead of the nonlinearity;
* the run names could stop parsing, or stop inheriting the mech rate.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models import create_model
from src.models.fa import FactorizedAttention, _VENDOR_ROOT
from src.utils import compose_config, model_kwargs_from_config

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

GEOMETRY = dict(fa_heads=4, fa_dim_head=4, fa_kernel_multiplier=2,
                fa_qk_norm=True)


def model_kwargs(**over):
    kw = dict(
        num_numeric_features=FIN, num_states=S, num_commodities=C,
        num_flows=FL, hidden_size=H, num_layers=1, num_heads=4, d_ff=32,
        dropout=0.0, num_combos=len(COORDS), features_per_group=FIN,
        combo_coords=COORDS, lattice_dims=(S, C, FL),
        combo_encoder="embeddings", state_embed_dim=2, comm_embed_dim=3,
        flow_embed_dim=2, **GEOMETRY,
    )
    kw.update(over)
    return kw


def attention(dims, softmax):
    torch.manual_seed(0)
    return FactorizedAttention(FIN, H, (S, C, FL), COORDS, dims=dims,
                               fa_kernel_softmax=softmax, **GEOMETRY)


# --- the switch actually reaches the authors' kernel -----------------------

def test_default_stays_the_authors_leakyrelu_path():
    """fa_* must be untouched: the arm is additive, not a change of default."""
    module = attention(2, False)
    assert module.kernel_softmax is False
    assert all(k.softmax is False for k in module.kernels)


def test_fa_sm_turns_on_the_upstream_softmax_switch():
    module = attention(2, True)
    assert module.kernel_softmax is True
    assert all(k.softmax is True for k in module.kernels)


@pytest.mark.parametrize("model", ("gru", "lstm", "transformer"))
def test_variant_name_forces_the_flag_even_when_the_config_omits_it(model):
    """The variant name is the contract.

    ``**kwargs`` swallowed the nested YAML blocks once already (RETRAIN.md F4).
    If it swallowed this flag the arm would train a second copy of ``fa_*``
    under a name claiming otherwise, and nothing downstream would notice.
    """
    built = create_model(model, "fa_sm_2d", **model_kwargs())
    assert isinstance(built.axial, FactorizedAttention)
    assert built.axial.kernel_softmax is True
    plain = create_model(model, "fa_2d", **model_kwargs())
    assert plain.axial.kernel_softmax is False


def test_an_explicit_false_in_config_cannot_disarm_the_arm():
    built = create_model("gru", "fa_sm_2d",
                         **model_kwargs(fa_kernel_softmax=False))
    assert built.axial.kernel_softmax is True


def test_each_arm_carries_the_scaling_upstream_pairs_with_its_mode():
    """Upstream FABlockS2 does not hold scaling fixed across the switch.

    It tempers the kernel by 1/sqrt(dim_head * kernel_multiplier) when softmax
    is on (or kernel_multiplier > 4) and uses scaling_factor otherwise. Under
    softmax that scaling IS the temperature, so pinning it at the LeakyReLU
    arm's 1.0 would run this arm at 8x the authors' sharpness here and describe
    neither released path.
    """
    gated, softmaxed = attention(2, False), attention(2, True)
    assert all(k.scaling == 1.0 for k in gated.kernels)
    expected = (GEOMETRY["fa_dim_head"] * GEOMETRY["fa_kernel_multiplier"]) ** -0.5
    assert all(k.scaling == pytest.approx(expected) for k in softmaxed.kernels)


def test_neither_arm_keeps_the_parent_axial_attention():
    """FA replaces AxialComboSA's per-axis stacks; it is not layered on them.

    combo_attention.py empties ``self.sa`` for exactly this reason -- a second
    live reference would re-register the parent's projections and inflate every
    params/FLOPs figure in the cost panel while contributing nothing.
    """
    from src.models.combo_attention import GroupSelfAttention

    for softmax in (False, True):
        module = attention(3, softmax)
        assert len(module.sa) == 0
        assert not any(isinstance(m, GroupSelfAttention)
                       for m in module.modules())


# --- the nonlinearity is the thing that changed ---------------------------

@pytest.mark.parametrize("dims", (2, 3, 4))
def test_kernels_are_row_stochastic_only_on_the_softmax_arm(dims):
    """Directly inspect the kernel both paths hand to the contraction."""
    x = torch.randn(2, 3, len(COORDS), FIN)
    kernels = {}
    for softmax in (False, True):
        module = attention(dims, softmax).eval()
        dense = module._dense_features(x)
        mixed = module.channel_mixer(dense)
        _, kernel_input, _ = torch.split(
            mixed, [module.heads * module.dim_head, module.hidden_dim,
                    module.hidden_dim], dim=-1)
        ax, reducer, kernel = (module.axes[0], module.reducers[0],
                               module.kernels[0])
        with torch.no_grad():
            kernels[softmax] = module._kernel(kernel_input, ax, reducer, kernel)

    gated, normalised = kernels[False], kernels[True]
    # LeakyReLU keeps a negative tail; softmax cannot produce one.
    assert (gated < 0).any()
    assert (normalised >= 0).all()
    rows = normalised.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)


@pytest.mark.parametrize("dims", (2, 3, 4))
def test_forward_backward_is_finite_on_the_sparse_lattice(dims):
    module = attention(dims, True)
    x = torch.randn(2, 3, len(COORDS), FIN, requires_grad=True)
    y = module(x)
    assert y.shape == (2, 3, len(COORDS), H)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None
               for p in module.parameters() if p.requires_grad)


def test_softmax_arm_is_not_a_rescale_of_the_gated_arm():
    """Guards the divisor swap.

    Keeping ``/ line_count`` under a row-normalised kernel would leave the two
    arms differing by a per-line scalar. Then the experiment would report the
    rescale, not the nonlinearity, and the difference would still look real.
    """
    x = torch.randn(2, 3, len(COORDS), FIN)
    outs = {}
    for softmax in (False, True):
        with torch.no_grad():
            outs[softmax] = attention(3, softmax).eval()(x)
    a, b = outs[False].flatten(), outs[True].flatten()
    ratio = b / a.masked_fill(a.abs() < 1e-6, float("nan"))
    finite = ratio[torch.isfinite(ratio)]
    assert finite.numel() > 0
    assert finite.std() > 1e-3, "outputs differ by a near-constant factor"


# --- the comparison stays single-variable ---------------------------------

@pytest.mark.parametrize("dims", (2, 3, 4))
def test_yaml_geometry_is_identical_to_its_fa_comparator(dims):
    fa = model_kwargs_from_config(compose_config("gru", f"fa_{dims}d"))
    sm = model_kwargs_from_config(compose_config("gru", f"fa_sm_{dims}d"))
    assert sm.pop("fa_kernel_softmax") is True
    assert fa.pop("fa_kernel_softmax", False) is False
    assert fa.pop("variant", None) == fa.pop("variant", None)
    sm.pop("variant", None)
    fa.pop("variant", None)
    assert sm == fa


# --- the launch path -------------------------------------------------------

def test_train_cli_accepts_every_declared_grid_variant():
    """The arm has to survive argparse before it can survive anything else.

    ``--variant`` used to carry its own transcription of the grid-variant list,
    so adding fa_sm to the model layer alone left every Exp 7 run dying at
    `invalid choice: 'fa_sm_2d'`.
    """
    from src.models import COMBO_GRID_VARIANTS
    from scripts.train import parse_args

    sys_argv = sys.argv
    for variant in ("fa_sm_2d", "fa_sm_3d", "fa_sm_4d"):
        assert variant in COMBO_GRID_VARIANTS
        try:
            sys.argv = ["train.py", "--model", "gru", "--variant", variant]
            args = parse_args()
        finally:
            sys.argv = sys_argv
        assert args.variant == variant


def test_fa_sm_takes_the_combo_path_not_the_flat_one():
    """The second transcription decided combo-vs-flat.

    Missing there, an Exp 7 run would have loaded the per-series dataset and
    trained a flat model under an N-D name -- silently, and with a plausible
    number at the end of it.
    """
    from src.models import COMBO_GRID_VARIANTS

    source = (REPO / "scripts" / "train.py").read_text()
    assert "_axial = COMBO_GRID_VARIANTS" in source
    for host in ("gru", "lstm", "transformer"):
        for d in (2, 3, 4):
            assert f"fa_sm_{d}d" in COMBO_GRID_VARIANTS, host


def test_no_module_re_transcribes_the_grid_variant_list():
    """Three copies existed; two were load-bearing and nothing compared them."""
    from src.models import COMBO_GRID_STEMS

    stems = '"cross_attention", "cafa", "authors_cafa"'
    for path in ("scripts/train.py", "scripts/evaluate.py", "scripts/sweep.py"):
        text = (REPO / path).read_text()
        assert stems not in text, f"{path} re-transcribes the stem list"
    assert "fa_sm" in COMBO_GRID_STEMS


# --- the sweep wiring ------------------------------------------------------

@pytest.mark.parametrize("tid,mixer", [("7", "fa_local"), ("8", "fa_sm")])
def test_each_arm_is_the_paired_mirror_of_the_fa_cells(tid, mixer):
    from scripts.sweep import AXIAL_MODELS, DIMS, build_matrix

    runs = build_matrix({tid}, [947, 732, 619], 64, 4)
    names = [n for n, _ in runs]
    assert len(names) == len(AXIAL_MODELS) * len(DIMS) * 3 == 27
    assert len(set(names)) == len(names)
    for m in AXIAL_MODELS:
        for d in DIMS:
            assert f"{m}_{mixer}_{d}d_embeddings_s947" in names


@pytest.mark.parametrize("tid", ["7", "8"])
def test_arm_refuses_to_share_a_runs_dir_with_the_declared_matrix(tid):
    from scripts.sweep import build_matrix

    with pytest.raises(SystemExit) as excinfo:
        build_matrix({tid, "3"}, [947], 64, 4)
    assert "on its own" in str(excinfo.value)


def test_the_two_diagnostic_arms_refuse_to_share_a_session():
    """Exp 8 only means something once Exp 7 has shown a difference exists."""
    from scripts.sweep import build_matrix

    with pytest.raises(SystemExit):
        build_matrix({"7", "8"}, [947], 64, 4)


def test_every_fa_family_gets_its_own_rate_arm():
    """The prefix test this replaced sent all three families to ``mech``.

    ``mech`` was probed on the AUTHORS' LeakyReLU ``fa_3d``. Routing fa_local
    or fa_sm there would compare a tuned arm against one running on someone
    else's rate -- the asymmetry F5 removed -- and the arms are far enough
    apart for it to matter (nd vs mech is 3e-4 vs 1e-2 on gru).
    """
    from scripts.sweep import EXP0_ARMS, nd_rate_arm

    assert nd_rate_arm("asa_3d") == "nd"
    assert nd_rate_arm("grid_2d") == "nd"
    assert nd_rate_arm("fa_3d") == "mech"
    assert nd_rate_arm("fa_local_3d") == "mechlocal"
    assert nd_rate_arm("fa_sm_3d") == "mechsm"
    for arm in ("mech", "mechlocal", "mechsm"):
        assert arm in EXP0_ARMS


@pytest.mark.parametrize("tid,arm,mixer", [("7", "mechlocal", "fa_local"),
                                           ("8", "mechsm", "fa_sm")])
def test_cells_train_at_their_own_arms_rate_not_at_mech(tid, arm, mixer):
    from scripts.sweep import build_matrix

    own = {"gru": "1e-3", "lstm": "3e-4", "transformer": "1e-3"}
    selection = {"selected": {
        arm: own,
        "mech": {m: "9e-9" for m in own},      # the authors' fa rate
        "nd": {m: "8e-8" for m in own},
    }}
    runs = build_matrix({tid}, [947], 64, 4, lr_selection=selection)
    assert runs
    for name, argv in runs:
        model = name.split("_", 1)[0]
        assert argv[argv.index("--lr") + 1] == own[model], name


def test_exp0_probes_each_diagnostic_mixer_on_its_own_variant():
    """A rate probed on a different operator is a rate for a different problem."""
    from scripts.sweep import EXP0_LR_GRID, build_exp0_matrix

    runs = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    for arm, variant in (("mechlocal", "fa_local_3d"), ("mechsm", "fa_sm_3d")):
        cells = [(n, a) for n, a in runs if n.startswith(f"exp0_{arm}_")]
        assert len(cells) == 3 * len(EXP0_LR_GRID), arm
        for _, argv in cells:
            assert argv[argv.index("--variant") + 1] == variant
        rates = {a[a.index("--lr") + 1] for _, a in cells}
        assert rates == set(EXP0_LR_GRID), arm


def test_run_names_parse_back_into_their_build_variant():
    from scripts.evaluate import parse_run

    cfg = parse_run("gru_fa_sm_4d_embeddings_s619")
    assert cfg["model"] == "gru"
    assert cfg["variant"] == "fa_sm_4d"
    assert cfg["build_variant"] == "fa_sm_4d"
    assert cfg["enc"] == "embeddings"
    assert cfg["seed"] == 619
    assert cfg["combo"] is True
