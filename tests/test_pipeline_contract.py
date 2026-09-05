"""Regressions for contracts that bind the scripts to each other.

Each test here corresponds to a defect where two entry points agreed by
coincidence rather than by construction, or where a real failure was reported
somewhere nobody reads.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate import _problem_summary
from scripts.significance_tests import dm_standard, paired_t, wilcoxon_series
from scripts.sweep import _log_reports_failure, train_target_months
from src.data.census_loader import CensusConfig, census_config_from_config
from src.training.trainer import Trainer
from src.utils import compose_data_config


# --------------------------------------------------------------------------
# Data contract: evaluation / cost / xgb must read config/census.yaml, not the
# CensusConfig dataclass defaults that happen to match it today.
# --------------------------------------------------------------------------

def test_composed_data_config_carries_the_census_data_block():
    cfg = compose_data_config()
    data = cfg.get("data", {})
    assert data.get("loader") == "census"
    for key in ("input_len", "lag_count", "lag_mode", "train_end", "val_end"):
        assert key in data, f"config/census.yaml no longer declares {key}"


def test_census_config_follows_the_config_not_the_dataclass_default():
    moved = {"data": {"input_len": 24, "lag_count": 6, "train_end": 120,
                      "val_end": 150, "lag_mode": "all"}}
    cfg = census_config_from_config(moved)
    assert (cfg.input_len, cfg.lag_count, cfg.train_end, cfg.val_end) == (24, 6, 120, 150)
    assert cfg.lag_mode == "all"
    default = CensusConfig()
    assert cfg.input_len != default.input_len, (
        "the fixture must differ from the default, or it proves nothing"
    )


def test_census_config_overrides_beat_the_config_block():
    cfg = census_config_from_config({"data": {"npz": "from/config.npz"}},
                                    npz="from/caller.npz", aggregate=True)
    assert cfg.npz == "from/caller.npz"
    assert cfg.aggregate is True


def test_census_config_ignores_non_dataclass_data_keys():
    # config/census.yaml carries loader/allow_legacy_artifact and friends; a new
    # key must not become a TypeError in every script at once.
    cfg = census_config_from_config({"data": {"loader": "census", "unknown_key": 1}})
    assert isinstance(cfg, CensusConfig)


def test_new_census_config_fields_reach_callers_without_a_hand_written_list():
    # The bug this replaces: each script kept its own tuple of field names, so
    # test_end/refit_normalization had to be added in several places by hand.
    cfg = census_config_from_config(
        {"data": {"test_end": 180, "refit_normalization": True, "aggregate": True}}
    )
    assert cfg.test_end == 180
    assert cfg.refit_normalization is True


# --------------------------------------------------------------------------
# Step-match auditing: the log line RETRAIN.md tells the operator to compare
# has to quote the same unit for both arms.
# --------------------------------------------------------------------------

class _FakeComboDataset:
    def __init__(self, groups):
        self.G = groups

    def __len__(self):
        return 4


class _FakeLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 4


def _trainer(batch_size, grad_accum=1, effective=None):
    trainer = Trainer({"training": {"batch_size": batch_size, "grad_accum": grad_accum,
                                    **({"effective_batch_size": effective}
                                       if effective else {})}})
    return trainer


def test_combo_effective_batch_is_quoted_in_observations():
    # One combo sample is a whole month carrying every group, so batch_size=1
    # is 18 observations, not 1. Quoting the raw sample count made a correctly
    # step-matched pair read as an 18x mismatch in the audit line.
    trainer = _trainer(batch_size=1)
    trainer.combo = True
    loader = _FakeLoader(_FakeComboDataset(18))
    assert trainer._effective_observations(loader) == 18


def test_flat_effective_batch_uses_the_pinned_exact_group():
    trainer = _trainer(batch_size=2048, effective=18)
    trainer.combo = False
    assert trainer._effective_observations(_FakeLoader(None)) == 18


def test_step_matched_arms_report_the_same_effective_batch():
    flat = _trainer(batch_size=2048, effective=18)
    flat.combo = False
    combo = _trainer(batch_size=1)
    combo.combo = True
    assert (flat._effective_observations(_FakeLoader(None))
            == combo._effective_observations(_FakeLoader(_FakeComboDataset(18))))


# --------------------------------------------------------------------------
# Sweep: the combo arm's samples/epoch follows the window and split, not a
# constant that goes stale when either moves.
# --------------------------------------------------------------------------

def test_train_target_months_follows_the_configured_window():
    assert train_target_months({"data": {"input_len": 6, "lag_count": 3,
                                         "train_end": 24}}) == 15
    assert train_target_months({"data": {"input_len": 36, "lag_count": 12,
                                         "train_end": 144}}) == 96


# --------------------------------------------------------------------------
# Significance suite: degenerate inputs must not be reported as results.
# --------------------------------------------------------------------------

def test_identical_loss_paths_are_not_infinitely_significant():
    same = np.ones((5, 8), dtype=np.float64)
    dm, p = dm_standard(same, same.copy())
    assert math.isnan(dm) and math.isnan(p)

    t, pt = paired_t(same.mean(1), same.mean(1))
    assert math.isnan(t) and math.isnan(pt)

    z, pw = wilcoxon_series(same.mean(1), same.mean(1))
    assert math.isnan(z) and math.isnan(pw)


def test_diebold_mariano_still_scores_a_real_difference():
    rng = np.random.default_rng(0)
    worse = rng.normal(1.0, 0.05, size=(6, 24))
    # A CONSTANT gap would have zero variance and correctly return NaN, so the
    # advantage has to vary month to month for the statistic to exist at all.
    better = worse - rng.normal(0.5, 0.05, size=worse.shape)
    dm, p = dm_standard(better, worse)
    assert dm < 0 and 0.0 <= p <= 1.0


def test_dm_rejects_a_single_month():
    with pytest.raises(ValueError):
        dm_standard(np.zeros((3, 1)), np.ones((3, 1)))


# --------------------------------------------------------------------------
# Evaluation reporting: refusals stay readable, and a dropped baseline is not
# silently absent from the leaderboard.
# --------------------------------------------------------------------------

def test_problem_summary_is_bounded_and_points_at_the_full_list():
    problems = [f"run_{i}: missing best.pth" for i in range(200)]
    text = _problem_summary("evaluation refused", problems, "outputs/failures.json")
    assert "200 problem(s)" in text
    assert "... 188 more" in text
    assert "outputs/failures.json" in text
    assert len(text.splitlines()) < 20


# --------------------------------------------------------------------------
# --status must not call a healthy run failed.
# --------------------------------------------------------------------------

_BENIGN_IMPORT_WARNING = (
    "src/models/__init__.py:62: UserWarning: Mamba-ND unavailable "
    "(ModuleNotFoundError(\"No module named 'triton'\")); "
    "'mamba_nd' model not registered.\n"
    "Final val loss: 0.030787\n"
)


def test_benign_import_warning_is_not_a_failure():
    # The optional-dependency notices embed ModuleNotFoundError in a warning,
    # so a bare `"Error" in text` marked every healthy log as failed.
    assert _log_reports_failure(_BENIGN_IMPORT_WARNING) is False


@pytest.mark.parametrize("text", [
    "Traceback (most recent call last):\n  File x\nValueError: boom\n",
    "\nSWEEP-FAIL: process produced no new best.pth\n",
    "RuntimeError: CUDA out of memory\n",
    "Error: something went wrong\n",
    # Dotted exception names, which a "first character is uppercase" test would
    # miss, and a traceback printed straight after a progress-bar carriage
    # return, which a plain line split would not see at a line start.
    "torch.cuda.OutOfMemoryError: tried to allocate 12.00 GiB\n",
    "epoch 3/200 |####      |\rTraceback (most recent call last):\n",
    "ValueError\n",
])
def test_real_failures_are_still_detected(text):
    assert _log_reports_failure(text) is True


# --------------------------------------------------------------------------
# Naming: CaFA is the authors' weather MODEL, the operator is FA, and every
# attention arm here is SELF-attention. Legacy spellings must keep resolving
# so manifests and checkpoints written before the rename still load.
# --------------------------------------------------------------------------

def _combo_kwargs():
    coords = [(s, c, f) for s in range(2) for c in range(3) for f in range(2)][:10]
    return dict(num_numeric_features=5, num_states=2, num_commodities=3, num_flows=2,
                num_combos=len(coords), features_per_group=5, hidden_size=16,
                num_layers=1, combo_coords=coords, lattice_dims=(2, 3, 2),
                combo_encoder="embeddings", state_embed_dim=2, comm_embed_dim=2,
                flow_embed_dim=2)


# Only the fa_* pair reaches the authors' submodule; the rest are local code.
# tests/test_fa_smoke.py guards the same dependency at module scope and states
# why: an unguarded case makes `pytest tests/` red on a fresh clone and hides
# real regressions in the noise. Guard the one parameter rather than the file,
# so the six local pairs keep running without the submodule.
_needs_cafa_submodule = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "external" / "cafa-authors"
         / "libs" / "factorization_module.py").exists(),
    reason=("authors' CaFA submodule not initialized; run "
            "`git submodule update --init external/cafa-authors`"),
)


@pytest.mark.parametrize("legacy,current", [
    ("cross_attention_2d", "asa_2d"),
    ("cross_attention_3d", "asa_3d"),
    ("cross_attention_4d", "asa_4d"),
    ("cafa_2d", "fa_local_2d"),
    ("cafa_3d", "fa_local_3d"),
    ("cafa_4d", "fa_local_4d"),
    pytest.param("authors_cafa_2d", "fa_2d", marks=_needs_cafa_submodule),
])
def test_legacy_variant_spellings_build_the_same_model(legacy, current):
    from src.models import create_model

    kw = _combo_kwargs()
    old = create_model("gru", legacy, **kw)
    new = create_model("gru", current, **kw)
    assert type(old.axial) is type(new.axial)
    assert sum(p.numel() for p in old.parameters()) == sum(
        p.numel() for p in new.parameters()
    )


def test_axial_backend_is_self_attention_not_cross_attention():
    # Q, K and V are all projections of the same input. Cross-attention would
    # draw Q from a different source; the upstream kernel keeps that separation
    # too (LowRankKernel defaults u_y = u_x).
    from src.models.combo_attention import GroupSelfAttention

    module = GroupSelfAttention(4, 8)
    x = torch.randn(1, 2, 3, 4)
    out, attn = module(x)
    assert out.shape == (1, 2, 3, 8)
    assert attn.shape == (1, 2, 3, 3)
    assert module.q_proj.in_features == module.k_proj.in_features == module.v_proj.in_features


def test_subclass_backends_do_not_leak_the_parent_projections():
    # fa/fa_local replace the per-axis backend by clearing self.sa. If any other
    # attribute still pointed at the parent's ModuleList, those unused
    # projections would stay registered and inflate every params/FLOPs figure.
    from src.models.fa_local import LocalFactorizedAttention

    coords = [(s, c, f) for s in range(2) for c in range(3) for f in range(2)][:10]
    module = LocalFactorizedAttention(5, 16, (2, 3, 2), coords, dims=3)
    groups = {name.split(".")[0] for name, _ in module.named_parameters()}
    assert "sa" not in groups and "ca" not in groups


def test_grid_tag_replaces_the_attention_tag_for_attention_free_ssms():
    from src.models.scan_schedule import VARIANT_CAT_AXES as _VARIANT_CAT_AXES

    for d, axes in ((2, (2,)), (3, (2, 3)), (4, (2, 3, 4))):
        assert _VARIANT_CAT_AXES[f"grid_{d}d"] == axes
        assert _VARIANT_CAT_AXES[f"cross_attention_{d}d"] == axes


def test_cost_record_quotes_effective_batch_in_observations():
    # The cost record and the audit log must agree. Upstream's FLOP accounting
    # landed while this was still batch_size * grad_accum, which is 1 for the
    # combo arm and would record a step-matched pair as a huge mismatch.
    trainer = _trainer(batch_size=1)
    trainer.combo = True
    trainer.effective_observations = trainer._effective_observations(
        _FakeLoader(_FakeComboDataset(18))
    )
    recorded = (trainer.effective_observations
                or trainer.effective_batch_size
                or trainer.batch_size * trainer.grad_accum)
    assert recorded == 18


# --------------------------------------------------------------------------
# FA head geometry: pinned in the main table, with an opt-in sensitivity arm.
# --------------------------------------------------------------------------

def test_fa_head_geometry_is_pinned_in_the_variant_configs():
    # FA geometry is scaled for this benchmark's 128-dim residual width: four
    # 32-dim value heads give a 128-dim value space, kernel_multiplier=2 gives
    # 64-dim Q/K per head. A benchmark-specific choice, not the authors'
    # 768-wide geometry. Pinned in advance so the main table is never selected
    # from the ablation.
    for dims in (2, 3, 4):
        text = Path(f"config/variants/fa_{dims}d.yaml").read_text()
        assert "fa_heads: 4" in text
        assert "fa_dim_head: 32" in text
        assert "fa_kernel_multiplier: 2" in text


def test_no_head_width_ablation_reaches_the_matrix():
    """The FA head-width ablation is gone; nothing may re-add a `dh` tag.

    Its runs could never have been scored: evaluate.py strips `dh<N>` into a
    leaderboard label and rebuilds from config/variants/fa_2d.yaml, which pins
    fa_dim_head: 32, so dh16/dh64 checkpoints were rebuilt at width 32 and
    load_state_dict failed -- reported as EVAL-FAIL, after which evaluate.py
    refuses to write any official result file. The geometry stays pinned in the
    variant configs (see the test above); the sweep no longer emits the arm.
    """
    from scripts.sweep import build_matrix, DEFAULT_SEEDS, FLAT_MODELS

    names = [n for n, _ in build_matrix(
        {"1", "1.1", "2", "3", "4", "6"}, DEFAULT_SEEDS, 2048, 1,
        accum=14, effective_batch_size=28292)]
    assert not any("_dh" in n for n in names)
def test_ablation_runs_score_as_their_own_rows_but_build_from_the_base_variant():
    from scripts.evaluate import parse_run

    cfg = parse_run("gru_fa_2d_embeddings_dh16_s947")
    assert cfg["variant"] == "fa_2d_dh16", "each head width needs its own row"
    assert cfg["build_variant"] == "fa_2d", "must still build from the base variant"
    assert cfg["enc"] == "embeddings" and cfg["seed"] == 947
    # The untagged run is unaffected.
    plain = parse_run("gru_fa_2d_embeddings_s947")
    assert plain["variant"] == "fa_2d"


# --------------------------------------------------------------------------
# An unrecognised variant must fail loudly, never fall back to a flat model.
# --------------------------------------------------------------------------

def _dispatch_kwargs():
    coords = [(s, c, f) for s in range(4) for c in range(6) for f in range(2)][:40]
    return dict(num_numeric_features=35, num_states=4, num_commodities=6, num_flows=2,
                num_combos=len(coords), features_per_group=35, hidden_size=32,
                num_layers=1, combo_coords=coords, lattice_dims=(4, 6, 2),
                combo_encoder="embeddings", state_embed_dim=2, comm_embed_dim=2,
                flow_embed_dim=2)


@pytest.mark.parametrize("host", ("gru", "lstm", "transformer", "gpt"))
@pytest.mark.parametrize("variant", ("grid_2d", "totally_bogus"))
def test_unknown_variant_raises_instead_of_building_a_flat_model(host, variant):
    # `grid_*d` is an SSM-only tag. transformer and lstm used to end their
    # factory with an unguarded `return <Flat>Model(variant=variant)`, so any
    # variant the encoder happened to accept silently trained a per-series
    # model that the run name claimed was axial.
    from src.models import create_model

    with pytest.raises(ValueError):
        create_model(host, variant, **_dispatch_kwargs())


@pytest.mark.parametrize("host", ("gru", "lstm", "transformer", "gpt"))
def test_flat_and_axial_variants_still_dispatch(host):
    from src.models import create_model

    kw = _dispatch_kwargs()
    # Flat classes do not all declare the attribute, so read it defensively.
    for variant in ("onehot", "embeddings"):
        model = create_model(host, variant, **kw)
        assert not getattr(model, "requires_combo_loader", False)
    for variant in ("asa_2d", "cross_attention_2d"):   # current + legacy
        model = create_model(host, variant, **kw)
        assert getattr(model, "requires_combo_loader", False)


def test_combo_variants_build_no_throwaway_encoder():
    # Every combo run used to build a full categorical encoder (170,416 params
    # at census cardinalities) only for _build_model to replace it with
    # nn.Identity. Combo variants now skip encoder construction outright.
    from src.models import create_model
    import src.models.base as base

    built = []
    original = base.BaseModel._create_encoder

    def probe(self, variant, **kwargs):
        encoder = original(self, variant, **kwargs)
        built.append(sum(p.numel() for p in encoder.parameters()))
        return encoder

    base.BaseModel._create_encoder = probe
    try:
        create_model("gru", "asa_2d", **_dispatch_kwargs())
        assert built == [0], f"combo run allocated a discarded encoder: {built}"
        built.clear()
        create_model("gru", "embeddings", **_dispatch_kwargs())
        assert built and built[0] > 0, "the flat path must still build its encoder"
    finally:
        base.BaseModel._create_encoder = original
