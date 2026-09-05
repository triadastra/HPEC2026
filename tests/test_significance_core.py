"""The statistical core, checked against scipy and against known answers.

This is the code that produces the paper's significance claims, and it had no
unit tests at all -- the only existing coverage checked that the pipeline
passed CLI flags through. It has already carried one silent numerical bug:
`p_noninf` computed as `1.0 - sf(x)` returned exactly 0.0 where the true value
was ~1.8e-37, because sf(x) rounds to 1.0 and the subtraction cancels. A
p-value of exactly zero reads as overwhelming evidence, which is the most
dangerous direction for it to be wrong in. Nothing pinned that fix until now.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.significance_tests import (PAIRS, _dm_moments, _t_isf, _t_sf,
                                        dm_standard, equivalence_margin,
                                        paired_t, tost_dm, wilcoxon_series)
from scripts.sweep import DEFAULT_SEEDS, build_matrix

scipy_stats = pytest.importorskip("scipy.stats")


# --------------------------------------------------------------------------
# t-distribution tails
# --------------------------------------------------------------------------

@pytest.mark.parametrize("df", [1, 2, 5, 30, 191])
@pytest.mark.parametrize("x", [-40.0, -12.0, -3.0, -0.5, 0.0, 0.5, 3.0, 12.0, 40.0])
def test_t_survival_matches_scipy(df, x):
    assert _t_sf(x, df) == pytest.approx(scipy_stats.t.sf(x, df), rel=1e-9, abs=1e-300)


@pytest.mark.parametrize("df", [2, 5, 30, 191])
@pytest.mark.parametrize("p", [0.4, 0.05, 0.01, 1e-4])
def test_t_inverse_survival_matches_scipy(df, p):
    assert _t_isf(p, df) == pytest.approx(scipy_stats.t.isf(p, df), rel=1e-7)


def test_the_far_tail_does_not_cancel_to_zero():
    """The bug: the LOWER tail written as `1 - sf(x)`.

    For large negative x, sf(x) rounds to 1.0, so the subtraction gives
    exactly 0.0 and the true (tiny) probability is destroyed. Taking sf(-x)
    instead keeps every significant digit. A p-value of exactly zero reads as
    overwhelming evidence, so this fails in the most dangerous direction.
    """
    x, df = -13.0, 191
    naive = 1.0 - _t_sf(x, df)
    correct = _t_sf(-x, df)
    assert naive == 0.0, "precondition: the naive form really does cancel"
    assert 0.0 < correct < 1e-25
    assert correct == pytest.approx(scipy_stats.t.cdf(x, df), rel=1e-9)


def test_tost_uses_the_non_cancelling_form():
    """End to end: a hugely non-inferior model must not report p_noninf == 0."""
    a, b = _mats(-5.0, 0.001, seed=11)      # A wins by a mile
    r = tost_dm(a, b, delta=0.05)
    assert r["p_noninf"] > 0.0, "the lower tail cancelled to exactly zero"
    assert r["p_noninf"] < 1e-20, "and it should still be a tiny number"


# --------------------------------------------------------------------------
# TOST
# --------------------------------------------------------------------------

def _mats(diff_mean, diff_sd, n_series=64, n_months=24, seed=0):
    """Two loss matrices whose per-month differential has a known mean/SD."""
    rng = np.random.default_rng(seed)
    base = rng.random((n_series, n_months)) + 1.0
    per_month = rng.normal(diff_mean, diff_sd, size=n_months)
    return base + per_month[None, :], base


def test_identical_models_are_equivalent():
    a, b = _mats(0.0, 1e-4)
    r = tost_dm(a, b, delta=0.05)
    assert r["equivalent"] is True
    assert r["p_tost"] < 0.05
    assert not r["degenerate"]


def test_a_clearly_worse_model_is_not_equivalent():
    a, b = _mats(0.5, 0.01)          # A loses by 0.5, margin is 0.05
    r = tost_dm(a, b, delta=0.05)
    assert r["equivalent"] is False
    assert r["p_tost"] > 0.05
    assert r["p_notsup"] < 0.05, "A is certainly not better than B"


def test_tost_is_dual_to_its_confidence_interval():
    """Equivalence at alpha holds exactly when the (1-2a) CI sits inside
    the margin -- the property that makes the printed interval readable."""
    for mean in (0.0, 0.01, 0.03, 0.06, 0.2):
        a, b = _mats(mean, 0.02, seed=1)
        r = tost_dm(a, b, delta=0.05, alpha=0.05)
        inside = r["lo"] > -r["delta"] and r["hi"] < r["delta"]
        assert r["equivalent"] is inside
        assert (r["p_tost"] < 0.05) == inside


def test_the_two_halves_answer_different_questions():
    """p_notsup: 'A is not better by delta'. p_noninf: 'A is not worse by
    delta'. A model that clearly wins must reject one and not the other."""
    a, b = _mats(-0.5, 0.01)          # A wins by 0.5
    r = tost_dm(a, b, delta=0.05)
    assert r["p_noninf"] < 0.05, "A is certainly not worse"
    assert r["p_notsup"] > 0.5, "A IS better, so this half must not reject"


def test_neither_p_value_is_ever_exactly_zero_for_finite_data():
    """The cancellation bug's signature. A p of exactly 0 is not a number the
    t-distribution produces from finite data; it is a lost computation."""
    for mean in (-2.0, -0.5, 0.0, 0.5, 2.0):
        r = tost_dm(*_mats(mean, 0.01, seed=2), delta=0.05)
        assert r["p_notsup"] > 0.0, f"p_notsup cancelled at mean={mean}"
        assert r["p_noninf"] > 0.0, f"p_noninf cancelled at mean={mean}"


def test_tost_p_values_match_scipy_on_the_same_moments():
    a, b = _mats(0.02, 0.03, seed=3)
    r = tost_dm(a, b, delta=0.05)
    dbar, se, T = _dm_moments(a, b)
    df = T - 1
    assert r["p_notsup"] == pytest.approx(
        scipy_stats.t.sf((dbar + 0.05) / se, df), rel=1e-9)
    assert r["p_noninf"] == pytest.approx(
        scipy_stats.t.cdf((dbar - 0.05) / se, df), rel=1e-9)


def test_a_zero_variance_differential_is_flagged_not_hidden():
    base = np.random.default_rng(4).random((16, 12)) + 1.0
    r = tost_dm(base + 0.01, base, delta=0.05)
    assert r["degenerate"] is True
    assert r["se"] == 0.0
    assert r["equivalent"] is True, "|0.01| < 0.05"


@pytest.mark.parametrize("bad_delta", [0.0, -0.1])
def test_a_nonpositive_margin_is_refused(bad_delta):
    a, b = _mats(0.0, 0.01)
    with pytest.raises(ValueError, match="margin must be positive"):
        tost_dm(a, b, delta=bad_delta)


@pytest.mark.parametrize("bad_alpha", [0.0, 0.5, 0.9, -0.1])
def test_an_alpha_outside_the_open_half_is_refused(bad_alpha):
    """At alpha >= 0.5 the TOST interval inverts and equivalence becomes
    trivially true."""
    a, b = _mats(0.0, 0.01)
    with pytest.raises(ValueError, match="alpha must be in"):
        tost_dm(a, b, delta=0.05, alpha=bad_alpha)


# --------------------------------------------------------------------------
# the margin itself
# --------------------------------------------------------------------------

def test_absolute_margin_is_taken_literally():
    assert equivalence_margin("abs", 0.03, np.ones((4, 4))) == pytest.approx(0.03)


def test_fractional_margin_scales_with_the_reference_loss():
    ref = np.full((4, 4), 0.2)
    assert equivalence_margin("frac", 0.05, ref) == pytest.approx(0.01)


def test_seed_margin_scales_with_the_reseeding_spread():
    diffs = [0.10, 0.14, 0.18]
    expected = 0.5 * float(np.std(diffs, ddof=1))
    assert equivalence_margin("seed", 0.5, None, diffs) == pytest.approx(expected)


def test_seed_margin_refuses_a_single_seed():
    """An SD on one observation is not a spread."""
    with pytest.raises(ValueError, match="at least two seeds"):
        equivalence_margin("seed", 0.5, None, [0.1])


def test_an_unknown_margin_mode_is_refused():
    with pytest.raises(ValueError, match="unknown delta mode"):
        equivalence_margin("percentile", 0.5, np.ones((2, 2)))


# --------------------------------------------------------------------------
# the difference tests
# --------------------------------------------------------------------------

def test_dm_agrees_with_a_paired_t_on_the_month_means():
    a, b = _mats(0.05, 0.02, seed=5)
    stat, p = dm_standard(a, b)
    months = (a - b).mean(axis=0)
    ref = scipy_stats.ttest_1samp(months, 0.0)
    assert stat == pytest.approx(ref.statistic, rel=1e-6)
    assert p == pytest.approx(ref.pvalue, rel=1e-6)


def test_dm_needs_more_than_one_test_month():
    with pytest.raises(ValueError, match="at least two test months"):
        dm_standard(np.ones((4, 1)), np.zeros((4, 1)))


@pytest.mark.parametrize("n", [50, 1000, 28292])
def test_wilcoxon_matches_the_normal_approximation_scipy_uses_at_scale(n):
    """The implementation is a normal approximation, deliberately.

    scipy's default switches to the EXACT distribution for small samples, so
    comparing against the default disagrees at n=50 -- that is scipy changing
    method, not an error here. At the benchmark's own scale (28,292 series)
    scipy uses the approximation too, and the two agree exactly.
    """
    rng = np.random.default_rng(6)
    ua, ub = rng.random(n), rng.random(n)
    _, p = wilcoxon_series(ua, ub)
    ref = scipy_stats.wilcoxon(ua, ub, method="approx").pvalue
    assert p == pytest.approx(ref, rel=1e-9)


def test_wilcoxon_reports_nan_rather_than_dividing_by_zero_on_all_ties():
    tied = np.arange(10.0)
    z, p = wilcoxon_series(tied, tied)
    assert np.isnan(z) and np.isnan(p)


def test_paired_t_matches_scipy():
    rng = np.random.default_rng(7)
    ua, ub = rng.random(50), rng.random(50)
    stat, p = paired_t(ua, ub)
    ref = scipy_stats.ttest_rel(ua, ub)
    assert stat == pytest.approx(ref.statistic, rel=1e-6)
    assert p == pytest.approx(ref.pvalue, rel=1e-6)


# --------------------------------------------------------------------------
# the hardcoded comparisons must name runs the matrix still produces
# --------------------------------------------------------------------------

def test_every_declared_pair_names_a_run_the_matrix_produces():
    """PAIRS is a hardcoded list of run-name prefixes, and the roster has been
    renamed under it before (cafa_* -> fa_*, cross_attention_* -> asa_*, and
    fa_local leaving entirely). A stale name here raises KeyError in the sig
    stage -- after the whole 477-run sweep has already been paid for."""
    runs = build_matrix({"1", "1.1", "2", "3", "4", "6"}, DEFAULT_SEEDS, 2048, 1,
                        accum=14, effective_batch_size=28292)
    # Exp 7 and Exp 8 are declared arms too -- diagnostic, run alone, outside
    # the 477 -- and their cells are legitimate pair operands. They build from
    # their own matrices, so a rename there has to fail this test as loudly as
    # a rename inside the main sweep would.
    for tid in ("7", "8"):
        runs = runs + build_matrix({tid}, DEFAULT_SEEDS, 2048, 1)
    produced = {re.sub(r"_s\d+$", "", name) for name, _ in runs}
    referenced = set()
    for _, a, b in PAIRS:
        referenced.add(a)
        if b is not None:
            referenced.add(b)
    missing = sorted(referenced - produced)
    assert not missing, (
        f"significance_tests.PAIRS names runs the matrix never emits: {missing}")
