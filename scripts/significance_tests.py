#!/usr/bin/env python
"""Standard-form significance suite on paired forecast losses.

- Diebold-Mariano (1995) on the T=24 monthly cross-sectional mean loss
  differentials: h=1 => truncation lag h-1=0 (plain variance), the
  Harvey-Leybourne-Newbold (1997) small-sample correction, p from t_{T-1}.
  Run for squared loss and mean absolute error.
- Wilcoxon signed-rank on per-series mean absolute errors.
- Paired t-test on per-series mean squared losses (parametric reference).

Those three are DIFFERENCE tests: their null is "no difference", so a large
p-value is a failure to detect a difference and is NOT evidence that two models
are equivalent. A thesis of the form "the N-D arm scores the same as the flat
arm" has to be argued the other way round, so the suite also runs:

- TOST (two one-sided tests) for equivalence within a margin +/-delta, on the
  same DM differentials and with the same HLN-corrected standard error, so the
  equivalence claim and the difference claim rest on identical footing.
- The two one-sided halves reported separately, because they answer different
  questions: `p_notsup` (A is not better than B by delta) is what a "structure
  does not help" claim needs; `p_noninf` (A is not worse than B by delta) is
  the ordinary non-inferiority statement.
- The (1-2*alpha) confidence interval on the mean differential. Equivalence at
  level alpha holds exactly when that interval lies inside (-delta, +delta),
  which is the form a reader can check by eye.

Choosing delta is a modelling decision, not a statistical one; see
``equivalence_margin`` for the three supported conventions.

Requires an error dump produced by the current scripts/dump_errors.py.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

REPO = Path(__file__).resolve().parent.parent


def _t_sf(x, df):
    """P(T_df > x), falling back to the normal tail when scipy is absent.

    The existing difference tests already degrade to a normal approximation
    without scipy; the equivalence tests use the same fallback so the two
    families never disagree because of how the tail was computed.
    """
    if HAVE_SCIPY:
        return float(stats.t.sf(x, df=df))
    return 0.5 * math.erfc(x / math.sqrt(2))


def _t_isf(p, df):
    """x such that P(T_df > x) = p, by bisection so the scipy-free path agrees.

    Parameterised by the UPPER tail rather than by a cumulative probability:
    forming ``1 - alpha`` first discards precision exactly where the quantile
    is steepest -- the same cancellation the survival function avoids elsewhere.
    """
    if HAVE_SCIPY:
        return float(stats.t.isf(p, df=df))
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_sf(mid, df) > p:   # sf decreasing: still left of the target
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def load_errors(path):
    dump = np.load(str(path))
    if "__metadata__" not in dump.files:
        raise RuntimeError(
            "errdump.npz has no provenance metadata; regenerate it with the current "
            "scripts/dump_errors.py"
        )
    metadata = json.loads(str(dump["__metadata__"].item()))
    if metadata.get("schema_version") != 3:
        raise RuntimeError(
            "unsupported errdump provenance schema; regenerate it with the current "
            "scripts/dump_errors.py"
        )
    expected_shape = tuple(metadata.get("shape", ()))
    if len(expected_shape) != 2:
        raise RuntimeError("errdump provenance does not declare an (N,T) shape")
    return dump, metadata, expected_shape


def seedstack(dump, metadata, expected_shape, prefix, loss, seeds=(947, 732, 619)):
    """Per-seed loss matrices, validated exactly as seedavg does.

    Kept separate because the seed-scaled equivalence margin needs the
    individual seeds rather than their mean.
    """
    suffix = "" if loss == "squared" else "__absolute"
    names = [f"{prefix}_s{seed}" for seed in seeds]
    keys = [f"{name}{suffix}" for name in names]
    missing = [key for key in keys if key not in dump.files]
    if missing:
        raise KeyError(f"{prefix} is missing required {loss} arrays: {missing}")
    unbound = [name for name in names if name not in metadata.get("runs", {})]
    if unbound:
        raise KeyError(f"{prefix} matrices lack checkpoint provenance: {unbound}")
    mats = [dump[key] for key in keys]
    bad_shapes = [key for key, matrix in zip(keys, mats) if matrix.shape != expected_shape]
    if bad_shapes:
        raise ValueError(f"{prefix} matrices have the wrong shape: {bad_shapes}")
    return mats


def seedavg(dump, metadata, expected_shape, prefix, loss, seeds=(947, 732, 619)):
    return np.mean(
        seedstack(dump, metadata, expected_shape, prefix, loss, seeds), axis=0
    )


def baseline_matrix(dump, expected_shape, loss):
    """The persistence anchor, gated on the same (N,T) contract as the runs."""
    key = "persistence" if loss == "squared" else "persistence__absolute"
    if key not in dump.files:
        raise KeyError(f"errdump has no {key} baseline array")
    matrix = dump[key]
    if matrix.shape != expected_shape:
        raise ValueError(f"{key} has shape {matrix.shape}, expected {expected_shape}")
    return matrix


def per_seed_diffs(dump, metadata, expected_shape, a, b, loss,
                   seeds=(947, 732, 619)):
    """Mean loss differential for each seed, for the seed-scaled margin."""
    A = seedstack(dump, metadata, expected_shape, a, loss, seeds)
    if b is None:
        B = [baseline_matrix(dump, expected_shape, loss)] * len(A)
    else:
        B = seedstack(dump, metadata, expected_shape, b, loss, seeds)
    return np.array([float((x - y).mean()) for x, y in zip(A, B)])


def _dm_moments(matA, matB):
    """Mean monthly loss differential, its HLN-corrected SE, and T.

    Factored out so the difference test and the equivalence tests are computed
    from one variance estimate with one small-sample correction. Derived
    separately, a disagreement between "no significant difference" and "not
    significantly equivalent" could be an artifact of the standard error rather
    than of the data. ``se`` is None when the loss paths are identical.
    """
    dt = (matA - matB).mean(axis=0)
    T = dt.size
    if T < 2:
        raise ValueError("DM needs at least two test months")
    dbar = float(dt.mean())
    var = float(((dt - dbar) ** 2).mean())
    if var <= 0.0:
        return dbar, None, T
    h = 1
    scale = math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return dbar, math.sqrt(var / T) / scale, T


def dm_standard(matA, matB):
    """DM h=1: lag-0 variance, HLN correction, t_{T-1} p-value."""
    dbar, se, T = _dm_moments(matA, matB)
    if se is None:
        # Identical loss paths: the statistic is undefined, not "infinitely
        # significant". Reporting inf/NaN as a test result is worse than saying
        # the comparison carries no information.
        return float("nan"), float("nan")
    dm = dbar / se
    return dm, 2 * _t_sf(abs(dm), df=T - 1)


def tost_dm(matA, matB, delta, alpha=0.05):
    """TOST equivalence on the DM differentials, plus each one-sided half.

    ``d = lossA - lossB``; lower loss is better, so d < 0 means A wins.

      p_notsup  H0: d <= -delta  ("A beats B by at least delta").
                Small p => A is NOT meaningfully better than B. This is the
                half a "the extra structure does not buy accuracy" claim needs.
      p_noninf  H0: d >= +delta  ("A loses to B by at least delta").
                Small p => A is not meaningfully worse than B.
      p_tost    max of the two; small => |d| < delta, i.e. equivalence.

    The reported interval is the (1-2*alpha) CI, which is the interval TOST is
    dual to: equivalence at ``alpha`` holds exactly when it sits inside
    (-delta, +delta). Sets ``degenerate`` when the differentials are identical;
    the limiting p-values are still returned, but a zero-variance sample is a
    data problem and the row should be read as such.
    """
    if not delta > 0:
        raise ValueError(f"equivalence margin must be positive, got {delta!r}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(
            f"alpha must be in (0, 0.5) for TOST to be meaningful, got {alpha!r}"
        )
    dbar, se, T = _dm_moments(matA, matB)
    df = T - 1
    if se is None:
        p_notsup = 0.0 if dbar > -delta else 1.0
        p_noninf = 0.0 if dbar < delta else 1.0
        return dict(dbar=dbar, se=0.0, delta=delta, lo=dbar, hi=dbar,
                    p_notsup=p_notsup, p_noninf=p_noninf,
                    p_tost=max(p_notsup, p_noninf),
                    equivalent=bool(-delta < dbar < delta),
                    half=0.0, degenerate=True)
    p_notsup = _t_sf((dbar + delta) / se, df=df)
    # Lower tail via sf(-x), not 1-sf(x): when the non-inferiority evidence is
    # strong, sf(x) rounds to 1.0 and the subtraction cancels the whole result
    # to exactly 0 instead of reporting the true (tiny) p-value.
    p_noninf = _t_sf(-(dbar - delta) / se, df=df)
    half = _t_isf(alpha, df=df) * se
    lo, hi = dbar - half, dbar + half
    return dict(dbar=dbar, se=se, delta=delta, lo=lo, hi=hi,
                p_notsup=p_notsup, p_noninf=p_noninf,
                p_tost=max(p_notsup, p_noninf),
                equivalent=bool(lo > -delta and hi < delta), half=half,
                degenerate=False)


def equivalence_margin(mode, value, ref_mat, seed_diffs=None):
    """Resolve delta, in loss units. Delta is an assumption -- state it.

      abs   delta = value. Only meaningful if the reader knows the loss scale.
      frac  delta = value * mean(ref_mat): a fraction of the comparison
            baseline's own loss, so "within 5% of the flat arm".
      seed  delta = value * SD of the per-seed mean differential: "smaller than
            the spread we get from reseeding the same model". The most
            defensible choice for an equivalence thesis, but with three seeds
            the SD is itself a noisy estimate on 2 df -- report it, do not
            silently treat it as a known constant.
    """
    if mode == "abs":
        return float(value)
    if mode == "frac":
        return float(value) * abs(float(np.mean(ref_mat)))
    if mode == "seed":
        if seed_diffs is None or len(seed_diffs) < 2:
            raise ValueError("seed-scaled delta needs at least two seeds")
        return float(value) * float(np.std(seed_diffs, ddof=1))
    raise ValueError(f"unknown delta mode {mode!r}")


def wilcoxon_series(uA, uB):
    diff = uA - uB
    dd = diff[diff != 0]
    n = dd.size
    if n == 0:
        # Every series tied: no signed ranks to sum, so the normal
        # approximation's standard deviation is zero.
        return float("nan"), float("nan")
    # Average ranks and the matching tie correction, including without scipy.
    _, inverse, counts = np.unique(np.abs(dd), return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - (counts - 1) / 2.0)[inverse]
    wplus = float(ranks[dd > 0].sum())
    mu = n * (n + 1) / 4.0
    ties = counts.astype(np.float64)
    variance = (n * (n + 1) * (2 * n + 1) - ((ties ** 3 - ties).sum() / 2)) / 24.0
    sd = math.sqrt(variance)
    z = (wplus - mu) / sd
    return z, math.erfc(abs(z) / math.sqrt(2))


def paired_t(uA, uB):
    diff = uA - uB
    n = diff.size
    if n < 2:
        raise ValueError("paired t-test needs at least two series")
    sd = float(diff.std(ddof=1))
    if sd <= 0.0:
        return float("nan"), float("nan")
    t = diff.mean() / (sd / math.sqrt(n))
    p = (2 * stats.t.sf(abs(t), df=n - 1) if HAVE_SCIPY
         else math.erfc(abs(t) / math.sqrt(2)))
    return t, p


PAIRS = [
    ("S4ND-4D vs S4 flat",        "s4nd_grid_4d_embeddings", "s4_embeddings_1d"),
    ("S4 flat vs Transformer",    "s4_embeddings_1d", "transformer_embeddings_1d"),
    ("S4 flat vs GRU FA-2D",      "s4_embeddings_1d", "gru_fa_2d_embeddings"),
    ("FA authors vs ASA 2-D",     "gru_fa_2d_embeddings", "gru_asa_2d_embeddings"),
    ("identity vs blind (Tr 2D)", "transformer_asa_2d_id_embeddings",
                                  "transformer_asa_2d_embeddings"),
    # Exp 7 / Exp 8. The 2-D cells are the ones with all four mixers dumped, so
    # they are where the operator question can be tested rather than eyeballed.
    # fa_local is OUR build of the FA operator (the draft's "CaFA"), fa is the
    # authors' released one, fa_sm is theirs on their own softmax switch.
    ("fa_local vs FA authors (GRU 2D)", "gru_fa_local_2d_embeddings",
                                        "gru_fa_2d_embeddings"),
    ("fa_sm vs FA authors (GRU 2D)",    "gru_fa_sm_2d_embeddings",
                                        "gru_fa_2d_embeddings"),
    ("fa_local vs fa_sm (GRU 2D)",      "gru_fa_local_2d_embeddings",
                                        "gru_fa_sm_2d_embeddings"),
    ("fa_local vs ASA (GRU 2D)",        "gru_fa_local_2d_embeddings",
                                        "gru_asa_2d_embeddings"),
    ("fa_sm vs ASA (GRU 2D)",           "gru_fa_sm_2d_embeddings",
                                        "gru_asa_2d_embeddings"),
    ("fa_local vs ASA (Tr 2D)",         "transformer_fa_local_2d_embeddings",
                                        "transformer_asa_2d_embeddings"),
    ("fa_sm vs ASA (Tr 2D)",            "transformer_fa_sm_2d_embeddings",
                                        "transformer_asa_2d_embeddings"),
    ("fa_local vs fa_sm (Tr 2D)",       "transformer_fa_local_2d_embeddings",
                                        "transformer_fa_sm_2d_embeddings"),
    ("fa_local vs ASA (gru 3D)", "gru_fa_local_3d_embeddings", "gru_asa_3d_embeddings"),
    ("fa_local vs ASA (gru 4D)", "gru_fa_local_4d_embeddings", "gru_asa_4d_embeddings"),
    ("fa_local vs ASA (lstm 3D)", "lstm_fa_local_3d_embeddings", "lstm_asa_3d_embeddings"),
    ("fa_local vs ASA (lstm 4D)", "lstm_fa_local_4d_embeddings", "lstm_asa_4d_embeddings"),
    ("fa_local vs ASA (transformer 3D)", "transformer_fa_local_3d_embeddings", "transformer_asa_3d_embeddings"),
    ("fa_local vs ASA (transformer 4D)", "transformer_fa_local_4d_embeddings", "transformer_asa_4d_embeddings"),
    # Exp 7/8 backfill: the 3-D and 4-D asa/fa cells the original sweep could
    # never dump. These decide whether the high-variance lstm wins survive.
    ("fa vs ASA (gru 3D)", "gru_fa_3d_embeddings", "gru_asa_3d_embeddings"),
    ("fa vs ASA (gru 4D)", "gru_fa_4d_embeddings", "gru_asa_4d_embeddings"),
    ("fa vs ASA (lstm 3D)", "lstm_fa_3d_embeddings", "lstm_asa_3d_embeddings"),
    ("fa vs ASA (lstm 4D)", "lstm_fa_4d_embeddings", "lstm_asa_4d_embeddings"),
    ("fa vs ASA (transformer 3D)", "transformer_fa_3d_embeddings", "transformer_asa_3d_embeddings"),
    ("fa vs ASA (transformer 4D)", "transformer_fa_4d_embeddings", "transformer_asa_4d_embeddings"),
]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--errors", default=str(REPO / "outputs/sweep/errdump.npz"),
        help="provenance-bound loss dump from scripts/dump_errors.py",
    )
    parser.add_argument(
        "--delta-mode", default="frac", choices=("abs", "frac", "seed"),
        help="how the equivalence margin is defined (see equivalence_margin)",
    )
    parser.add_argument(
        "--delta", type=float, default=0.05,
        help="margin value: absolute loss units (abs), a fraction of the "
             "baseline loss (frac, default 0.05 = 5%%), or a multiple of the "
             "per-seed SD (seed)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="one-sided level for TOST; the printed CI is (1-2*alpha)",
    )
    args = parser.parse_args(argv)
    if not args.delta > 0:
        parser.error("--delta must be positive")
    if not 0.0 < args.alpha < 0.5:
        parser.error("--alpha must lie in (0, 0.5) for TOST to be meaningful")
    dump, metadata, expected_shape = load_errors(Path(args.errors).resolve())

    print(f"scipy={HAVE_SCIPY}")
    print(f"{'pair':28} {'dMSE':>10} | {'DM_sq':>7} {'p':>7} | {'DM_abs':>7} {'p':>7} | "
          f"{'Wilcx_abs z':>11} {'p':>9} | {'t_sq':>8} {'p':>9}")
    equiv_rows = []
    pairs = PAIRS + [
        ("S4 flat vs persistence", "s4_embeddings_1d", None),
        ("S4ND-4D vs persistence", "s4nd_grid_4d_embeddings", None),
    ]
    # A pair whose runs are absent from the dump used to raise, which aborted
    # the whole table -- one un-dumped cell (identity vs blind has been waiting
    # on a GPU since 2026-08-30) took every pair after it down with it, and the
    # ones that could have been computed were never printed. Defer that pair and
    # keep going; the deferred list is reported at the end, where it can be read
    # as "not yet measured" rather than mistaken for "not significant".
    deferred = []
    for label, a, b in pairs:
        try:
            A_sq = seedavg(dump, metadata, expected_shape, a, "squared")
            A_abs = seedavg(dump, metadata, expected_shape, a, "absolute")
            if b is not None:
                seedavg(dump, metadata, expected_shape, b, "squared")
        except KeyError as exc:
            deferred.append((label, str(exc).split(":")[0].strip('"')))
            continue
        if b is None:
            B_sq = baseline_matrix(dump, expected_shape, "squared")
            B_abs = baseline_matrix(dump, expected_shape, "absolute")
        else:
            B_sq = seedavg(dump, metadata, expected_shape, b, "squared")
            B_abs = seedavg(dump, metadata, expected_shape, b, "absolute")
        dm_s, p_s = dm_standard(A_sq, B_sq)
        dm_a, p_a = dm_standard(A_abs, B_abs)
        z, pw = wilcoxon_series(A_abs.mean(1), B_abs.mean(1))
        t, pt = paired_t(A_sq.mean(1), B_sq.mean(1))
        print(f"{label:28} {A_sq.mean()-B_sq.mean():+10.6f} | "
              f"{dm_s:7.2f} {p_s:7.3f} | {dm_a:7.2f} {p_a:7.3f} | "
              f"{z:11.2f} {pw:9.2e} | {t:8.2f} {pt:9.2e}")
        seed_diffs = (per_seed_diffs(dump, metadata, expected_shape, a, b, "squared")
                      if args.delta_mode == "seed" else None)
        delta = equivalence_margin(args.delta_mode, args.delta, B_sq, seed_diffs)
        equiv_rows.append((label, tost_dm(A_sq, B_sq, delta, alpha=args.alpha)))

    # Second table. The difference tests above cannot support "the two arms
    # score the same" -- that claim needs these.
    print()
    print(f"equivalence on squared-loss DM differentials | delta mode="
          f"{args.delta_mode} value={args.delta:g} alpha={args.alpha:g}")
    ci_hdr = str(int(round((1 - 2 * args.alpha) * 100))) + "% CI"
    print(f"{'pair':28} {'d':>10} {'delta':>10} | {ci_hdr:>21} | "
          f"{'p_notsup':>9} {'p_noninf':>9} {'p_TOST':>9} {'equiv':>6}")
    degenerate_seen = False
    for label, r in equiv_rows:
        ci = f"[{r['lo']:+.5f},{r['hi']:+.5f}]"
        if r.get("degenerate"):
            flag, degenerate_seen = "DEGEN", True
        else:
            flag = "YES" if r["equivalent"] else "no"
        print(f"{label:28} {r['dbar']:+10.6f} {r['delta']:10.6f} | {ci:>21} | "
              f"{r['p_notsup']:9.2e} {r['p_noninf']:9.2e} {r['p_tost']:9.2e} "
              f"{flag:>6}")
    print()
    print("d = lossA - lossB (negative => A better). p_notsup small => A is NOT")
    print("better than B by delta; p_TOST small => equivalent within +/-delta.")
    print("A large p is never evidence of equivalence -- read p_TOST, not DM p.")
    if degenerate_seen:
        print("DEGEN: zero-variance differentials -- inspect those inputs before "
              "quoting the row.")
    if deferred:
        print()
        print("DEFERRED -- no error dump for these cells, so they are UNMEASURED,")
        print("not 'no significant difference'. Dump them and rerun:")
        for label, why in deferred:
            print(f"  {label}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
