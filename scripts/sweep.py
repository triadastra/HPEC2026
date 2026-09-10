#!/usr/bin/env python
"""Job-parallel sweep launcher for the census HS6 benchmark (PLAN.md §4.1).

The experiment matrix is many *small independent* runs (model × variant ×
encoder × seed), so the right parallelism is **one whole run per GPU**, not DDP.
This scheduler pins each ``train.py`` subprocess to a GPU via
``CUDA_VISIBLE_DEVICES`` and keeps every GPU busy; it scales straight from 2 to
5+ GPUs with zero code change and no distributed-training complexity.

    python scripts/sweep.py --gpus 0,1 --dry-run              # print the matrix
    python scripts/sweep.py --gpus 0,1 --tests 3 --epochs 30  # embeds N-d only
    python scripts/sweep.py --gpus 0,1 --limit 4 --epochs 1   # quick smoke

Each run logs to ``outputs/sweep/logs/<name>.log``; a summary prints at the end.
"""
import argparse, itertools, json, math, os, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (DIVERGED_EXIT_CODE, DIVERGED_FILE, completion_path,
                       compose_data_config, run_fingerprint,
                       run_input_fingerprint, validate_completion,
                       write_completion)

REPO = Path(__file__).resolve().parent.parent
FLAT_MODELS = ["gru", "lstm", "transformer", "s4", "mamba2", "mamba3"]  # +flat SSMs: S4(DPLR), Mamba-2(SSD), Mamba-3
AXIAL_MODELS = ["gru", "lstm", "transformer"]        # models with axial/CaFA variants
AXIAL_SSM = ["mamba_nd"]                              # grid-native SSM: scan over promoted axes
HYBRID_SSM = ["mamba2", "mamba3"]                     # Test 6: axial/CaFA grid mixing + SSM temporal backbone
MECHS = ["asa", "fa"]                      # default grid-mixing mechanisms
# Off the declared roster, reachable with --mechs:
#
#   fa_local  a LOCAL reimplementation of the FA operator (src/models/fa_local.py).
#             Dropped from the benchmark: it differs from the authors' fa_* in at
#             least four ways at once (softmax vs LeakyReLU gating, gamma-MLP vs
#             PoolingReducer, single-head vs LowRankKernel, and a different sparse
#             renormalization), so "fa vs fa_local" cannot be attributed to any one
#             of them. Every FA claim in the paper now rests on the authors' own
#             components. The code and its regression tests stay: they carry the
#             sparse-renormalization fix and let a reviewer rerun the arm on request.
#   aca       axial CROSS-attention (src/models/aca.py) -- the one arm whose Q and K
#             come from different sources.
OPTIONAL_MECHS = ["fa_local", "aca"]

# The FA head-width ablation was removed: the paper does not need it, and its
# runs could never have been scored. evaluate.py strips the `dh<N>` tag into a
# leaderboard label and rebuilds from config/variants/fa_2d.yaml, which pins
# fa_dim_head: 32 -- so the dh16/dh64 checkpoints were rebuilt at width 32 and
# load_state_dict failed, which evaluate.py reports as EVAL-FAIL and then
# refuses to write any official result file. FA geometry stays pinned in the
# variant configs; `train.py --fa-dim-head` remains for one-off inspection.
# axial SELF-attention vs a local build of the FA operator vs the authors' FA.
# None of these is "CaFA": that is the authors' weather model, not the operator.
ALL_MECHS = MECHS + OPTIONAL_MECHS

# ---------------------------------------------------------------- Exp 0 ----
# Exp 0 is the learning-rate SELECTION stage, and it gates every other test.
#
# Why it exists: the flat arm had a learning-rate grid (Test 5) while every
# N-D configuration carried no --lr at all, inheriting either the base.yaml
# default or a per-model override established on the FLAT task under a
# different optimizer. Dodge et al. (EMNLP 2019) state the consequence: when a
# larger-budget arm beats a smaller-budget arm, the difference is attributable
# to the budget as much as to the model. This benchmark's headline is that the
# multidimensional arm does NOT earn its compute -- precisely the comparison
# the old protocol could not support. An N-D win under zero search would have
# been sound; a flat win is not.
#
# Per-model rates are correct and are kept. What Exp 0 fixes is that every
# compared cell now gets the same OPPORTUNITY to land on its own rate.
# Arms that YIELD a selected rate. "roll" is separate from "agg" because the
# rolling folds are a different optimisation problem: fold 2020 trains on
# roughly half the history the fixed split gets, and a rate chosen on the full
# split is not automatically right for it -- the same transfer assumption Exp 0
# already refuses to make silently for dimensionality.
# "mech" is the fa mixer's own rate. Kept under that name rather than renamed
# to "fa": sessions already on disk contain exp0_mech_* runs, and select_lr
# builds its run-name regex from this tuple, so a rename would make those
# unparseable and refuse the selection they belong to.
# "mechlocal" and "mechsm" are the same idea for the two diagnostic mixers:
# fa_local (Exp 7, our own build of the FA operator) and fa_sm (Exp 8, the
# authors' operator on their softmax switch). Each gets its own arm rather than
# inheriting "mech", because "mech" was probed on the authors' LeakyReLU fa_3d
# and the measured spread between arms is large enough that inheritance is not
# a rounding error: nd vs mech is 3e-4 vs 1e-2 on gru (33x) and reverses
# direction on transformer (1e-3 vs 3e-4). Handing an untuned arm to a tuned
# comparator is exactly the asymmetry F5 removed.
EXP0_ARMS = ("flat", "nd", "agg", "roll", "mech", "mechlocal", "mechsm")
# Arms that exist only to CHECK a transfer assumption; they are reported, never
# selected from. Each corresponds to one axis along which a probe's command
# line differs from the runs it selects rates for -- every such difference is
# an assumption, and the ones that are not checked are the ones that bite.
#
#   dim<N>    the N-D rate, across dimensionality (2-D vs 4-D)
#   rollend   the rolling rate, on the longest fold instead of the shortest
#   encflat   the flat rate, under the one-hot encoder (Test 2) instead of
#             embeddings
#   encnd     the N-D rate, under the one-hot combo encoder (Test 2)
#   axid      the N-D rate, with --axis-identity (Test 4), which adds learned
#             embeddings on the promoted axes and so changes the model
#   mech      the N-D rate, under the fa mixer (Tests 2/3/4/6). The probe runs
#             asa, and fa swaps the whole grid-mixing operator -- a larger
#             model change than the encoder swap encnd already checks
EXP0_CHECK_ARMS = ("dim", "rollend", "encflat", "encnd", "axid")

# Which selecting arm each check arm is testing the transfer OF.
EXP0_CHECK_BASE = {"rollend": "roll", "encflat": "flat",
                   "encnd": "nd", "axid": "nd"}


def exp0_arm_pattern():
    """Regex alternation matching every Exp 0 arm name.

    Built here and imported by select_lr.py rather than spelled out twice:
    an arm added in one place and missing in the other would make its runs
    unparseable, and select_lr.py refuses a session it cannot parse. Longest
    alternative first, so "rollend" cannot be split as "roll" + "end_<model>".
    """
    names = sorted(set(EXP0_ARMS) | set(EXP0_CHECK_ARMS) - {"dim"},
                   key=len, reverse=True)
    return "|".join(names) + r"|dim\d+"
ND_BACKBONES = AXIAL_MODELS + HYBRID_SSM + AXIAL_SSM + ["s4nd"]

# The probe variant differs by backbone because the roster does: attention
# hosts run asa_*, the grid-native SSMs run grid_*. Probing a backbone on a
# variant it never runs would select a rate for the wrong optimisation problem.
EXP0_PROBE_DIM = 3
EXP0_ND_MECH = {m: "grid" for m in AXIAL_SSM + ["s4nd"]}   # default below: asa

# Five rates, log-spaced by exactly sqrt(10), spanning two decades.
#
# Learning-rate effects are multiplicative, so a grid should spend its trials
# on ratios. The span is weighted to BRACKET the optimum rather than resolve it
# finely: near its optimum an LR curve is broad and flat, so a 3.16x step loses
# little, whereas landing on an endpoint is the visible failure that invites
# "you did not search enough" -- fatal for a negative result. A 2026-08-25
# probe hit the ceiling in all three arms at a 1e-3 top, with validation loss
# still improving monotonically toward it.
#
# Extension goes UP rather than down because the scheduler is asymmetric:
# reduce_on_plateau can only lower a rate, so a too-high start partly
# self-heals while a too-low start is never recovered. 1e-4 is retained
# because transformer and mamba3 are documented unstable at 1e-3.
#
# If 1e-2 diverges that is a result, not a wasted trial: it shows the grid
# brackets the optimum. It is also the cheapest cell in wall-clock, since a
# diverging run now stops at the first non-finite loss.
EXP0_LR_GRID = ("1e-4", "3e-4", "1e-3", "3e-3", "1e-2")

# Phase 0 spot-checks that a rate chosen at one dimensionality transfers to the
# others, on one attention backbone and one SSM backbone.
EXP0_DIM_CHECK_MODELS = ("gru", "s4nd")
EXP0_DIM_CHECK_DIMS = (2, 4)

# The rolling arm probes the SHORTEST fold -- the one with the least training
# history, where a too-high rate has the fewest steps to recover from. Probing
# every fold would cost 6x for a quantity that is broad and flat near its
# optimum; probing the hardest fold and spot-checking the easiest is the same
# bargain the dimensional check makes. mamba3 is in the check set because it is
# the documented-unstable backbone: if any model's rate fails to transfer
# across fold length, it is that one.
EXP0_ROLL_CHECK_MODELS = ("gru", "mamba3")

# Encoder and identity transfer checks. One attention host and one SSM host,
# the same bargain the dimensional check makes: probing every cell under every
# encoder would double the flat and N-D arms for a quantity that is broad and
# flat near its optimum.
EXP0_ENC_CHECK_MODELS = ("gru", "mamba3")
EXP0_ND_ENC_CHECK_MODELS = ("gru", "s4nd")
EXP0_ID_CHECK_MODELS = ("gru", "transformer")

# The attention hosts (Tests 2/3/4) and the hybrid SSMs (Test 6) each run BOTH
# mixers, but the N-D probe runs only asa -- all 144 fa cells of the declared
# matrix train at a rate measured on a different grid-mixing operator. One
# attention host and one hybrid host; mamba3 again because it is the
# documented-unstable backbone. mamba_nd and s4nd are excluded: they run only
# grid_* variants and have no fa cells to select for.
# Every N-D backbone, not a two-model spot check. The check version found
# that the fa mixer wants a materially different rate from asa -- for mamba3
# at 3-D, fa reaches 0.634 at its own 1e-3 but only 1.024 at the 1e-4 the asa
# probe selected, which is worse than asa's own 0.879. Inheriting the asa rate
# therefore reverses that cell's conclusion, and 144 of the 477 declared runs
# inherit it. Probing fa directly costs 25 runs.
# Exactly the backbones the declared matrix runs on an fa variant. Probing a
# grid-native SSM (mamba_nd, s4nd) on fa_3d would select a rate for cells that
# do not exist -- those two only ever run grid_*.
EXP0_MECH_CHECK_MODELS = tuple(AXIAL_MODELS + HYBRID_SSM)
# The two diagnostic mixers only ever run on the attention hosts (Exp 7/8 are
# gru/lstm/transformer), so their probes cover exactly those: 3 models x 5
# rates = 15 runs per arm.
EXP0_MECH_LOCAL_MODELS = tuple(AXIAL_MODELS)
EXP0_MECH_SM_MODELS = tuple(AXIAL_MODELS)

# DIVERGED_EXIT_CODE / DIVERGED_FILE are imported from src.utils above: the
# contract between train.py (which writes the record and exits with the code)
# and this scheduler (which counts it) must have exactly one definition.

# Test ids that are no longer runnable, and what to do instead. Kept as an
# explicit rejection rather than deleted from the CLI so an old command line
# gets an explanation instead of a silently smaller sweep.
RETIRED_TESTS = {
    "5": (
        "--tests 5 is retired: Exp 0 (--tests 0) replaces it. Test 5 searched "
        "four rates over one decade for the FLAT arm only, which is the exact "
        "asymmetry it was supposed to fix -- the N-D arm it was compared "
        "against never got a search at all. Exp 0 searches five rates over two "
        "decades in every selecting arm (flat/nd/agg/roll/mech), so the flat LR "
        "grid is now a strict subset of a symmetric one. The mamba3 stability finding that "
        "came out of Test 5 stands on the runs already collected and is "
        "reported from those; it is not re-trained. See future_work.md."
    ),
}


def lr_tag(lr):
    """Filename-safe tag for a learning rate, e.g. 1e-4 -> lr1e4."""
    return "lr" + str(lr).replace("-", "").replace(".", "")


def exp0_nd_variant(model, dims=EXP0_PROBE_DIM):
    """The N-D variant `model` actually runs at `dims` on the declared roster."""
    return f"{EXP0_ND_MECH.get(model, 'asa')}_{dims}d"
# An environment collapse -- the GPU detached from the container, a driver
# reset, a worker env without CUDA -- fails EVERY run, and fails it before any
# real work happens. A model-specific failure fails one cell after minutes of
# it. Time and consecutiveness are what separate the two, so the scheduler
# stops on a streak of instant failures instead of grinding through the matrix
# reporting them all at the end.
#
# This is not hypothetical: box A lost /dev/nvidia0 mid-session and a 27-run
# Exp 7 burned to run 24 in five minutes, every one failing at startup, before
# anyone looked. 90s is comfortably above data loading plus construction and
# far below a real epoch.
INSTANT_FAIL_SECONDS = 90
INSTANT_FAIL_ABORT = 3

DIMS = [2, 3, 4]
ENCODERS = ["embeddings", "onehot"]          # Test 3 / Test 2
DEFAULT_SEEDS = [947, 732, 619]
ROLLING_AGG_TEST_YEARS = tuple(range(2020, 2026))
PANEL_START_YEAR = 2010


DEFAULT_NPZ = "data/census_port/processed/census_lattice_9ch.npz"

# Named run slices from RETRAIN.md §4, as regexes over run names. These exist so
# "re-run exactly what finding X invalidated" is one flag rather than a
# hand-assembled list that drifts from the document.
RETRAIN_SETS = {
    # F1 step-matched protocol: every per-series (flat) run, LR bracket included.
    "f1": r"_1d_",
    # F2 Mamba-ND leftover encoder: only the dims where un-scanned axes exist.
    "f2": r"^mamba_nd_grid_[23]d",
    # F3 Mamba-3 head count: every Mamba-3 run (flat, hybrid, aggregate).
    "f3": r"^mamba3_",
    # Union of the finding-specific diagnostic slices; not a supported
    # substitute for the full retrain.
    "mandatory": r"(_1d_|^mamba3_|^mamba_nd_grid_[23]d)",
}


def axial_variant(mechanism, dims):
    return f"{mechanism}_{dims}d"


def rolling_aggregate_folds():
    """Annual expanding-origin folds for Test 1.1.

    One year is the validation/test window, not the training window. The first
    fold has five years of usable training targets after the 48-month feature
    burn-in: train through 2018, validate 2019, test 2020.
    """
    folds = []
    for test_year in ROLLING_AGG_TEST_YEARS:
        test_start = (test_year - PANEL_START_YEAR) * 12
        folds.append({
            "test_year": test_year,
            "train_end": test_start - 12,
            "val_end": test_start,
            "test_end": test_start + 12,
        })
    return folds


def lattice_n_series(npz_path):
    """Series count G from the lattice sidecar meta, or None if unavailable."""
    meta = Path(npz_path).with_suffix(".json")
    try:
        return int(json.load(open(meta))["n_series"])
    except Exception:
        return None


def train_target_months(data_cfg=None):
    """Number of training target months, i.e. the combo arm's samples/epoch.

    Targets start at ``input_len + lag_count`` (the first month with a full
    history AND full lags) and stop at ``train_end``, so the count follows the
    composed data config rather than a constant that silently goes stale when
    the window or the split boundary moves.
    """
    data = (data_cfg or compose_data_config()).get("data", {})
    burn = int(data.get("input_len", 36)) + int(data.get("lag_count", 12))
    return int(data.get("train_end", 144)) - burn


def flat_accum(n_series, flat_bs, combo_bs=1):
    """Micro-batches per optimizer step so the flat arm matches the combo arm.

    The two paths define a SAMPLE differently: the flat dataset emits one
    (series, month) pair, the combo dataset emits one month carrying all G
    series. Both therefore consume the same G x months target values per epoch,
    but at --flat-bs 512 / --combo-bs 1 the flat arm took ~55x more optimizer
    steps at a ~55x smaller effective batch, on the same lr. Any "structure
    doesn't help" conclusion then partly measures the grid models' much smaller
    update budget rather than their architecture.

    Grouping ceil(G / flat_bs) micro-batches and shortening the final one gives
    the flat arm an exact effective batch of G and the SAME number of optimizer
    steps per epoch as the combo arm, without materialising a G-sample batch. The combo arm is
    the one that cannot move: one sample is one month with every group present,
    and the axial/grid models need all groups at once. See RETRAIN.md. (F1)

    The combo arm's own batch scales its step count too: at --combo-bs C it
    takes 1/C as many steps per epoch, so the flat arm must accumulate C times
    as many micro-batches to stay matched. (F1b)
    """
    return max(1, math.ceil(n_series * max(combo_bs, 1) / max(flat_bs, 1)))


def build_exp0_matrix(seeds, flat_bs, combo_bs, accum, npz, effective_batch_size,
                      lrs=EXP0_LR_GRID, dim_models=EXP0_DIM_CHECK_MODELS,
                      only_models=None, only_arms=None):
    """Exp 0: the learning-rate probe, one cell per (arm, model, rate).

    Every probe reproduces the batch protocol of the arm it selects for -- the
    flat probes carry the same --grad-accum/--effective-batch-size step
    matching as the real flat runs, and the N-D probes run at --combo-bs. A
    rate chosen under a different batch protocol would not transfer, which is
    the mistake the transformer's inherited 1e-4 embodies.
    """
    runs = []
    base = [sys.executable, str(REPO / "scripts/train.py"),
            "--config", "config/base.yaml", "--data-config", "config/census.yaml",
            "--npz", str(npz)]
    flat_match = (["--effective-batch-size", str(effective_batch_size)]
                  if effective_batch_size else [])
    keep = (lambda m: True) if not only_models else (lambda m: m in only_models)
    arms_wanted = set(only_arms) if only_arms else None
    flat_models = [m for m in FLAT_MODELS if keep(m)]
    nd_models = [m for m in ND_BACKBONES if keep(m)]
    dim_models = [m for m in dim_models if keep(m)]
    roll_check_models = [m for m in EXP0_ROLL_CHECK_MODELS if keep(m)]
    enc_check_models = [m for m in EXP0_ENC_CHECK_MODELS if keep(m)]
    nd_enc_check_models = [m for m in EXP0_ND_ENC_CHECK_MODELS if keep(m)]
    id_check_models = [m for m in EXP0_ID_CHECK_MODELS if keep(m)]
    mech_check_models = [m for m in EXP0_MECH_CHECK_MODELS if keep(m)]
    mech_local_models = [m for m in EXP0_MECH_LOCAL_MODELS if keep(m)]
    mech_sm_models = [m for m in EXP0_MECH_SM_MODELS if keep(m)]

    # The rolling probes must reproduce the fold contract of the runs they
    # select for -- fold-local boundaries, a fold-local normalization refit and
    # a fresh model session -- or the rate would be chosen under a different
    # data contract than the one it is applied to.
    folds = rolling_aggregate_folds()
    shortest_fold, longest_fold = folds[0], folds[-1]

    def roll_args(fold):
        return ["--aggregate", "--refit-normalization", "--fresh-model-session",
                "--train-end", str(fold["train_end"]),
                "--val-end", str(fold["val_end"]),
                "--test-end", str(fold["test_end"])]
    for seed in seeds:
        for lr in lrs:
            tag = lr_tag(lr)
            for m in flat_models:                       # per-series flat arm
                runs.append((f"exp0_flat_{m}_{tag}_s{seed}",
                             base + ["--model", m, "--variant", "embeddings",
                                     "--lr", lr, "--batch-size", str(flat_bs),
                                     "--grad-accum", str(accum),
                                     *flat_match, "--seed", str(seed)]))
            for m in nd_models:                         # multidimensional arm
                runs.append((f"exp0_nd_{m}_{tag}_s{seed}",
                             base + ["--model", m,
                                     "--variant", exp0_nd_variant(m),
                                     "--combo-encoder", "embeddings", "--lr", lr,
                                     "--batch-size", str(combo_bs), "--seed", str(seed)]))
            for m in flat_models:                       # aggregate (Test 1) arm
                runs.append((f"exp0_agg_{m}_{tag}_s{seed}",
                             base + ["--model", m, "--variant", "embeddings",
                                     "--aggregate", "--lr", lr,
                                     "--batch-size", "32", "--seed", str(seed)]))
            for m in flat_models:                       # rolling (Test 1.1) arm
                runs.append((f"exp0_roll_{m}_{tag}_s{seed}",
                             base + ["--model", m, "--variant", "embeddings",
                                     *roll_args(shortest_fold), "--lr", lr,
                                     "--batch-size", "32", "--seed", str(seed)]))
            for m in roll_check_models:                 # fold-length transfer
                runs.append((f"exp0_rollend_{m}_{tag}_s{seed}",
                             base + ["--model", m, "--variant", "embeddings",
                                     *roll_args(longest_fold), "--lr", lr,
                                     "--batch-size", "32", "--seed", str(seed)]))
            for m in enc_check_models:                  # flat, one-hot (Test 2)
                runs.append((f"exp0_encflat_{m}_{tag}_s{seed}",
                             base + ["--model", m, "--variant", "onehot",
                                     "--lr", lr, "--batch-size", str(flat_bs),
                                     "--grad-accum", str(accum),
                                     *flat_match, "--seed", str(seed)]))
            for m in nd_enc_check_models:               # N-D, one-hot (Test 2)
                runs.append((f"exp0_encnd_{m}_{tag}_s{seed}",
                             base + ["--model", m,
                                     "--variant", exp0_nd_variant(m),
                                     "--combo-encoder", "onehot", "--lr", lr,
                                     "--batch-size", str(combo_bs),
                                     "--seed", str(seed)]))
            for m in id_check_models:                   # N-D, identity (Test 4)
                runs.append((f"exp0_axid_{m}_{tag}_s{seed}",
                             base + ["--model", m,
                                     "--variant", exp0_nd_variant(m),
                                     "--combo-encoder", "embeddings",
                                     "--axis-identity", "--lr", lr,
                                     "--batch-size", str(combo_bs),
                                     "--seed", str(seed)]))
            for m in mech_check_models:                 # N-D, fa mixer
                runs.append((f"exp0_mech_{m}_{tag}_s{seed}",
                             base + ["--model", m,
                                     "--variant", f"fa_{EXP0_PROBE_DIM}d",
                                     "--combo-encoder", "embeddings",
                                     "--lr", lr,
                                     "--batch-size", str(combo_bs),
                                     "--seed", str(seed)]))
            for m in mech_local_models:                 # N-D, fa_local mixer
                runs.append((f"exp0_mechlocal_{m}_{tag}_s{seed}",
                             base + ["--model", m,
                                     "--variant", f"fa_local_{EXP0_PROBE_DIM}d",
                                     "--combo-encoder", "embeddings",
                                     "--lr", lr,
                                     "--batch-size", str(combo_bs),
                                     "--seed", str(seed)]))
            for m in mech_sm_models:                    # N-D, fa softmax mixer
                runs.append((f"exp0_mechsm_{m}_{tag}_s{seed}",
                             base + ["--model", m,
                                     "--variant", f"fa_sm_{EXP0_PROBE_DIM}d",
                                     "--combo-encoder", "embeddings",
                                     "--lr", lr,
                                     "--batch-size", str(combo_bs),
                                     "--seed", str(seed)]))
            for m in dim_models:                        # Phase 0 transfer check
                for d in EXP0_DIM_CHECK_DIMS:
                    runs.append((f"exp0_dim{d}_{m}_{tag}_s{seed}",
                                 base + ["--model", m,
                                         "--variant", exp0_nd_variant(m, d),
                                         "--combo-encoder", "embeddings", "--lr", lr,
                                         "--batch-size", str(combo_bs),
                                         "--seed", str(seed)]))
    if arms_wanted:
        # Execution filter only: prepare_runs still declares the whole matrix,
        # so a supplementary probe in its own --runs-dir cannot shrink another
        # session's manifest.
        runs = [(n, a) for n, a in runs
                if re.match(r"^exp0_([a-z0-9]+)_", n).group(1) in arms_wanted]
    return runs


def nd_rate_arm(variant):
    """Which selection arm owns the rate for an N-D variant.

    Most specific prefix first, and that ordering is load-bearing. ``fa_local``
    and ``fa_sm`` both start with ``fa_``, so the single-prefix test this
    replaced silently handed both of them the rate probed on the AUTHORS'
    LeakyReLU ``fa_3d``. Every diagnostic mixer would then have been compared
    against a tuned arm while running on someone else's rate -- the asymmetry
    F5 removed, reintroduced through a string prefix.

    The arms are far enough apart for that to matter: nd vs mech is 3e-4 vs
    1e-2 on gru (33x) and reverses direction on transformer (1e-3 vs 3e-4).
    """
    v = str(variant)
    if v.startswith("fa_local_"):
        return "mechlocal"
    if v.startswith("fa_sm_"):
        return "mechsm"
    if v.startswith("fa_"):
        return "mech"
    return "nd"


def build_mixer_arm_matrix(mixer, seeds, combo_bs, npz, lr_selection=None,
                           models=None, dims=None, enc="embeddings"):
    """One diagnostic mixer, mirrored onto the fa_* cells of Test 3.

    Shape is fixed by the comparator, not chosen: the same hosts, dims,
    encoder, seeds, batch protocol and epoch cap as the ``fa_{d}d`` cells, so
    the only thing that moves is the grid-mixing operator. Each mixer trains at
    its OWN Exp 0 arm (see nd_rate_arm), because comparing a tuned arm against
    an untuned one is what F5 exists to prevent.

    Neither arm is part of the declared 477. Both run alone, in their own
    --runs-dir, and must not enter the main leaderboard completeness check.
    """
    models = list(AXIAL_MODELS if models is None else models)
    dims = list(DIMS if dims is None else dims)
    runs = []
    base = [sys.executable, str(REPO / "scripts/train.py"),
            "--config", "config/base.yaml", "--data-config", "config/census.yaml",
            "--npz", str(npz)]
    for seed in seeds:
        for m, d in itertools.product(models, dims):
            variant = axial_variant(mixer, d)
            runs.append((f"{m}_{variant}_{enc}_s{seed}",
                         base + ["--model", m, "--variant", variant,
                                 "--combo-encoder", enc,
                                 "--batch-size", str(combo_bs),
                                 *_lr_args(lr_selection, nd_rate_arm(variant), m),
                                 "--seed", str(seed)]))
    return runs


def build_exp7_matrix(seeds, combo_bs, npz, lr_selection=None, **kw):
    """Exp 7: does OUR OWN CaFA build still beat axial under this protocol?

    This is the load-bearing replication, and it comes before Exp 8. The
    submitted draft's CaFA advantage -- CaFA winning 12/12 on GRU and 12/12 on
    LSTM -- was produced by ``fa_local_*`` (the workbook spells it ``cafa_*``),
    our own build of the FA operator from the paper. The rerun never ran that
    module: it runs ``fa_*``, the authors' released components, which lose
    0/12 and 1/12 on the same hosts. So the reversal in the paper confounds two
    things at once, a change of OPERATOR and a change of PROTOCOL, and nothing
    published so far separates them.

    Running fa_local under the current cohort, normalisation, step matching and
    epoch cap separates them directly:

    * fa_local still beats asa -> the draft's finding survives; what changed is
      that the authors' operator behaves differently at this scale, and the
      paper says exactly that.
    * fa_local also loses -> the advantage was the old cohort/protocol, the
      operator question is moot, and Exp 8 is answering something nobody needs.

    That second branch is why this runs FIRST. Exp 8 decomposes a difference
    that Exp 7 has to establish exists.
    """
    return build_mixer_arm_matrix("fa_local", seeds, combo_bs, npz,
                                  lr_selection=lr_selection, **kw)


def build_exp8_matrix(seeds, combo_bs, npz, lr_selection=None, **kw):
    """Exp 8: if fa_local and fa really differ, is it the kernel NONLINEARITY?

    Only meaningful once Exp 7 has shown the two operators disagree under one
    protocol. They differ four ways at once (scripts/sweep.py OPTIONAL_MECHS),
    and this arm moves exactly one of them: the authors' own
    ``LowRankKernel(softmax=True)`` switch, with PoolingReducer, head geometry,
    the channel mixer and Q/K RMS norm pinned at the fa_* values. fa_sm and fa
    have identical parameter counts at every lattice size and byte-identical
    initial weights under the same seed, so the pair is as tightly matched as a
    pair can be.

    NOT a config-only flip: fa.py applies the LeakyReLU gate outside
    LowRankKernel, divides by a uniform quadrature count, and upstream retunes
    the kernel temperature when softmax is on. See FactorizedAttention.
    """
    return build_mixer_arm_matrix("fa_sm", seeds, combo_bs, npz,
                                  lr_selection=lr_selection, **kw)


def _lr_args(lr_selection, arm, model):
    """``--lr`` for one cell of the main matrix, from the Exp 0 selection.

    Fails closed: if a selection file is supplied but does not cover a cell,
    that cell must not quietly fall back to a config default, because the
    resulting leaderboard would mix searched and unsearched rates in the same
    table without saying so.
    """
    if not lr_selection:
        return []
    selected = lr_selection.get("selected", {})
    # An fa cell used to fall back silently to the nd (asa-probed) rate when
    # the selection carried no mech arm. The mech arm exists precisely because
    # that inheritance reverses a measured cell (mamba3 at 3-D: fa reaches
    # 0.634 at its own 1e-3 but 1.024 at the asa probe's 1e-4), so a selection
    # without it is refused like any other uncovered cell rather than quietly
    # downgraded.
    if arm not in selected or model not in selected[arm]:
        # Every mixer arm is added the same way -- its own probe session, then
        # merged in -- so every one of them gets the recipe. Naming only "mech"
        # left the operator of a fa_local or fa_sm sweep with a refusal and no
        # way out of it, which is the same dead end with a different label.
        hint = (
            f"; this selection predates the {arm} probe -- run it in its "
            f"own session (--tests 0 --exp0-arms {arm} --runs-dir <dir>) and "
            "fold it in with --lr-selection-patch"
            if arm in ("mech", "mechlocal", "mechsm") else ""
        )
        sys.exit(
            f"lr-selection covers no {arm}/{model} cell; re-run Exp 0 for it "
            "or drop --lr-selection to run on unsearched config defaults" + hint
        )
    return ["--lr", str(selected[arm][model])]


def build_matrix(tests, seeds, flat_bs, combo_bs, accum=1,
                 npz=DEFAULT_NPZ, effective_batch_size=None, mechs=None,
                 lr_selection=None,
                 exp0_models=None, exp0_arms=None, hybrid_identity=False):
    """Yield (name, argv) runs. Test 2 = onehot N-d, Test 3 = embeds N-d
    (both include the 1-D flat baseline under the matching encoder);
    Test 4 = the same axial N-d grid with --axis-identity, BOTH encoders.
    Test 5 is retired -- see RETIRED_TESTS. hybrid_identity is the opt-in
    identity-aware extension of Test 6's hybrid grid (see future_work.md)."""
    if "0" in tests:
        if tests - {"0"}:
            sys.exit("--tests 0 (LR selection) must be run on its own: it gates "
                     "the other tests and belongs in its own --runs-dir")
        return build_exp0_matrix(seeds, flat_bs, combo_bs, accum, npz,
                                 effective_batch_size, only_models=exp0_models,
                                 only_arms=exp0_arms)
    # Same rule as Exp 0, for the same reason: these arms are diagnostic and are
    # NOT among the declared 477, so sharing a --runs-dir would leave a manifest
    # that reads as a main sweep with 27 extra cells and no complete seed cohort
    # for them. They are also mutually exclusive: Exp 8 only means something
    # once Exp 7 has shown the two operators disagree, so running both in one
    # session would spend GPU time on a question that may not exist.
    for tid, builder, label in (("7", build_exp7_matrix, "the fa_local replication"),
                                ("8", build_exp8_matrix, "the FA kernel-nonlinearity arm")):
        if tid in tests:
            if tests - {tid}:
                sys.exit(f"--tests {tid} (Exp {tid}, {label}) must be run on its "
                         "own, in its own --runs-dir: it is diagnostic and not "
                         "part of the declared matrix")
            return builder(seeds, combo_bs, npz, lr_selection=lr_selection)
    retired = tests & set(RETIRED_TESTS)
    if retired:
        # Refuse rather than return an empty matrix: a silently-skipped test id
        # would look like a completed sweep and quietly shrink the seed matrix
        # that evaluate.py checks for completeness.
        sys.exit("; ".join(RETIRED_TESTS[t] for t in sorted(retired)))
    mechs = list(MECHS if mechs is None else mechs)
    runs = []
    base = [sys.executable, str(REPO / "scripts/train.py"),
            "--config", "config/base.yaml", "--data-config", "config/census.yaml",
            "--npz", str(npz)]
    flat_match = (["--effective-batch-size", str(effective_batch_size)]
                  if effective_batch_size else [])
    want_oh, want_emb = "2" in tests, "3" in tests
    want_agg = "1" in tests
    want_roll = "1.1" in tests
    want_id = "4" in tests
    want_hybrid = "6" in tests
    for seed in seeds:
        # Test 1: aggregate — collapse ALL series into one national series; every
        # model, no categorical encoder (num_states/comm/flow=1 -> trivial embeds).
        if want_agg:
            for m in FLAT_MODELS:
                runs.append((f"{m}_aggregate_s{seed}",
                             base + ["--model", m, "--variant", "embeddings", "--aggregate",
                                     "--batch-size", "32",
                                     *_lr_args(lr_selection, "agg", m),
                                     "--seed", str(seed)]))
        # Test 1.1: annual expanding-origin aggregate backtest. Every fold gets
        # a fold-specific train-only transform and exactly 12 validation/test
        # target months. Models start from scratch; no checkpoint is warm-started.
        if want_roll:
            for fold in rolling_aggregate_folds():
                year = fold["test_year"]
                for m in FLAT_MODELS:
                    runs.append((f"{m}_aggregate_roll_y{year}_s{seed}",
                                 base + ["--model", m, "--variant", "embeddings",
                                         "--aggregate", "--refit-normalization",
                                         "--fresh-model-session",
                                         "--train-end", str(fold["train_end"]),
                                         "--val-end", str(fold["val_end"]),
                                         "--test-end", str(fold["test_end"]),
                                         "--batch-size", "32",
                                         *_lr_args(lr_selection, "roll", m),
                                         "--seed", str(seed)]))
        # 1-D flat baselines (the encoder IS the variant here)
        if want_oh:
            for m in FLAT_MODELS:
                runs.append((f"{m}_onehot_1d_s{seed}",
                             base + ["--model", m, "--variant", "onehot",
                                     "--batch-size", str(flat_bs),
                                     "--grad-accum", str(accum), *flat_match,
                                     *_lr_args(lr_selection, "flat", m),
                                     "--seed", str(seed)]))
        if want_emb:
            for m in FLAT_MODELS:
                runs.append((f"{m}_embeddings_1d_s{seed}",
                             base + ["--model", m, "--variant", "embeddings",
                                     "--batch-size", str(flat_bs),
                                     "--grad-accum", str(accum), *flat_match,
                                     *_lr_args(lr_selection, "flat", m),
                                     "--seed", str(seed)]))
        # N-D axial / CaFA, per combo-encoder
        for enc in ([e for e, w in (("onehot", want_oh), ("embeddings", want_emb)) if w]):
            for m, mech, d in itertools.product(AXIAL_MODELS, mechs, DIMS):
                variant = axial_variant(mech, d)
                runs.append((f"{m}_{variant}_{enc}_s{seed}",
                             base + ["--model", m, "--variant", variant,
                                     "--combo-encoder", enc,
                                     "--batch-size", str(combo_bs),
                                     *_lr_args(lr_selection, nd_rate_arm(variant), m),
                                     "--seed", str(seed)]))
        # Multidim SSM: Mamba-ND scans the (State,Commodity,Flow) lattice x time.
        # Cross-attention axes only (no CaFA); the leftover encoder is a no-op
        # (identity is positional on the lattice), so run once/dim under the
        # embeddings tag so it joins the embeds N-D comparison.
        if want_emb:
            for m, d in itertools.product(AXIAL_SSM, DIMS):
                runs.append((f"{m}_grid_{d}d_embeddings_s{seed}",
                             base + ["--model", m, "--variant", f"grid_{d}d",
                                     "--combo-encoder", "embeddings",
                                     "--batch-size", str(combo_bs),
                                     # f"grid_{d}d", not the loop-leaked
                                     # `variant` from the axial block above --
                                     # that held fa_4d here and silently gave
                                     # the grid-native SSMs the fa rate.
                                     *_lr_args(lr_selection,
                                               nd_rate_arm(f"grid_{d}d"), m),
                                     "--seed", str(seed)]))
        # Test 4: the SAME axial/FA N-d grid but identity-AWARE
        # (--axis-identity: learned embeddings on the promoted axes), both
        # encoders in one test. A/B against Tests 2/3 isolates the
        # permutation-equivariance confound (flat baselines always see
        # identity; identity-blind axial did not). mamba_nd excluded — its
        # ordered scan is already position-aware.
        if want_id:
            for enc in ENCODERS:
                for m, mech, d in itertools.product(AXIAL_MODELS, mechs, DIMS):
                    variant = axial_variant(mech, d)
                    runs.append((f"{m}_{variant}_id_{enc}_s{seed}",
                                 base + ["--model", m, "--variant", variant,
                                         "--combo-encoder", enc, "--axis-identity",
                                         "--batch-size", str(combo_bs),
                                         *_lr_args(lr_selection, nd_rate_arm(variant), m),
                                         "--seed", str(seed)]))
        # Test 6: SSM-structure completion arm (roster extension 2026-07-14).
        # (a) mamba2/mamba3 x axial-SA/FA hybrids — the EXACT grid-mixing
        #     modules of the gru/lstm/transformer combos with only the temporal
        #     operator swapped to the Mamba stack (tier-1 same-harness cells).
        #     Base/torch-2.13 env (Triton mamba kernels).
        # (b) s4nd — genuine S4ND (separable per-axis DPLR LTI kernels =
        #     outer-product N-D conv); empirically tests the translation-
        #     equivariance argument. cross_attention_{d}d slots reused as grid
        #     tags (cf. mamba_nd); no factorized-attention arm (natively
        #     factorized, and it has no attention at all). t28 env.
        if want_hybrid:
            for enc in ENCODERS:
                for m, mech, d in itertools.product(HYBRID_SSM, mechs, DIMS):
                    variant = axial_variant(mech, d)
                    runs.append((f"{m}_{variant}_{enc}_s{seed}",
                                 base + ["--model", m, "--variant", variant,
                                         "--combo-encoder", enc,
                                         "--batch-size", str(combo_bs),
                                         *_lr_args(lr_selection, nd_rate_arm(variant), m),
                                         "--seed", str(seed)]))
                for d in DIMS:
                    runs.append((f"s4nd_grid_{d}d_{enc}_s{seed}",
                                 base + ["--model", "s4nd",
                                         "--variant", f"grid_{d}d",
                                         "--combo-encoder", enc,
                                         "--batch-size", str(combo_bs),
                                         *_lr_args(lr_selection, nd_rate_arm(f"grid_{d}d"), "s4nd"),
                                         "--seed", str(seed)]))
        # OPT-IN (off the declared matrix): Test 6's hybrid grid with
        # --axis-identity, mirroring Test 4 for the SSM-hybrid hosts. The
        # declared benchmark isolates the identity confound on the attention
        # hosts only, so if Test 4 shows identity matters, the hybrid
        # conclusions inherit it until this arm runs (+72 runs; see
        # future_work.md). s4nd/mamba_nd stay excluded: an LTI kernel or an
        # ordered scan along an axis is already position-aware.
        if want_hybrid and hybrid_identity:
            for enc in ENCODERS:
                for m, mech, d in itertools.product(HYBRID_SSM, mechs, DIMS):
                    variant = axial_variant(mech, d)
                    runs.append((f"{m}_{variant}_id_{enc}_s{seed}",
                                 base + ["--model", m, "--variant", variant,
                                         "--combo-encoder", enc, "--axis-identity",
                                         "--batch-size", str(combo_bs),
                                         *_lr_args(lr_selection, nd_rate_arm(variant), m),
                                         "--seed", str(seed)]))
    return runs


STOP_FILE = "STOP"          # created inside the selected runs/session directory


def prepare_runs(matrix, epochs, runs_dir):
    """Attach output paths and provenance fingerprints to a run matrix."""
    prepared = []
    for name, argv in matrix:
        argv = argv + ["--epochs", str(epochs),
                       "--out-dir", str(runs_dir / name)]
        entry = {
            "name": name,
            "argv": argv,
            "fingerprint": run_fingerprint(argv, REPO),
            "input_fingerprint": run_input_fingerprint(argv, REPO),
        }
        prepared.append((name, argv, entry))
    return prepared


def _arm_of(run_name):
    """`exp0_mech_gru_lr1e3_s947` -> `mech`."""
    parts = str(run_name).split("_")
    return parts[1] if len(parts) > 1 and parts[0] == "exp0" else None


def _refuse_arm_probe_over_existing_session(arms_spec, runs_dir):
    """Stop --exp0-arms from eating a manifest that holds other arms.

    Narrowing the declaration is what lets a probe emit a partial selection,
    but it also means the manifest written here covers ONLY these arms. Run
    that over a finished Exp 0 and the other arms' declarations are gone --
    with them the record of what that session was, which no amount of
    surviving run directories reconstructs.
    """
    manifest = Path(runs_dir) / "manifest.json"
    if not manifest.exists():
        return
    try:
        declared = json.loads(manifest.read_text()).get("runs", [])
    except (OSError, ValueError):
        return
    wanted = {a for a in arms_spec.split(",") if a}
    present = {arm for arm in (_arm_of(e.get("name")) for e in declared) if arm}
    stranded = present - wanted
    if stranded:
        sys.exit(
            f"--exp0-arms {arms_spec} would rewrite {manifest} down to "
            f"{sorted(wanted)}, stranding {len(stranded)} other declared "
            f"arm(s): {', '.join(sorted(stranded))}.\n"
            "A supplementary probe belongs in its own --runs-dir; fold its "
            "result in with --lr-selection-patch."
        )


def _protocol_argv(entry):
    """Command identity independent of the Python environment executable.

    Split-environment sweeps legitimately use different Python executables for
    different model families. The train script and every argument after it are
    experiment protocol and must remain identical before an old declaration can
    be retained for a cell this machine is not executing.
    """
    argv = list(entry.get("argv", []))
    return argv[1:] if argv else argv


def _is_done(entry, runs_dir):
    """Only a checkpoint bound to this exact declaration counts as finished."""
    return validate_completion(runs_dir, entry) is None


# Failure markers, anchored to the START of a log line:
#   - the Python traceback header,
#   - this scheduler's own audit marker,
#   - the unindented exception line that terminates a traceback
#     ("ValueError: ...", "torch.cuda.OutOfMemoryError: ...").
_FAILED_LOG_RE = re.compile(
    r"^(?:Traceback\b"
    r"|SWEEP-FAIL:"
    r"|[\w.]*(?:Error|Exception)(?::|$))",
    re.MULTILINE)


def _log_reports_failure(text):
    """True only for a real failure signal, not an exception name in prose.

    An unanchored ``"Error" in text`` also matches benign output. Without
    Triton installed, ``src/models/__init__.py`` warns
    ``UserWarning: Mamba-2 unavailable (ModuleNotFoundError(...))`` on EVERY
    run, so in the non-Mamba half of the split-environment sweep RETRAIN.md
    describes, every cell that was not already ``done`` reported ``failed``,
    including runs that finished cleanly and whose declaration had merely
    moved. Matching at line starts keeps tracebacks and exception lines while
    ignoring exception names quoted inside a warning, a file path, or progress
    output. Carriage returns are normalised to newlines so a traceback printed
    straight after a progress-bar line is still seen at a line start.
    """
    return _FAILED_LOG_RE.search(text.replace("\r", "\n")) is not None


def run_state(entry, runs_dir, logdir):
    """Return done, stale, partial, failed, or todo for ``--status``.

    The provenance fingerprint covers the command, exact batch protocol, data,
    configuration, source tree, submodules, and training runtime.
    """
    name = entry["name"]
    if _is_done(entry, runs_dir):
        return "done"
    lg = logdir / f"{name}.log"
    checkpoint = runs_dir / name / "best.pth"
    marker = completion_path(runs_dir, name)
    if not lg.exists() and not checkpoint.exists() and not marker.exists():
        return "todo"
    try:
        text = lg.read_text(errors="ignore") if lg.exists() else ""
    except OSError:
        text = ""
    if _log_reports_failure(text):
        return "failed"
    if "Final val loss" in text:
        return "stale" if checkpoint.exists() else "partial"
    return "stale" if checkpoint.exists() or marker.exists() else "partial"


def select(runs, only=None, exclude=None):
    """Filter runs by REGEX over the run name.

    Run names carry model, variant, encoder and seed, so a regex selects the
    slices RETRAIN.md talks about without needing a flag per slice:
      --only '_1d_'                      every flat run
      --only '^mamba3_'                  every Mamba-3 run (flat + hybrid + agg)
      --only '^mamba_nd_grid_[23]d'      the two affected Mamba-ND dims
    """
    out = runs
    if only:
        rx = re.compile(only)
        out = [(n, a) for n, a in out if rx.search(n)]
    if exclude:
        rx = re.compile(exclude)
        out = [(n, a) for n, a in out if not rx.search(n)]
    return out


def resolve_gpus(device_count=None):
    """The GPU pool when --gpus was not given.

    Returns the visible CUDA devices, and REFUSES when there are none. The old
    ``list(range(count)) or [0]`` pinned CUDA_VISIBLE_DEVICES to a device that
    did not exist whenever the count was zero, so every run failed at startup
    and the sweep only said so 27 runs later. There is no sensible default for
    "no GPU": either the card is gone or this interpreter has a CPU-only torch,
    and both deserve to be said out loud rather than guessed around.
    """
    if device_count is None:
        import torch
        device_count = torch.cuda.device_count()
    if device_count == 0:
        sys.exit(
            "no CUDA device visible: torch.cuda.device_count() == 0. Either "
            "the GPU is detached (check nvidia-smi and ls /dev/nvidia0) or "
            "this interpreter has a CPU-only torch. Pass --gpus explicitly to "
            "override."
        )
    return list(range(device_count))


def schedule(runs, gpus, epochs, logdir, dry, resume=True, reset_manifest=False,
             declared_runs=None, stop_file=None, worker_env=None):
    logdir.mkdir(parents=True, exist_ok=True)
    runs_dir = logdir.parent

    # Execution can be filtered by --models for split CUDA environments, while
    # the declaration remains the complete selected experiment matrix.
    full = prepare_runs(runs, epochs, runs_dir)
    declaration_full = prepare_runs(
        declared_runs if declared_runs is not None else runs, epochs, runs_dir
    )
    # Freeze the declared run matrix before resume filtering. Evaluation uses
    # this manifest to reject an incomplete official leaderboard.
    manifest_path = logdir.parent / "manifest.json"
    current_declared = {name: entry for name, _, entry in declaration_full}
    declared = dict(current_declared)
    executing = {name: entry for name, _, entry in full}
    if manifest_path.exists() and not reset_manifest:
        try:
            old = json.loads(manifest_path.read_text()).get("runs", [])
            old_declared = {entry["name"]: entry for entry in old}
            # Preserve provenance from a different interpreter/environment for
            # compatible cells this invocation is not executing. Never retain
            # removed cells or entries from a different epoch/data/batch
            # protocol: doing so can create an apparently complete leaderboard
            # whose models were trained under different experiments.
            for name, current in current_declared.items():
                previous = old_declared.get(name)
                if (name not in executing and previous is not None
                        and _protocol_argv(previous) == _protocol_argv(current)):
                    declared[name] = previous
            declared.update(executing)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    manifest = {"schema_version": 3, "epochs": epochs,
                "runs": list(declared.values())}
    if not dry:
        manifest_tmp = manifest_path.with_suffix(".tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2))
        manifest_tmp.replace(manifest_path)
    # A manifest reset declares a fresh benchmark and must not silently reuse
    # completion records from an older run, even when names are unchanged.
    if not dry and reset_manifest:
        for name, _, _ in declaration_full:
            completion_path(runs_dir, name).unlink(missing_ok=True)
    elif not dry and not resume:
        for name, _, _ in full:
            completion_path(runs_dir, name).unlink(missing_ok=True)
    resume = resume and not reset_manifest
    if resume:
        n0 = len(full)
        stale = [n for n, _, e in full
                 if (validate_completion(runs_dir, e) or "").endswith(
                     "completion fingerprint does not match manifest")]
        full = [(n, a, e) for n, a, e in full if not _is_done(e, runs_dir)]
        if n0 - len(full):
            print(f"resume: skipping {n0 - len(full)} finished runs "
                  f"(--force to re-run all)")
        if stale:
            # The provenance fingerprint covers every .py/.yaml under src/,
            # scripts/, config/ and external/, plus the training runtime. So an
            # unrelated source edit mid-sweep invalidates finished checkpoints
            # and they silently re-train. Say so instead of just re-running.
            print(f"resume: WARNING {len(stale)} run(s) have a checkpoint whose "
                  "provenance no longer matches this declaration and will be "
                  "RETRAINED, e.g. " + ", ".join(sorted(stale)[:3]))
            print("resume: this is what a changed command, data, config, source "
                  "tree, submodule or training runtime looks like. Freeze the "
                  "tree for the duration of a sweep; `--status` lists every "
                  "stale cell.")
    print(f"{len(full)} runs to go across GPUs {gpus} | epochs={epochs} | logs -> {logdir}")
    if dry:
        for name, argv, _ in full:
            print("  ", name, "::", " ".join(a for a in argv if "/" not in a or a.endswith(".yaml")))
        return
    queue, procs, free = list(full), {}, list(gpus)
    done, failed, diverged, total = [], [], [], len(full)
    stopped = False
    instant_streak, aborted = 0, False
    while queue or procs:
        # Graceful stop: create the session-specific STOP file and the scheduler stops
        # LAUNCHING new runs while letting in-flight ones finish, so no partial
        # checkpoint is left behind and --resume picks up exactly where this
        # left off. Ctrl-C would kill mid-epoch runs instead.
        if stop_file is not None and stop_file.exists() and not stopped:
            stopped = True
            print(f"\nSTOP flag seen ({stop_file}): draining {len(procs)} "
                  f"in-flight run(s), {len(queue)} still queued. "
                  f"Delete the file and re-run to continue.")
            queue = []
        while free and queue:
            gpu = free.pop(0); name, argv, entry = queue.pop(0)
            # Loader workers ride in the environment, alongside the GPU pin,
            # rather than in argv: worker count does not change what is trained
            # (verified; see the note in train.py), so it must not enter
            # run_fingerprint and invalidate finished runs when it is retuned.
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   **(worker_env or {})}
            marker = completion_path(runs_dir, name)
            marker.unlink(missing_ok=True)
            # The divergence sidecar carries no fingerprint, so select_lr.py
            # trusts it by pathname alone. Left behind, a marker from an
            # earlier declaration would keep classifying this cell as diverged
            # even after it trains successfully -- silently removing its
            # validation curve from the ranking and possibly changing the
            # selected rate. It is regenerated by the run itself if it
            # diverges again.
            (runs_dir / name / DIVERGED_FILE).unlink(missing_ok=True)
            checkpoint = runs_dir / name / "best.pth"
            old_checkpoint = ((checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
                              if checkpoint.exists() else None)
            lf = open(logdir / f"{name}.log", "w")
            p = subprocess.Popen(argv, cwd=str(REPO), env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
            procs[gpu] = (name, p, lf, entry, old_checkpoint,
                          time.monotonic())
            print(f"[gpu{gpu}] START {name}")
        time.sleep(3)
        for gpu, (name, p, lf, entry, old_checkpoint, started) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            lf.close(); del procs[gpu]; free.append(gpu)
            checkpoint = runs_dir / name / "best.pth"
            new_checkpoint = ((checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
                              if checkpoint.exists() else None)
            if rc == 0 and (new_checkpoint is None or new_checkpoint == old_checkpoint):
                rc = 2
                with open(logdir / f"{name}.log", "a") as audit_log:
                    audit_log.write("\nSWEEP-FAIL: process produced no new best.pth\n")
            if rc == 0:
                try:
                    write_completion(runs_dir, entry)
                except (OSError, ValueError, TypeError) as exc:
                    rc = 3
                    with open(logdir / f"{name}.log", "a") as audit_log:
                        audit_log.write(f"\nSWEEP-FAIL: cannot write completion record: {exc}\n")
            # A diverged run is an ANSWER, not a failure: train.py exits with
            # DIVERGED_EXIT_CODE after recording run_diverged.json. Exp 0 probes
            # a deliberately-too-high rate expecting exactly this, so counting
            # it as a failure would make a correct Exp 0 sweep exit non-zero and
            # stop hpec_pipeline.sh before selection ever runs.
            if rc == DIVERGED_EXIT_CODE:
                diverged.append(name); tag = "DIVERGED"
            elif rc == 0:
                done.append(name); tag = "OK "
            else:
                failed.append(name); tag = f"FAIL(rc={rc})"
            finished = len(done) + len(failed) + len(diverged)
            elapsed = time.monotonic() - started
            print(f"[gpu{gpu}] {tag} {name}   [{finished}/{total}]"
                  + (f"  ({elapsed:.0f}s)" if rc not in (0, DIVERGED_EXIT_CODE)
                     else ""))
            # Only a FAILURE that arrives before any real work counts toward
            # the streak. A success, or a failure that took real time, means
            # the environment is intact and resets it.
            if (rc not in (0, DIVERGED_EXIT_CODE)
                    and elapsed < INSTANT_FAIL_SECONDS):
                instant_streak += 1
            else:
                instant_streak = 0
            if instant_streak >= INSTANT_FAIL_ABORT and not aborted:
                aborted = True
                print(f"\nABORTING: {instant_streak} consecutive runs failed in "
                      f"under {INSTANT_FAIL_SECONDS}s. That is an environment "
                      "failure, not a model one -- check that the GPU is still "
                      "attached (nvidia-smi, ls /dev/nvidia0), that the worker "
                      "environment has CUDA, and that the data files are "
                      "readable. Nothing further is launched; in-flight runs "
                      "are left to finish. Re-run to resume where this stopped.")
                queue = []
    print(f"\nDONE: {len(done)} ok, {len(failed)} failed, "
          f"{len(diverged)} diverged."
          + (" (STOPPED early by flag file)" if stopped else ""))
    if diverged:
        # Reported, never hidden, and never fatal on its own. These cells have
        # no checkpoint and no completion record, so evaluate.py's seed-matrix
        # gate still refuses to score them as results.
        print(f"DIVERGED ({len(diverged)}): " + ", ".join(sorted(diverged)))
        print(f"(each wrote {DIVERGED_FILE}; in Exp 0 this is a measurement, "
              "in the main matrix it is a cell that must be re-run at a "
              "selected rate)")
    if failed:
        print("FAILED:", ", ".join(failed), f"\n(see {logdir}/<name>.log)")
        sys.exit(1)
    if stopped:
        # Do not let hpec_pipeline.sh mark training complete or advance to
        # evaluation. All in-flight processes have still closed cleanly.
        sys.exit(130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=None, help="comma list, e.g. 0,1 (default: all)")
    ap.add_argument("--tests", default="2,3",
                    help="which tests: 0=LR selection probe (Exp 0; run alone, "
                         "gates the rest), 1=aggregate, 1.1=annual rolling aggregate, "
                         "2=onehot N-d, 3=embeds N-d, "
                         "4=identity-aware axial (both encoders), "
                         "6=SSM-structure completion (mamba2/mamba3 axial+cafa+fa "
                         "hybrids + genuine s4nd), both encoders, "
                         "7=Exp 7, the fa_local replication (OUR build of the "
                         "FA operator, the one the draft's CaFA numbers came "
                         "from, rerun under the current protocol), "
                         "8=Exp 8, the FA kernel-nonlinearity arm (fa_sm_*; "
                         "only meaningful after Exp 7). Both mirror the fa_* "
                         "cells of Test 3, run alone, 27 runs each, diagnostic "
                         "and NOT part of the declared 477")
    ap.add_argument("--lr-selection", default=None,
                    help="lr_selection.json from scripts/select_lr.py. Every run "
                         "then carries the rate selected for its (arm, model) "
                         "cell. Without it the sweep runs on unsearched config "
                         "defaults, which is the asymmetry Exp 0 exists to fix.")
    ap.add_argument("--lr-selection-patch", default=None,
                    help="a second lr_selection.json whose arms are merged OVER "
                         "--lr-selection. Additive and auditable: the base file "
                         "is never rewritten, and what the patch changed is "
                         "recorded in the manifest. Use it to fold in a "
                         "supplementary probe without disturbing the session "
                         "that produced the base.")
    ap.add_argument("--exp0-arms", default=None,
                    help="comma list of Exp 0 arms to run (e.g. 'mech'). Unlike "
                         "--exp0-models this narrows the DECLARATION as well, "
                         "and it has to: select_lr.py writes nothing while a "
                         "declared cell is missing, so a probe that declared "
                         "all 210 cells and ran 25 could never emit the "
                         "selection it exists to produce. The cost is that "
                         "pointed at a finished session it would rewrite that "
                         "manifest down to these arms -- so give it its OWN "
                         "--runs-dir and fold the result in with "
                         "--lr-selection-patch.")
    ap.add_argument("--exp0-models", default=None,
                    help="restrict which models Exp 0 EXECUTES (comma list). For "
                         "split environments -- a box without Triton cannot run "
                         "the mamba cells at all. The declaration stays complete, "
                         "so each half of a split run adds to one manifest "
                         "instead of replacing the other half's cells.")
    ap.add_argument("--seeds", default=None, help="comma list (default 947,732,619)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--flat-bs", type=int, default=2048,
                    help="maximum flat MICRO-batch. Under --step-match the final "
                         "micro-batch in each exact group may be smaller; pick the largest "
                         "value that fits, it only affects memory and speed.")
    ap.add_argument("--step-match", dest="step_match", action="store_true", default=True,
                    help="give the flat arm the same effective batch and "
                         "optimizer-step count as the combo arm (default on)")
    ap.add_argument("--no-step-match", dest="step_match", action="store_false",
                    help="legacy behaviour: flat arm runs at --flat-bs with "
                         "grad_accum=1 and ~55x more optimizer steps than the "
                         "combo arm. To reproduce pre-RETRAIN.md runs also pass "
                         "--flat-bs 512 (the old default; it is 2048 now).")
    ap.add_argument("--npz", default=DEFAULT_NPZ,
                    help="lattice .npz; its sidecar .json supplies the series "
                         "count used to compute the step-matched accumulation")
    ap.add_argument("--series", type=int, default=None,
                    help="override the series count G (when the sidecar meta "
                         "is unavailable)")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader workers for the FLAT path (default 4). It "
                         "calls __getitem__ G x 96 times per epoch, so this is "
                         "where parallelism pays. Passed by environment, not "
                         "argv: worker count provably does not affect what is "
                         "trained, so retuning it must not invalidate finished "
                         "runs by changing their fingerprint.")
    ap.add_argument("--combo-num-workers", type=int, default=None,
                    help="DataLoader workers for the COMBO path (default 4). "
                         "The combo loader emits only ~96 batches per epoch and "
                         "its __getitem__ is a contiguous slice, so 1-2 is "
                         "usually enough -- leave the cores to the flat arm. "
                         "The sweep runs one job per GPU, so total worker "
                         "processes is gpus x workers.")
    ap.add_argument("--hybrid-identity", action="store_true",
                    help="opt-in: extend Test 6's mamba2/mamba3 hybrid grid "
                         "with --axis-identity (+72 runs), mirroring Test 4 "
                         "for the SSM-hybrid hosts. Off the declared matrix; "
                         "requires 6 in --tests. See future_work.md.")
    ap.add_argument("--mechs", default=",".join(MECHS),
                    help="grid-mixing mechanisms in the axial tests. Default: "
                         + ",".join(MECHS) + ". Off-roster: 'aca' (axial "
                         "CROSS-attention, +144 runs) and 'fa_local' (the local FA "
                         "reimplementation, +144 runs).")
    ap.add_argument("--combo-bs", type=int, default=1,
                    help="combo/axial batch (per-group GRU flattens B*G=B*28292). "
                         "MEASURED training peak at batch 1, G=28,308, "
                         "forward+backward+AdamW on an H20: 40-61 GB across the "
                         "roster, worst cell mamba3_fa_4d at 61.4 GB. The older "
                         "'2 OOMs a 96GB H20' note described fa_local_4d under "
                         "its since-replaced per-line implementation and no "
                         "longer describes this code. Changing this changes the "
                         "step-matched protocol, so it is not a free knob.")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N (smoke)")
    ap.add_argument("--models", default=None,
                    help="comma list of --model values to include, e.g. mamba2,mamba3,mamba_nd "
                         "(default: all). Lets a 2.13 env run only the Mamba subset while a "
                         "torch-2.8 env runs everything else.")
    ap.add_argument("--force", action="store_true", help="re-run even finished runs (default: resume/skip them)")
    ap.add_argument("--only", default=None,
                    help="regex over RUN NAMES; run only matching runs. e.g. "
                         "'_1d_' (all flat), '^mamba3_' (all Mamba-3). Combines "
                         "with --force to redo a slice: --only '^mamba3_' --force")
    ap.add_argument("--exclude", default=None,
                    help="regex over run names to skip")
    ap.add_argument("--retrain", default=None, choices=sorted(RETRAIN_SETS),
                    help="named slice from RETRAIN.md, e.g. --retrain mandatory "
                         "(implies --force for the selected runs)")
    ap.add_argument("--status", action="store_true",
                    help="print per-run state (done/stale/partial/failed/todo) and exit")
    ap.add_argument("--reset-manifest", action="store_true",
                    help="start a fresh declared matrix and re-run its cells instead of "
                         "merging/resuming old completion records")
    ap.add_argument("--runs-dir", default="outputs/sweep",
                    help="session output directory containing manifest, run folders, and logs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.gpus:
        gpus = [int(g) for g in a.gpus.split(",")]
    else:
        # A dry run only prints the matrix, so it has to work on a laptop with
        # no CUDA at all -- inspecting what would run is not the thing that
        # needs a card. Training is: pinning CUDA_VISIBLE_DEVICES to a device
        # that is not there is exactly how a 27-run sweep came to fail every
        # run at startup. So the refusal applies there, and only there.
        try:
            gpus = resolve_gpus()
        except SystemExit:
            if not a.dry_run:
                raise
            gpus = [0]
        except Exception:
            gpus = [0]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else DEFAULT_SEEDS
    if "0" in set(a.tests.split(",")) and not a.seeds:
        # Exp 0 selects a hyperparameter; it does not estimate performance, so
        # it runs one seed by default rather than the full reporting cohort.
        # That the selection rests on a single seed is a real limitation and
        # belongs in the protocol paragraph -- select_lr.py records the seeds
        # it used. Pass --seeds explicitly to widen.
        seeds = seeds[:1]
        print(f"Exp 0: selection probe, using seed {seeds[0]} only "
              "(pass --seeds to widen)")

    requested_tests = set(a.tests.split(","))

    # Step-matched protocol (F1). The flat arm accumulates ceil(G / flat_bs)
    # micro-batches per optimizer step so both arms take the same number of
    # steps per epoch at the same effective batch.
    accum = 1
    effective_batch_size = None
    # Test 0 belongs here as much as 2/3 do: Exp 0's flat probes exist to pick
    # the rate the Test 2/3 flat runs will train at, and a rate does not
    # transfer across a 14x change in optimizer steps per epoch. Omitting it
    # probed at --grad-accum 1 with no effective batch, i.e. under a different
    # protocol than the runs it selects for -- exactly the mistake
    # build_exp0_matrix's docstring says it must not make.
    if a.step_match and requested_tests.intersection({"0", "2", "3"}):
        train_months = train_target_months()
        if train_months <= 0:
            sys.exit(f"config/census.yaml yields {train_months} training target "
                     "months; check input_len/lag_count/train_end")
        if train_months % a.combo_bs:
            sys.exit(f"--combo-bs must divide the benchmark's {train_months} training "
                     "target months when --step-match is enabled")
        n_series = a.series or lattice_n_series(a.npz)
        if n_series is None:
            sys.exit(
                f"--step-match needs the series count: {Path(a.npz).with_suffix('.json')} "
                "is unreadable. Pass --series G, or --no-step-match to run the "
                "legacy unmatched protocol."
            )
        accum = flat_accum(n_series, a.flat_bs, a.combo_bs)
        effective_batch_size = n_series * a.combo_bs
        print(f"step-match: G={n_series:,} series | flat micro-batch {a.flat_bs} "
              f"x {accum} micro-batches/group = exact effective "
              f"{effective_batch_size:,} "
              f"(combo arm: {a.combo_bs} month x {n_series:,} series)")
    elif not a.step_match:
        print("step-match DISABLED: flat arm runs ~55x more optimizer steps "
              "than the combo arm (legacy protocol; see RETRAIN.md)")

    mechs = [m for m in a.mechs.split(",") if m]
    unknown_mechs = sorted(set(mechs) - set(ALL_MECHS))
    if unknown_mechs:
        sys.exit("unknown --mechs: " + ", ".join(unknown_mechs)
                 + " (known: " + ", ".join(ALL_MECHS) + ")")
    lr_selection = None
    if a.lr_selection:
        selection_path = Path(a.lr_selection)
        if not selection_path.is_absolute():
            selection_path = REPO / selection_path
        try:
            lr_selection = json.loads(selection_path.read_text())
        except (OSError, ValueError) as exc:
            sys.exit(f"cannot read --lr-selection {selection_path}: {exc}")
        if lr_selection.get("schema_version") != 1:
            sys.exit(f"unsupported lr-selection schema in {selection_path}; "
                     "regenerate it with the current scripts/select_lr.py")
        # A patch is ADDITIVE and kept in its own file. The base selection is
        # the audit record of one Exp 0 session and is never rewritten; a
        # supplementary probe (say the fa mixer, run later in its own
        # --runs-dir) contributes its arms here, and the merged view is what
        # gets printed and injected. Nothing about the base session's runs,
        # checkpoints or manifest is touched.
        if a.lr_selection_patch:
            patch_path = Path(a.lr_selection_patch)
            if not patch_path.is_absolute():
                patch_path = REPO / patch_path
            try:
                patch = json.loads(patch_path.read_text())
            except (OSError, ValueError) as exc:
                sys.exit(f"cannot read --lr-selection-patch {patch_path}: {exc}")
            if patch.get("schema_version") != 1:
                sys.exit(f"unsupported schema in {patch_path}")
            added = []
            for arm, cells in sorted(patch.get("selected", {}).items()):
                base_cells = lr_selection["selected"].setdefault(arm, {})
                for model, rate in sorted(cells.items()):
                    was = base_cells.get(model)
                    if was != rate:
                        added.append(f"{arm}/{model}: {was or '-'} -> {rate}")
                    base_cells[model] = rate
            lr_selection.setdefault("patches", []).append(
                dict(path=str(patch_path), official=patch.get("official"),
                     applied=added))
            print(f"lr-selection patch ({patch_path.name}): "
                  + (", ".join(added) if added else "no change"))
        print("lr-selection: " + " | ".join(
            f"{arm}: " + ",".join(f"{m}={lr}" for m, lr in sorted(cells.items()))
            for arm, cells in sorted(lr_selection["selected"].items())))
    elif requested_tests - {"0"}:
        # Not fatal -- a diagnostic slice on config defaults is legitimate --
        # but silence here is how the unsearched-arm asymmetry got shipped.
        print("WARNING: no --lr-selection; every run uses its config-default "
              "learning rate, which is NOT a searched quantity. An official "
              "cross-arm table needs Exp 0 (--tests 0) plus select_lr.py.")

    # --models and --exp0-models filter EXECUTION only: the declaration stays
    # the complete matrix, so each split-environment half adds to one manifest
    # instead of rewriting it from its own partial view and dropping the other
    # half's cells.
    #
    # --exp0-arms is the deliberate exception. select_lr.py refuses to emit
    # anything while a declared cell is missing, so a supplementary probe MUST
    # declare only the arms it runs or it could never produce the selection it
    # exists for. That makes it destructive against a session that holds other
    # arms, which is what the guard below is for.
    if a.hybrid_identity and "6" not in requested_tests:
        sys.exit("--hybrid-identity extends Test 6's hybrid grid; include 6 "
                 "in --tests")
    declared_runs = build_matrix(
        requested_tests, seeds, a.flat_bs, a.combo_bs, accum=accum,
        npz=a.npz, effective_batch_size=effective_batch_size, mechs=mechs,
        lr_selection=lr_selection,
        exp0_arms=([x for x in a.exp0_arms.split(",") if x]
                   if a.exp0_arms else None),
        hybrid_identity=a.hybrid_identity,
    )
    force = a.force
    only = a.only
    if a.retrain:
        only = RETRAIN_SETS[a.retrain]
        force = True            # a named retrain slice always re-runs
        print(f"retrain set {a.retrain!r}: selecting /{only}/ (forced re-run)")
    runs = select(declared_runs, only, a.exclude)
    model_filters = [a.models, a.exp0_models]
    for spec in model_filters:         # filter by the actual --model value in each argv
        if not spec:
            continue
        keep = {m for m in spec.split(",") if m}
        runs = [(n, argv) for n, argv in runs
                if argv[argv.index("--model") + 1] in keep]
    if a.limit:
        runs = runs[:a.limit]
    if not runs:
        sys.exit("no runs matched the requested filters")

    runs_dir = Path(a.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = REPO / runs_dir
    logdir = runs_dir / "logs"
    if a.exp0_arms:
        _refuse_arm_probe_over_existing_session(a.exp0_arms, runs_dir)
    if a.status:
        runs_dir = logdir.parent
        counts = {}
        for name, _, entry in prepare_runs(runs, a.epochs, runs_dir):
            st = run_state(entry, runs_dir, logdir)
            counts[st] = counts.get(st, 0) + 1
            print(f"  {st:8} {name}")
        print("\n" + " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
              + f" | total {len(runs)}")
        sys.exit(0)

    worker_env = {}
    if a.num_workers is not None:
        worker_env["CENSUS_NUM_WORKERS"] = str(a.num_workers)
    if a.combo_num_workers is not None:
        worker_env["CENSUS_COMBO_NUM_WORKERS"] = str(a.combo_num_workers)

    schedule(runs, gpus, a.epochs, logdir, a.dry_run, resume=not force,
             reset_manifest=a.reset_manifest, declared_runs=declared_runs,
             stop_file=runs_dir / STOP_FILE, worker_env=worker_env)
