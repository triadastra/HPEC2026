"""Exp 0 must select rates under the protocol the selected runs actually use.

Each test here pins a way the probe could silently diverge from the runs it
selects for. All four were real: a rate is only transferable if the thing it
was measured on matches the thing it is applied to, and every mismatch below
was invisible in the output -- the sweep still ran, still produced a
leaderboard, and still looked searched.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.sweep import DIVERGED_EXIT_CODE, DIVERGED_FILE, FLAT_MODELS, schedule

SWEEP = str(REPO / "scripts" / "sweep.py")


def _dry_run(*args):
    proc = subprocess.run(
        [sys.executable, SWEEP, "--seeds", "947", "--series", "28292",
         "--dry-run", "--gpus", "0", *args],
        cwd=str(REPO), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return [l for l in proc.stdout.splitlines() if " :: " in l]


def _flag(line, flag):
    m = re.search(rf"{re.escape(flag)} (\S+)", line)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# the batch protocol
# --------------------------------------------------------------------------

def test_exp0_flat_probes_match_the_step_protocol_they_select_for():
    """The probe ran at --grad-accum 1 with no effective batch while the runs
    it selects for ran at accum 14 -- a 14x difference in optimizer steps per
    epoch, under which a learning rate does not transfer."""
    probe = [l for l in _dry_run("--tests", "0", "--epochs", "20")
             if "exp0_flat_gru_" in l][0]
    real = [l for l in _dry_run("--tests", "3") if "gru_embeddings_1d_" in l][0]

    assert _flag(probe, "--grad-accum") == _flag(real, "--grad-accum")
    assert _flag(probe, "--effective-batch-size") == \
        _flag(real, "--effective-batch-size")
    assert _flag(probe, "--batch-size") == _flag(real, "--batch-size")
    assert _flag(probe, "--grad-accum") != "1", "step matching silently absent"


def test_exp0_nd_probes_match_the_combo_batch_they_select_for():
    probe = [l for l in _dry_run("--tests", "0", "--epochs", "20")
             if "exp0_nd_gru_" in l][0]
    real = [l for l in _dry_run("--tests", "3") if "gru_asa_3d_embeddings" in l][0]
    assert _flag(probe, "--batch-size") == _flag(real, "--batch-size")


def test_exp0_aggregate_probes_match_the_aggregate_batch():
    probe = [l for l in _dry_run("--tests", "0", "--epochs", "20")
             if "exp0_agg_gru_" in l][0]
    real = [l for l in _dry_run("--tests", "1") if "gru_aggregate_s" in l][0]
    assert _flag(probe, "--batch-size") == _flag(real, "--batch-size") == "32"


# --------------------------------------------------------------------------
# split environments
# --------------------------------------------------------------------------

def test_exp0_models_restricts_execution():
    runs = _dry_run("--tests", "0", "--epochs", "20", "--exp0-models", "gru,lstm")
    models = {_flag(l, "--model") for l in runs}
    assert models == {"gru", "lstm"}


def test_a_restricted_execution_still_declares_the_whole_matrix(tmp_path):
    """The reported failure: each split-environment half rewrote manifest.json
    from its own partial declaration, dropping the other half. select_lr.py
    reads that manifest, so it could then only ever produce a partial
    selection -- which the main sweep correctly refuses, for every model the
    other machine ran."""
    full = [(f"exp0_flat_{m}_lr1e4_s947", [sys.executable, "-c", "pass"])
            for m in FLAT_MODELS]
    subset = [r for r in full if "_gru_" in r[0]]
    runs_dir = tmp_path / "exp0"

    with pytest.raises(SystemExit):     # the stubs write no best.pth
        schedule(subset, gpus=[0], epochs=20, logdir=runs_dir / "logs",
                 dry=False, declared_runs=full)

    manifest = json.loads((runs_dir / "manifest.json").read_text())
    assert {e["name"] for e in manifest["runs"]} == {n for n, _ in full}, (
        "the manifest must declare every cell, not just the executed subset")


# --------------------------------------------------------------------------
# stale divergence markers
# --------------------------------------------------------------------------

def test_a_stale_divergence_marker_is_cleared_before_relaunch(tmp_path):
    """run_diverged.json carries no fingerprint, so select_lr.py trusts it by
    pathname. Left behind from an earlier declaration it would keep excluding
    this cell's validation curve after the cell trains fine -- silently
    changing the selected rate."""
    name = "exp0_flat_gru_lr1e2_s947"
    runs_dir = tmp_path / "exp0"
    (runs_dir / name).mkdir(parents=True)
    stale = runs_dir / name / DIVERGED_FILE
    stale.write_text(json.dumps({"status": "diverged", "loss": "nan"}))

    with pytest.raises(SystemExit):     # stub writes no best.pth
        schedule([(name, [sys.executable, "-c", "pass"])], gpus=[0], epochs=20,
                 logdir=runs_dir / "logs", dry=False)

    assert not stale.exists(), "the stale divergence marker outlived its run"


def test_a_run_that_diverges_again_rewrites_its_marker(tmp_path):
    """Clearing the marker must not lose a genuine, current divergence."""
    name = "exp0_flat_gru_lr1e2_s947"
    runs_dir = tmp_path / "exp0"
    marker = runs_dir / name / DIVERGED_FILE
    probe = tmp_path / "diverge.py"
    probe.write_text(
        "import json, pathlib, sys\n"
        f"p = pathlib.Path({str(marker)!r})\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_text(json.dumps({'status': 'diverged', 'loss': 'nan'}))\n"
        f"sys.exit({DIVERGED_EXIT_CODE})\n"
    )
    schedule([(name, [sys.executable, str(probe)])], gpus=[0], epochs=20,
             logdir=runs_dir / "logs", dry=False)
    assert json.loads(marker.read_text())["status"] == "diverged"


# --------------------------------------------------------------------------
# the general form: enumerate EVERY difference, allow only declared ones
# --------------------------------------------------------------------------
#
# The four bugs this file was written for were all the same shape: a probe and
# the runs it selects for disagreeing on some flag, invisibly. Testing them one
# at a time only catches the ones already known. This enumerates every flag
# difference and fails on any that is not covered by a declared check arm.

from scripts.sweep import (EXP0_ARMS, EXP0_CHECK_BASE,  # noqa: E402
                           build_exp0_matrix)

_CACHE = {}


def _runs_for(tests):
    if tests not in _CACHE:
        _CACHE[tests] = _dry_run("--tests", tests, "--epochs", "20")
    return _CACHE[tests]


def _flags(line):
    """Flag -> value for one dry-run command line, minus what legitimately
    differs on every probe: the rate being probed, the truncated budget, and
    the output path."""
    argv = line.split(" :: ", 1)[1].split()
    out, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[argv[i]] = argv[i + 1]; i += 2
            else:
                out[argv[i]] = True; i += 1
        else:
            i += 1
    for ignore in ("--lr", "--epochs", "--out-dir", "--seed",
                   "--config", "--data-config", "--npz"):
        out.pop(ignore, None)
    return out


def _one(lines, prefix):
    hit = [l for l in lines if l.split(" :: ", 1)[0].strip().startswith(prefix)]
    assert hit, f"no run named {prefix}*"
    return _flags(hit[0])


# (probe, tests, real run, flags allowed to differ, the check arm covering them)
COVERAGE = [
    ("exp0_flat_gru_", "3",   "gru_embeddings_1d_",        set(),                  None),
    ("exp0_flat_gru_", "2",   "gru_onehot_1d_",            {"--variant"},          "encflat"),
    ("exp0_nd_gru_",   "3",   "gru_asa_3d_embeddings_",    set(),                  None),
    ("exp0_nd_gru_",   "2",   "gru_asa_3d_onehot_",        {"--combo-encoder"},    "encnd"),
    ("exp0_nd_gru_",   "4",   "gru_asa_3d_id_embeddings_", {"--axis-identity"},    "axid"),
    ("exp0_nd_gru_",   "3",   "gru_fa_3d_embeddings_",     {"--variant"},          "mech"),
    ("exp0_nd_mamba3_", "6",  "mamba3_fa_3d_embeddings_",  {"--variant"},          "mech"),
    ("exp0_agg_gru_",  "1",   "gru_aggregate_s",           set(),                  None),
    ("exp0_roll_gru_", "1.1", "gru_aggregate_roll_y2020_", set(),                  None),
    ("exp0_roll_gru_", "1.1", "gru_aggregate_roll_y2025_",
     {"--train-end", "--val-end", "--test-end"}, "rollend"),
]


@pytest.mark.parametrize("probe,tests,real,allowed,check_arm", COVERAGE)
def test_probe_and_run_differ_only_where_a_check_arm_says_they_may(
        probe, tests, real, allowed, check_arm):
    p = _one(_runs_for("0"), probe)
    r = _one(_runs_for(tests), real)
    differing = {k for k in set(p) | set(r) if p.get(k) != r.get(k)}
    undeclared = differing - allowed
    assert not undeclared, (
        f"{probe}* and {real}* differ on {sorted(undeclared)}, which no check "
        "arm covers. Either match the protocol or declare a check arm for it: "
        "an undeclared difference is an unexamined assumption that the rate "
        "still transfers."
    )
    assert not (allowed - differing), (
        f"{sorted(allowed - differing)} is declared as a difference but the "
        "two command lines now agree on it -- the allowance is stale")


@pytest.mark.parametrize("probe,tests,real,allowed,check_arm", COVERAGE)
def test_each_declared_difference_has_a_check_arm_that_actually_runs(
        probe, tests, real, allowed, check_arm):
    if check_arm is None:
        assert not allowed
        return
    # A difference is covered either by a CHECK arm (reported, so the paper can
    # state whether the transfer held) or by a SELECTING arm (resolved, because
    # those cells get their own rate). `mech` began as the former and was
    # promoted to the latter once its two-model version showed the asa->fa
    # inheritance reverses a cell.
    assert check_arm in EXP0_CHECK_BASE or check_arm in EXP0_ARMS, (
        f"{check_arm} is neither a declared check arm nor a selecting arm")
    matrix = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    assert any(n.startswith(f"exp0_{check_arm}_") for n, _ in matrix), (
        f"{check_arm} covers {sorted(allowed)} but emits no runs")
    if check_arm in EXP0_ARMS:
        # Resolved, not merely reported: those runs must actually receive the
        # arm's rate rather than the base arm's.
        from scripts.sweep import nd_rate_arm
        assert nd_rate_arm("fa_3d") == check_arm
