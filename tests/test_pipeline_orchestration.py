"""hpec_pipeline.sh must run the stages in the order the protocol requires.

Exp 0 selects a learning rate per (arm, model) on validation loss; the main
matrix then runs on those rates. If the selection is not threaded through, the
sweep silently falls back to config defaults that were never searched for the
multidimensional arm -- and nothing downstream says so, because an unsearched
rate looks exactly like a searched one in results.csv.

These tests drive the real script with a stub interpreter and read back the
command lines it composed.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "hpec_pipeline.sh"


@pytest.fixture
def stub(tmp_path):
    """A fake `python` that records its arguments instead of running them."""
    log = tmp_path / "calls.log"
    path = tmp_path / "stub_python"
    # It also stands in for select_lr.py's side effect, so the exp0 -> train
    # handoff can be exercised inside a single invocation.
    path.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$LOGFILE"\n'
        'case "$*" in\n'
        '  *select_lr.py*)\n'
        '    mkdir -p "$EXP0_DIR"\n'
        '    printf \'{"schema_version": 1, "selected": {}}\' > "$EXP0_DIR/lr_selection.json"\n'
        '    ;;\n'
        'esac\n'
        'exit 0\n')
    path.chmod(0o755)
    return path, log


def _run(stub, env_extra, tmp_path, expect_rc=0):
    path, log = stub
    session = "pytest-" + tmp_path.name.replace("_", "-")[:24]
    env = {**os.environ, "PY": str(path), "LOGFILE": str(log),
           "SERIES": "28292", "SESSION_ID": session,
           "EXP0_DIR": str(tmp_path / "exp0"), **env_extra}
    proc = subprocess.run(["bash", str(SCRIPT)], cwd=str(REPO), env=env,
                          capture_output=True, text=True)
    shutil.rmtree(REPO / "outputs" / "sweep" / "sessions" / session,
                  ignore_errors=True)
    assert proc.returncode == expect_rc, proc.stdout + proc.stderr
    calls = log.read_text().splitlines() if log.exists() else []
    return calls, proc.stdout + proc.stderr


def _find(calls, script):
    return [c for c in calls if script in c]


# --------------------------------------------------------------------------
# the exp0 stage
# --------------------------------------------------------------------------

def test_exp0_stage_probes_then_selects(stub, tmp_path):
    calls, _ = _run(stub, {"STAGES": "exp0"}, tmp_path)
    probe = _find(calls, "scripts/sweep.py")
    select = _find(calls, "scripts/select_lr.py")
    assert len(probe) == 1 and len(select) == 1
    assert "--tests 0" in probe[0]
    assert calls.index(probe[0]) < calls.index(select[0]), "probe must run first"


def test_exp0_runs_in_its_own_directory(stub, tmp_path):
    """Mixing Exp 0 into the main manifest would declare runs nobody reports."""
    calls, _ = _run(stub, {"STAGES": "exp0"}, tmp_path)
    probe = _find(calls, "scripts/sweep.py")[0]
    assert f"--runs-dir {tmp_path / 'exp0'}" in probe
    assert "sessions" not in probe.split("--runs-dir")[1].split()[0]


def test_exp0_uses_a_short_epoch_budget_by_default(stub, tmp_path):
    """Ranking rates does not need the 200-epoch cap the real runs get."""
    calls, _ = _run(stub, {"STAGES": "exp0"}, tmp_path)
    assert "--epochs 20" in _find(calls, "scripts/sweep.py")[0]


def test_exp0_epoch_budget_is_overridable(stub, tmp_path):
    calls, _ = _run(stub, {"STAGES": "exp0", "EXP0_EPOCHS": "35"}, tmp_path)
    assert "--epochs 35" in _find(calls, "scripts/sweep.py")[0]


def test_exp0_model_restriction_is_passed_through(stub, tmp_path):
    """Split environments: a box without Triton cannot run the mamba cells."""
    calls, _ = _run(stub, {"STAGES": "exp0", "EXP0_MODELS": "gru,lstm"}, tmp_path)
    assert "--exp0-models gru,lstm" in _find(calls, "scripts/sweep.py")[0]


# --------------------------------------------------------------------------
# the selection reaches the training stage
# --------------------------------------------------------------------------

def test_an_explicit_selection_is_passed_to_the_sweep(stub, tmp_path):
    selection = tmp_path / "lr_selection.json"
    selection.write_text("{}")
    calls, _ = _run(stub, {"STAGES": "train", "LR_SELECTION": str(selection)},
                    tmp_path)
    assert f"--lr-selection {selection}" in _find(calls, "scripts/sweep.py")[0]


def test_the_exp0_output_is_picked_up_without_being_named(stub, tmp_path):
    """Forgetting the flag would silently revert the sweep to unsearched
    defaults, so a selection sitting in EXP0_DIR is used by default."""
    exp0 = tmp_path / "exp0"
    exp0.mkdir(parents=True)
    (exp0 / "lr_selection.json").write_text("{}")
    calls, out = _run(stub, {"STAGES": "train"}, tmp_path)
    assert f"--lr-selection {exp0 / 'lr_selection.json'}" in \
        _find(calls, "scripts/sweep.py")[0]
    assert "using learning rates from" in out


def test_training_without_a_selection_says_so_loudly(stub, tmp_path):
    calls, out = _run(stub, {"STAGES": "train"}, tmp_path)
    assert "--lr-selection" not in _find(calls, "scripts/sweep.py")[0]
    assert "WARNING" in out and "never searched" in out


def test_an_explicit_selection_wins_over_the_exp0_directory(stub, tmp_path):
    exp0 = tmp_path / "exp0"
    exp0.mkdir(parents=True)
    (exp0 / "lr_selection.json").write_text("{}")
    explicit = tmp_path / "chosen.json"
    explicit.write_text("{}")
    calls, _ = _run(stub, {"STAGES": "train", "LR_SELECTION": str(explicit)},
                    tmp_path)
    sweep = _find(calls, "scripts/sweep.py")[0]
    assert f"--lr-selection {explicit}" in sweep
    assert str(exp0 / "lr_selection.json") not in sweep


# --------------------------------------------------------------------------
# equivalence testing is configurable
# --------------------------------------------------------------------------

def test_the_sig_stage_carries_the_equivalence_margin(stub, tmp_path):
    calls, _ = _run(stub, {"STAGES": "sig"}, tmp_path)
    sig = _find(calls, "scripts/significance_tests.py")[0]
    assert "--delta-mode frac" in sig and "--delta 0.05" in sig
    assert "--alpha 0.05" in sig


def test_the_equivalence_margin_is_overridable(stub, tmp_path):
    calls, _ = _run(stub, {"STAGES": "sig", "DELTA_MODE": "seed",
                           "DELTA": "0.25", "ALPHA": "0.01"}, tmp_path)
    sig = _find(calls, "scripts/significance_tests.py")[0]
    assert "--delta-mode seed" in sig
    assert "--delta 0.25" in sig and "--alpha 0.01" in sig


# --------------------------------------------------------------------------
# the retired test must not reappear in the docs
# --------------------------------------------------------------------------

def test_the_script_never_suggests_the_retired_test():
    source = SCRIPT.read_text()
    assert "1,1.1,2,3,4,5,6" not in source
    assert "LR_BRACKET_MODELS" not in source


def test_the_help_text_covers_the_whole_header():
    """--help prints a line range; adding to the header must not truncate it."""
    out = subprocess.run(["bash", str(SCRIPT), "--help"], cwd=str(REPO),
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "STAGES" in out.stdout and "exp0" in out.stdout
    assert "FORCE=1" in out.stdout, "the tail of the header was cut off"
    assert "set -uo pipefail" not in out.stdout, "the range ran past the header"


# --------------------------------------------------------------------------
# loader workers: infrastructure, deliberately outside the fingerprint
# --------------------------------------------------------------------------

def test_worker_counts_reach_the_sweep(stub, tmp_path):
    calls, _ = _run(stub, {"STAGES": "train", "NUM_WORKERS": "8",
                           "COMBO_NUM_WORKERS": "1"}, tmp_path)
    sweep = _find(calls, "scripts/sweep.py")[0]
    assert "--num-workers 8" in sweep
    assert "--combo-num-workers 1" in sweep


def test_worker_counts_reach_the_exp0_probe_too(stub, tmp_path):
    calls, _ = _run(stub, {"STAGES": "exp0", "NUM_WORKERS": "8"}, tmp_path)
    assert "--num-workers 8" in _find(calls, "scripts/sweep.py")[0]


def test_worker_count_never_enters_a_run_fingerprint(tmp_path):
    """The whole point of passing workers by environment: retuning them must
    not invalidate finished checkpoints."""
    from scripts.sweep import build_matrix, prepare_runs

    matrix = build_matrix({"3"}, [947], 2048, 1, accum=14,
                          effective_batch_size=28292)
    runs = prepare_runs(matrix, 200, tmp_path)
    for _, argv, entry in runs:
        assert "--num-workers" not in argv
        assert "CENSUS_NUM_WORKERS" not in " ".join(argv)
        assert entry["fingerprint"]


def test_the_scheduler_puts_worker_counts_in_the_child_environment(tmp_path):
    """End to end: the child process must actually see the variables."""
    from scripts.sweep import schedule

    probe = tmp_path / "probe.py"
    out = tmp_path / "env.txt"
    probe.write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(out)!r}).write_text(\n"
        "    os.environ.get('CENSUS_NUM_WORKERS', '<unset>') + ',' +\n"
        "    os.environ.get('CENSUS_COMBO_NUM_WORKERS', '<unset>'))\n"
    )
    runs = [("gru_embeddings_1d_s947", [sys.executable, str(probe)])]
    with pytest.raises(SystemExit):
        # exits 2: the probe writes no best.pth. The environment is the point.
        schedule(runs, gpus=[0], epochs=1, logdir=tmp_path / "s" / "logs",
                 dry=False,
                 worker_env={"CENSUS_NUM_WORKERS": "8",
                             "CENSUS_COMBO_NUM_WORKERS": "1"})
    assert out.read_text() == "8,1"


def test_train_py_reads_the_worker_count_from_the_environment():
    source = (REPO / "scripts/train.py").read_text()
    assert "CENSUS_COMBO_NUM_WORKERS" in source
    assert "CENSUS_NUM_WORKERS" in source
    assert "num_workers=4," not in source, "the hard-coded count must be gone"


def test_exp0_then_train_in_one_invocation_uses_the_generated_selection(stub, tmp_path):
    """STAGES=exp0,train is the natural combined invocation, and it silently
    trained the whole matrix on config defaults.

    sweep_args is assembled once at startup, before any stage runs, so the
    "did exp0 leave a selection?" check ran while the file could not exist
    yet. The run then printed "the train stage picks them up automatically"
    and did the opposite.
    """
    calls, out = _run(stub, {"STAGES": "exp0,train"}, tmp_path)
    train = [c for c in _find(calls, "scripts/sweep.py") if "--tests 0" not in c]
    assert len(train) == 1, "expected exactly one main sweep invocation"
    assert f"--lr-selection {tmp_path / 'exp0' / 'lr_selection.json'}" in train[0]
    assert "WARNING" not in out, "it found the selection, so must not warn"


def test_exp0_alone_still_leaves_the_selection_for_a_later_invocation(stub, tmp_path):
    calls, _ = _run(stub, {"STAGES": "exp0"}, tmp_path)
    assert (tmp_path / "exp0" / "lr_selection.json").exists()
    calls2, _ = _run(stub, {"STAGES": "train"}, tmp_path)
    # the stub appends to one shared log, so skip the earlier exp0 probe call
    train = [c for c in _find(calls2, "scripts/sweep.py") if "--tests 0" not in c]
    assert len(train) == 1
    assert "--lr-selection" in train[0]


def test_the_selection_flag_is_never_passed_twice(stub, tmp_path):
    """apply_lr_selection runs at assembly AND before the train stage."""
    exp0 = tmp_path / "exp0"
    exp0.mkdir(parents=True)
    (exp0 / "lr_selection.json").write_text("{}")
    calls, _ = _run(stub, {"STAGES": "train"}, tmp_path)
    assert _find(calls, "scripts/sweep.py")[0].count("--lr-selection") == 1


def test_training_without_any_selection_still_warns(stub, tmp_path):
    """The second apply_lr_selection call must not mask the warning when there
    genuinely is no selection to find."""
    calls, out = _run(stub, {"STAGES": "train"}, tmp_path)
    assert "--lr-selection" not in _find(calls, "scripts/sweep.py")[0]
    assert "WARNING" in out


def test_the_pipeline_runs_python_unbuffered():
    """A multi-hour sweep redirected to a file shows no progress at all
    otherwise: Python block-buffers stdout when it is not a terminal, so
    `nohup ... > sweep.log` stays empty while every GPU is busy, and a stalled
    sweep looks identical to a working one."""
    source = SCRIPT.read_text()
    assert "export PYTHONUNBUFFERED=1" in source


def test_the_fa_probe_is_folded_in_without_being_named(stub, tmp_path):
    """The supplementary probe lives in its own session directory. When it has
    produced a selection, the train stage merges it over the base without the
    operator having to remember a flag -- forgetting it would silently train
    144 fa runs on a rate probed on a different mixer."""
    exp0 = tmp_path / "exp0"
    exp0.mkdir(parents=True)
    (exp0 / "lr_selection.json").write_text("{}")
    fa = tmp_path / "exp0_fa"
    fa.mkdir(parents=True)
    (fa / "lr_selection.json").write_text("{}")
    calls, out = _run(stub, {"STAGES": "train", "EXP0_FA_DIR": str(fa)}, tmp_path)
    sweep = _find(calls, "scripts/sweep.py")[0]
    assert f"--lr-selection {exp0 / 'lr_selection.json'}" in sweep
    assert f"--lr-selection-patch {fa / 'lr_selection.json'}" in sweep
    assert "folding in the fa probe" in out


def test_no_patch_flag_when_no_fa_probe_exists(stub, tmp_path):
    """A base-only run must stay exactly as it was."""
    exp0 = tmp_path / "exp0"
    exp0.mkdir(parents=True)
    (exp0 / "lr_selection.json").write_text("{}")
    calls, _ = _run(stub, {"STAGES": "train",
                           "EXP0_FA_DIR": str(tmp_path / "nope")}, tmp_path)
    assert "--lr-selection-patch" not in _find(calls, "scripts/sweep.py")[0]
