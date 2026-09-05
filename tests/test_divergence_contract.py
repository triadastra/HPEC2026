"""Divergence is a RESULT, not a crash -- and never a silent one.

The failure this guards against: the axial modules used to scrub NaN to zero
mid-forward, so a diverged run finished training, wrote a checkpoint, earned a
completion record, passed every provenance gate, and landed in results.csv as
an ordinary bad number. The scrubbing is gone; these tests pin the replacement
contract end to end -- trainer raises, train.py records and exits 17, the
scheduler buckets it, select_lr.py counts the rate as attempted.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.sweep import DIVERGED_EXIT_CODE, DIVERGED_FILE, schedule
from src.training import NonFiniteLossError
from src.utils import write_divergence
from src.utils.run_manifest import DIVERGED_FILE as MANIFEST_DIVERGED_FILE


# --------------------------------------------------------------------------
# one definition, shared by every stage
# --------------------------------------------------------------------------

def test_the_divergence_contract_has_a_single_definition():
    """train.py writes it, sweep.py counts it, select_lr.py reads it. Three
    copies of the filename would drift and the divergence would go unnoticed."""
    assert DIVERGED_FILE == MANIFEST_DIVERGED_FILE == "run_diverged.json"
    assert DIVERGED_EXIT_CODE == 17


def test_every_stage_imports_the_constant_rather_than_spelling_it():
    for path in ("scripts/train.py", "scripts/sweep.py", "scripts/select_lr.py"):
        source = (REPO / path).read_text()
        body = source.split('"""', 2)[-1]        # ignore the module docstring
        assert '"run_diverged.json"' not in body, f"{path} re-declares the name"


# --------------------------------------------------------------------------
# the record itself
# --------------------------------------------------------------------------

def test_a_nan_loss_is_recorded_as_strict_json(tmp_path):
    """json.dump would emit a bare NaN token: readable by Python, rejected by
    every strict parser, so the record would be unusable outside this repo."""
    path = write_divergence(tmp_path / "run", {
        "status": "diverged", "loss": float("nan"), "epoch": 0, "step": 3})
    raw = path.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    record = json.loads(raw)          # strict: no NaN literal accepted
    assert record["loss"] == "nan"
    assert record["epoch"] == 0 and record["step"] == 3


def test_an_infinite_loss_is_recorded_too(tmp_path):
    path = write_divergence(tmp_path / "run", {"loss": float("inf")})
    assert json.loads(path.read_text())["loss"] == "inf"


def test_the_record_lands_next_to_the_checkpoint_that_was_never_written(tmp_path):
    out = tmp_path / "runs" / "exp0_flat_gru_lr1e2_s947"
    path = write_divergence(out, {"loss": float("nan")})
    assert path == out / DIVERGED_FILE
    assert not (out / "best.pth").exists(), (
        "a diverged run must leave no checkpoint, or it could earn a "
        "completion record and be scored as a result")


def test_the_exception_carries_where_it_broke():
    exc = NonFiniteLossError(float("nan"), epoch=4, step=117)
    assert exc.epoch == 4 and exc.step == 117
    assert "diverged" in str(exc)


# --------------------------------------------------------------------------
# the trainer raises instead of continuing
# --------------------------------------------------------------------------

def test_the_trainer_stops_at_the_first_non_finite_loss():
    """Continuing is provably useless: the backward pass writes NaN into every
    parameter, so every later forward is NaN too."""
    import torch
    from src.training.trainer import Trainer

    source = (REPO / "src/training/trainer.py").read_text()
    assert "raise NonFiniteLossError" in source
    # and the check runs on the scalar that is accumulated, before it is used
    idx_check = source.index("if not math.isfinite(batch_loss)")
    idx_accum = source.index("total_loss += batch_loss")
    assert idx_check < idx_accum, "a NaN must never reach the running total"


def test_no_module_scrubs_nan_mid_forward():
    """nan_to_num in an attention block laundered divergence into finite
    numbers, which is what let a diverged run pass every gate downstream."""
    for name in ("combo_attention.py", "fa.py", "fa_local.py", "aca.py"):
        path = REPO / "src" / "models" / name
        if not path.exists():
            continue
        code = "\n".join(line for line in path.read_text().splitlines()
                         if not line.strip().startswith("#"))
        assert "nan_to_num" not in code, f"{name} still scrubs NaN"


# --------------------------------------------------------------------------
# the scheduler buckets it, and does not fail the sweep over it
# --------------------------------------------------------------------------

def _fake_run(exit_code, name):
    """A run that exits with a chosen code without training anything."""
    return (name, [sys.executable, "-c", f"import sys; sys.exit({exit_code})"])


def test_the_scheduler_counts_divergence_separately(tmp_path, capsys):
    runs_dir = tmp_path / "session"
    logdir = runs_dir / "logs"
    schedule([_fake_run(DIVERGED_EXIT_CODE, "exp0_flat_gru_lr1e2_s947")],
             gpus=[0], epochs=1, logdir=logdir, dry=False)
    out = capsys.readouterr().out
    assert "DIVERGED" in out
    assert "1 diverged" in out
    assert "0 failed" in out


def test_a_diverged_run_alone_does_not_fail_the_sweep(tmp_path):
    """Exp 0 probes a deliberately-too-high rate. If divergence exited
    non-zero, a CORRECT Exp 0 sweep would stop the pipeline before selection."""
    runs_dir = tmp_path / "session"
    schedule([_fake_run(DIVERGED_EXIT_CODE, "exp0_flat_gru_lr1e2_s947")],
             gpus=[0], epochs=1, logdir=runs_dir / "logs", dry=False)
    # no SystemExit


def test_a_real_failure_still_fails_the_sweep(tmp_path):
    runs_dir = tmp_path / "session"
    with pytest.raises(SystemExit) as excinfo:
        schedule([_fake_run(1, "gru_embeddings_1d_s947")],
                 gpus=[0], epochs=1, logdir=runs_dir / "logs", dry=False)
    assert excinfo.value.code == 1


def test_divergence_and_failure_are_not_confused(tmp_path, capsys):
    runs_dir = tmp_path / "session"
    with pytest.raises(SystemExit):
        schedule([_fake_run(DIVERGED_EXIT_CODE, "exp0_flat_gru_lr1e2_s947"),
                  _fake_run(1, "exp0_flat_lstm_lr1e2_s947")],
                 gpus=[0], epochs=1, logdir=runs_dir / "logs", dry=False)
    out = capsys.readouterr().out
    assert "1 diverged" in out and "1 failed" in out
    assert "exp0_flat_gru_lr1e2_s947" not in out.split("FAILED:")[-1]


def test_a_diverged_run_earns_no_completion_record(tmp_path):
    """The provenance gate must still reject it as evidence."""
    from src.utils import completion_path
    runs_dir = tmp_path / "session"
    schedule([_fake_run(DIVERGED_EXIT_CODE, "exp0_flat_gru_lr1e2_s947")],
             gpus=[0], epochs=1, logdir=runs_dir / "logs", dry=False)
    assert not completion_path(runs_dir, "exp0_flat_gru_lr1e2_s947").exists()


# --------------------------------------------------------------------------
# train.py wires the two ends together
# --------------------------------------------------------------------------

def test_train_py_catches_divergence_and_exits_with_the_reserved_code():
    source = (REPO / "scripts/train.py").read_text()
    assert "except NonFiniteLossError" in source
    assert "write_divergence" in source
    assert "sys.exit(DIVERGED_EXIT_CODE)" in source
    # and it must not fall through into the completion/cost bookkeeping
    caught = source.index("except NonFiniteLossError")
    assert source.index("sys.exit(DIVERGED_EXIT_CODE)") > caught
    assert source.index('print("Training completed!")') > \
        source.index("sys.exit(DIVERGED_EXIT_CODE)")


# --------------------------------------------------------------------------
# per-epoch diagnostics: detail has to be free, or it will be turned off
# --------------------------------------------------------------------------

def test_the_clip_norm_is_captured_not_discarded():
    """clip_grad_norm_ computes the pre-clip total norm and returns it, and
    the loop already pays for that computation -- discarding it bought
    nothing.

    It is also the evidence behind the stability finding: grad-clip turns some
    divergence into finite-but-bad loss rather than NaN, so a run can look
    merely poor while actually being clipped at every step. clip_fraction is
    what tells those apart.
    """
    source = (REPO / "src/training/trainer.py").read_text()
    assert "grad_norms.append(torch.nn.utils.clip_grad_norm_(" in source
    for field in ("grad_norm_mean", "grad_norm_max", "clip_fraction"):
        assert field in source


def test_the_norms_are_reduced_once_per_epoch_not_per_step():
    """The norm is a device tensor, so .item() on it forces a sync. Per
    optimizer step that is a real cost on a multi-hour sweep; once per epoch
    over a few dozen scalars it is not measurable. Verified at -0.0% wall
    clock against the same run before the change, with a bit-identical
    train_loss.
    """
    source = (REPO / "src/training/trainer.py").read_text()
    # The norm is kept as a device tensor: .detach() and append, never
    # .item(). (The loop's existing loss.item() is a separate, deliberate
    # sync for the divergence check and is not what this guards.)
    capture = source[source.index("grad_norms.append"):]
    capture = capture[:capture.index(")\n")]
    assert ".detach()" in capture, "the norm must be detached, not materialised"
    assert ".item()" not in capture and "float(" not in capture, (
        "materialising the grad norm at the optimizer step syncs the device "
        "on every step")
    assert "torch.stack(grad_norms)" in source, "reduce once, at epoch end"


def test_the_epoch_record_carries_what_a_long_unattended_run_needs():
    source = (REPO / "src/training/trainer.py").read_text()
    for field in ("train_seconds", "val_seconds", "steps", "total_steps",
                  "observations", "best_val_loss",
                  "epochs_without_improvement", "gpu_peak_gb"):
        assert f'"{field}"' in source, f"{field} missing from the epoch record"
