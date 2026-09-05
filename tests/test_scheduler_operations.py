"""Scheduler behaviours a long sweep depends on, none of which were covered.

Two of these only matter at operating scale -- GPU pinning when the box has
more than one card, and the STOP flag when a sweep runs for days and has to be
halted without corrupting it. Both worked when probed; these pin them, because
a regression in either is expensive exactly when it is least convenient.
"""

import pathlib
import sys
import tempfile
import threading
import time
from collections import Counter

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.sweep import STOP_FILE, schedule
from src.utils import completion_path


def _writer(tmp_path):
    """A stub run that behaves like a successful one: writes a checkpoint."""
    script = tmp_path / "writer.py"
    script.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'best.pth').write_bytes(b'ckpt-' + sys.argv[1].encode())\n")
    return script


def _slow_writer(tmp_path, seconds=3):
    script = tmp_path / "slow.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        f"time.sleep({seconds})\n"
        "(out / 'best.pth').write_bytes(b'ckpt-' + sys.argv[1].encode())\n")
    return script


# --------------------------------------------------------------------------
# GPU pinning
# --------------------------------------------------------------------------

@pytest.fixture
def no_instant_abort(monkeypatch):
    """Neutralise the instant-failure abort for tests that are about something
    else. Their probes deliberately write no checkpoint, so every run counts as
    an instant failure and the abort would stop the sweep before the behaviour
    under test -- GPU pinning across the whole queue -- could be observed.
    """
    import scripts.sweep as sweep
    monkeypatch.setattr(sweep, "INSTANT_FAIL_ABORT", 10 ** 6)


def test_every_run_is_pinned_to_exactly_one_gpu(tmp_path, no_instant_abort):
    seen_dir = tmp_path / "seen"; seen_dir.mkdir()
    probe = tmp_path / "who.py"
    probe.write_text(
        "import os, pathlib, sys\n"
        f"pathlib.Path({str(seen_dir)!r}, sys.argv[1]).write_text(\n"
        "    os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))\n")
    runs = [(f"r{i}", [sys.executable, str(probe), f"r{i}"]) for i in range(9)]
    with pytest.raises(SystemExit):        # the probe writes no best.pth
        schedule(runs, gpus=[0, 1, 2], epochs=1, logdir=tmp_path / "s" / "logs",
                 dry=False)
    seen = {p.name: p.read_text() for p in seen_dir.iterdir()}
    assert len(seen) == 9, "not every run launched"
    assert set(seen.values()) == {"0", "1", "2"}
    assert "<unset>" not in seen.values(), "a run saw every GPU on the box"


def test_the_gpu_pool_recycles_evenly(tmp_path, no_instant_abort):
    """A GPU must return to the pool when its run ends. If it did not, the
    sweep would serialise onto whichever card happened to free first."""
    seen_dir = tmp_path / "seen"; seen_dir.mkdir()
    probe = tmp_path / "who.py"
    probe.write_text(
        "import os, pathlib, sys\n"
        f"pathlib.Path({str(seen_dir)!r}, sys.argv[1]).write_text(\n"
        "    os.environ['CUDA_VISIBLE_DEVICES'])\n")
    runs = [(f"r{i}", [sys.executable, str(probe), f"r{i}"]) for i in range(9)]
    with pytest.raises(SystemExit):
        schedule(runs, gpus=[0, 1, 2], epochs=1, logdir=tmp_path / "s" / "logs",
                 dry=False)
    counts = Counter(p.read_text() for p in seen_dir.iterdir())
    assert dict(counts) == {"0": 3, "1": 3, "2": 3}


def test_the_allocator_env_is_set_for_every_child(tmp_path):
    seen = tmp_path / "alloc.txt"
    probe = tmp_path / "alloc.py"
    probe.write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(seen)!r}).write_text(\n"
        "    os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>'))\n")
    with pytest.raises(SystemExit):
        schedule([("r0", [sys.executable, str(probe)])], gpus=[0], epochs=1,
                 logdir=tmp_path / "s" / "logs", dry=False)
    assert "expandable_segments" in seen.read_text()


# --------------------------------------------------------------------------
# the STOP flag
# --------------------------------------------------------------------------

def _stop_after(path, delay=2.0):
    threading.Thread(target=lambda: (time.sleep(delay), path.write_text("x")),
                     daemon=True).start()


def test_stop_drains_in_flight_work_and_exits_130(tmp_path):
    """The graceful halt: nothing new launches, everything already running
    finishes and earns its completion record, and the exit code tells
    hpec_pipeline.sh not to advance to evaluation."""
    runs_dir = tmp_path / "sess"; (runs_dir / "logs").mkdir(parents=True)
    slow = _slow_writer(tmp_path)
    runs = [(f"s{i}", [sys.executable, str(slow), str(runs_dir / f"s{i}")])
            for i in range(8)]
    stop_file = runs_dir / STOP_FILE
    _stop_after(stop_file)

    with pytest.raises(SystemExit) as excinfo:
        schedule(runs, gpus=[0, 1], epochs=1, logdir=runs_dir / "logs",
                 dry=False, stop_file=stop_file)
    assert excinfo.value.code == 130, "a stopped sweep must not look successful"

    launched = {p.stem for p in (runs_dir / "logs").glob("s*.log")}
    assert launched, "nothing ran before the flag appeared"
    assert len(launched) < 8, "the flag did not stop new launches"
    for name in launched:
        assert (runs_dir / name / "best.pth").exists(), f"{name} was killed mid-run"
        assert completion_path(runs_dir, name).exists(), f"{name} lost its record"


def test_a_queued_run_left_by_stop_has_no_partial_state(tmp_path):
    """Whatever STOP leaves behind must be resumable, not half-written."""
    runs_dir = tmp_path / "sess"; (runs_dir / "logs").mkdir(parents=True)
    slow = _slow_writer(tmp_path)
    runs = [(f"s{i}", [sys.executable, str(slow), str(runs_dir / f"s{i}")])
            for i in range(8)]
    stop_file = runs_dir / STOP_FILE
    _stop_after(stop_file)
    with pytest.raises(SystemExit):
        schedule(runs, gpus=[0, 1], epochs=1, logdir=runs_dir / "logs",
                 dry=False, stop_file=stop_file)
    launched = {p.stem for p in (runs_dir / "logs").glob("s*.log")}
    for name in (f"s{i}" for i in range(8)):
        if name in launched:
            continue
        assert not (runs_dir / name / "best.pth").exists()
        assert not completion_path(runs_dir, name).exists()


def test_a_sweep_that_finishes_normally_does_not_exit_130(tmp_path):
    runs_dir = tmp_path / "sess"; (runs_dir / "logs").mkdir(parents=True)
    writer = _writer(tmp_path)
    runs = [(f"s{i}", [sys.executable, str(writer), str(runs_dir / f"s{i}")])
            for i in range(3)]
    schedule(runs, gpus=[0], epochs=1, logdir=runs_dir / "logs", dry=False,
             stop_file=runs_dir / STOP_FILE)          # flag never created


def test_stop_is_reported_in_the_summary_line(tmp_path, capsys):
    runs_dir = tmp_path / "sess"; (runs_dir / "logs").mkdir(parents=True)
    slow = _slow_writer(tmp_path)
    runs = [(f"s{i}", [sys.executable, str(slow), str(runs_dir / f"s{i}")])
            for i in range(6)]
    stop_file = runs_dir / STOP_FILE
    _stop_after(stop_file)
    with pytest.raises(SystemExit):
        schedule(runs, gpus=[0, 1], epochs=1, logdir=runs_dir / "logs",
                 dry=False, stop_file=stop_file)
    out = capsys.readouterr().out
    assert "STOP flag seen" in out
    assert "STOPPED early by flag file" in out
    assert "still queued" in out, "the operator needs to know how much is left"


def test_resuming_after_stop_picks_up_exactly_where_it_left_off(tmp_path):
    """The operator flow the STOP flag exists for: halt a long sweep, then
    restart it. Anything already finished must not be retrained -- that is the
    whole point of stopping gracefully rather than killing it -- and nothing
    left queued may be skipped.

    The command must be IDENTICAL across both invocations. run_fingerprint
    covers argv, so changing the stub between the halt and the resume is a
    genuine protocol change and correctly invalidates the finished
    checkpoints; an earlier version of this test did exactly that and read the
    resulting retrain as a bug.
    """
    runs_dir = tmp_path / "sess"; (runs_dir / "logs").mkdir(parents=True)
    calls = tmp_path / "calls"; calls.mkdir()
    script = tmp_path / "run.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        f"log = pathlib.Path({str(calls)!r}, out.name)\n"
        "with log.open('a') as fh: fh.write('x')\n"
        "time.sleep(3)\n"
        "(out / 'best.pth').write_bytes(b'ckpt-' + sys.argv[1].encode())\n")
    names = [f"s{i}" for i in range(6)]
    runs = [(n, [sys.executable, str(script), str(runs_dir / n)]) for n in names]

    stop_file = runs_dir / STOP_FILE
    _stop_after(stop_file)
    with pytest.raises(SystemExit) as excinfo:
        schedule(runs, gpus=[0, 1], epochs=1, logdir=runs_dir / "logs",
                 dry=False, stop_file=stop_file)
    assert excinfo.value.code == 130
    before = {p.name: p.read_text() for p in calls.iterdir()}
    assert 0 < len(before) < len(names), "the stop did not leave work queued"

    stop_file.unlink()
    schedule(runs, gpus=[0, 1], epochs=1, logdir=runs_dir / "logs", dry=False,
             stop_file=stop_file)

    after = {p.name: p.read_text() for p in calls.iterdir()}
    retrained = [n for n, v in after.items() if len(v) > len(before.get(n, ""))
                 and n in before]
    assert not retrained, f"resume retrained finished work: {sorted(retrained)}"
    assert set(after) == set(names), "resume skipped runs the stop left queued"
    for name in names:
        assert completion_path(runs_dir, name).exists(), f"{name} never completed"
        assert len(after[name]) == 1, f"{name} was invoked {len(after[name])} times"


# --------------------------------------------------------------------------
# environment collapse: stop, do not grind through the matrix
# --------------------------------------------------------------------------

def _noop(tmp_path):
    """A stub run that fails the way a detached GPU fails: instantly, with no
    checkpoint. The scheduler turns 'exited 0 but wrote no best.pth' into a
    failure, so this is the cheapest faithful stand-in."""
    script = tmp_path / "noop.py"
    script.write_text("import sys\n")
    return script


def _launched(tmp_path):
    return sorted(p.stem for p in (tmp_path / "s" / "logs").glob("*.log"))


def test_a_streak_of_instant_failures_aborts_the_sweep(tmp_path, capsys):
    """The incident this exists for: box A's card detached between two
    sessions, and a 27-run sweep burned to run 24 in five minutes, every run
    failing at startup, reporting it only at the end."""
    noop = _noop(tmp_path)
    runs = [(f"r{i}", [sys.executable, str(noop), f"r{i}"]) for i in range(9)]
    with pytest.raises(SystemExit):
        schedule(runs, gpus=[0], epochs=1, logdir=tmp_path / "s" / "logs",
                 dry=False)
    launched = _launched(tmp_path)
    assert len(launched) == 3, f"kept launching after the streak: {launched}"
    out = capsys.readouterr().out
    assert "ABORTING" in out and "environment failure" in out


def test_a_slow_failure_does_not_count_toward_the_streak(tmp_path, monkeypatch):
    """One cell failing after real work is a model problem, not an environment
    one, and must not stop the other 26."""
    import scripts.sweep as sweep
    monkeypatch.setattr(sweep, "INSTANT_FAIL_SECONDS", 1.0)
    slow = tmp_path / "slowfail.py"
    slow.write_text("import time\ntime.sleep(2)\n")     # fails, but not instantly
    runs = [(f"r{i}", [sys.executable, str(slow), f"r{i}"]) for i in range(4)]
    with pytest.raises(SystemExit):
        schedule(runs, gpus=[0], epochs=1, logdir=tmp_path / "s" / "logs",
                 dry=False)
    assert len(_launched(tmp_path)) == 4, "a slow failure tripped the abort"


def test_a_success_resets_the_streak(tmp_path, monkeypatch):
    import scripts.sweep as sweep
    monkeypatch.setattr(sweep, "INSTANT_FAIL_ABORT", 2)
    noop, ok = _noop(tmp_path), _writer(tmp_path)
    # fail, fail would abort; fail, ok, fail, ok must not.
    runs = []
    for i in range(2):
        runs.append((f"bad{i}", [sys.executable, str(noop), f"bad{i}"]))
        out = str(tmp_path / "s" / f"good{i}")
        runs.append((f"good{i}", [sys.executable, str(ok), out]))
    with pytest.raises(SystemExit):
        schedule(runs, gpus=[0], epochs=1, logdir=tmp_path / "s" / "logs",
                 dry=False)
    assert len(_launched(tmp_path)) == 4, "a success did not reset the streak"


# --------------------------------------------------------------------------
# there is no sensible default GPU pool when there is no GPU
# --------------------------------------------------------------------------

def test_the_gpu_pool_is_the_visible_devices():
    from scripts.sweep import resolve_gpus
    assert resolve_gpus(1) == [0]
    assert resolve_gpus(4) == [0, 1, 2, 3]


def test_zero_visible_devices_is_refused_not_defaulted_to_gpu_zero():
    """`list(range(count)) or [0]` pinned CUDA_VISIBLE_DEVICES=0 on a box with
    no card, which is how every run came to fail at startup."""
    from scripts.sweep import resolve_gpus
    with pytest.raises(SystemExit) as excinfo:
        resolve_gpus(0)
    message = str(excinfo.value)
    assert "no CUDA device visible" in message
    assert "nvidia-smi" in message and "--gpus" in message


def test_a_dry_run_still_works_on_a_machine_with_no_gpu(tmp_path):
    """Inspecting the matrix is not the thing that needs a card.

    The refusal has to bite when a sweep would TRAIN on a device that is not
    there, and nowhere else -- otherwise it stops anyone reading the plan on a
    laptop, which is most of the time anyone reads it.
    """
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "sweep.py"),
         "--tests", "7", "--runs-dir", str(tmp_path / "s"), "--dry-run"],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "27 runs to go" in proc.stdout
    assert "no CUDA device visible" not in proc.stdout + proc.stderr
