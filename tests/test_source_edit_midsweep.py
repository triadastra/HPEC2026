"""A source edit mid-sweep must not silently retrain what is already done.

The provenance fingerprint covers every .py/.yaml under src/, scripts/,
config/ and external/, so merging ANY branch -- even one that changes only
comments -- moves it. Runs finished under the old fingerprint then fail
`validate_completion` against a freshly declared manifest and are retrained.

That is correct behaviour and must stay. What these tests pin is the escape
hatch that makes a mid-sweep merge survivable: a cell the invocation is not
executing keeps its previous manifest entry, so restricting execution with
--only leaves finished work alone. Losing that would mean re-running the whole
matrix after any code change.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import sweep  # noqa: E402


def _plant(runs_dir, name, argv, fingerprint):
    """A finished run: manifest entry, checkpoint, completion record."""
    d = runs_dir / name
    (d / "logs").mkdir(parents=True, exist_ok=True)
    weights = b"weights"
    (d / "best.pth").write_bytes(weights)
    (d / "run_complete.json").write_text(json.dumps(
        {"name": name, "fingerprint": fingerprint,
         # validate_completion binds the record to the bytes on disk, so a
         # record without this is rejected regardless of the fingerprint.
         "checkpoint_sha256": hashlib.sha256(weights).hexdigest()}))
    return {"name": name, "argv": argv, "fingerprint": fingerprint}


def _manifest(runs_dir):
    return {e["name"]: e for e in
            json.loads((runs_dir / "manifest.json").read_text())["runs"]}


class _Launched(Exception):
    """Raised by the stubbed launcher once the manifest is on disk."""


def _declare(monkeypatch, sweep_dir, runs, declared):
    """Run schedule() far enough to write the manifest, then stop.

    The manifest is written before the first job is launched, and it is only
    written when dry is False -- so a dry run would leave us reading back the
    file the test itself planted. Stubbing the launcher gets the real manifest
    without training anything.
    """
    real = sweep.subprocess.Popen

    def _boom(*a, **kw):
        # Fingerprinting shells out on its own -- git for the tree identity,
        # and the platform module runs `file` to identify the interpreter.
        # Both happen before the manifest is built, so match the training
        # launcher by its argv rather than by its keywords.
        argv = a[0] if a else []
        if isinstance(argv, (list, tuple)) and any(
                str(x).endswith("train.py") for x in argv):
            raise _Launched
        return real(*a, **kw)
    monkeypatch.setattr(sweep.subprocess, "Popen", _boom)
    try:
        sweep.schedule(runs=runs, gpus=[0], epochs=200,
                       logdir=sweep_dir / "logs", dry=False,
                       declared_runs=declared)
    except _Launched:
        pass
    return _manifest(sweep_dir)


@pytest.fixture
def sweep_dir(tmp_path):
    runs = tmp_path / "sweep"
    (runs / "logs").mkdir(parents=True)
    return runs


ARGV = ["python", "scripts/train.py", "--model", "gru", "--seed", "947"]


def test_a_finished_run_left_out_of_the_run_set_keeps_its_provenance(
        sweep_dir, monkeypatch):
    """The escape hatch. Deploy new code, then --only the work that remains:
    the finished cells are not in the executing set, so they keep the entry
    they were trained under and stay valid."""
    old = _plant(sweep_dir, "done_cell", ARGV + ["--epochs", "200",
                 "--out-dir", str(sweep_dir / "done_cell")], "OLD_FINGERPRINT")
    (sweep_dir / "manifest.json").write_text(json.dumps(
        {"schema_version": 3, "runs": [old]}))

    declared = [("done_cell", ARGV), ("todo_cell", ARGV)]
    manifest = _declare(monkeypatch, sweep_dir, [("todo_cell", ARGV)], declared)

    kept = manifest["done_cell"]
    assert kept["fingerprint"] == "OLD_FINGERPRINT"
    from src.utils.run_manifest import validate_completion
    assert validate_completion(sweep_dir, kept) is None, (
        "the finished run no longer validates -- it would be retrained")


def test_a_finished_run_inside_the_run_set_is_retrained_after_a_source_edit(
        sweep_dir, monkeypatch):
    """The hazard, stated as a fact rather than a warning. Re-declaring a
    finished cell under a moved fingerprint invalidates its checkpoint. This
    is why the tree is frozen for the duration of a sweep."""
    old = _plant(sweep_dir, "done_cell", ARGV + ["--epochs", "200",
                 "--out-dir", str(sweep_dir / "done_cell")], "OLD_FINGERPRINT")
    (sweep_dir / "manifest.json").write_text(json.dumps(
        {"schema_version": 3, "runs": [old]}))

    declared = [("done_cell", ARGV)]
    entry = _declare(monkeypatch, sweep_dir, declared, declared)["done_cell"]
    assert entry["fingerprint"] != "OLD_FINGERPRINT"
    from src.utils.run_manifest import validate_completion
    assert validate_completion(sweep_dir, entry) is not None, (
        "a moved fingerprint must invalidate the old checkpoint")


def test_preservation_requires_the_command_to_be_unchanged(
        sweep_dir, monkeypatch):
    """Provenance is only carried over for a cell whose protocol is identical.
    A changed epoch count or batch size is a different experiment, and reusing
    its record would build a leaderboard out of two protocols."""
    old = _plant(sweep_dir, "done_cell",
                 ARGV + ["--epochs", "20", "--out-dir",
                         str(sweep_dir / "done_cell")], "OLD_FINGERPRINT")
    (sweep_dir / "manifest.json").write_text(json.dumps(
        {"schema_version": 3, "runs": [old]}))

    declared = [("done_cell", ARGV), ("todo_cell", ARGV)]
    manifest = _declare(monkeypatch, sweep_dir, [("todo_cell", ARGV)], declared)

    assert manifest["done_cell"]["fingerprint"] != "OLD_FINGERPRINT"
