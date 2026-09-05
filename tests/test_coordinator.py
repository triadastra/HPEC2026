"""Merging two GPU boxes' halves must not preserve anything stale.

The coordinator's whole job is to assemble one scoreable directory out of two
machines. Every failure mode here is the same shape: something that was true
at an earlier pull is still on disk, and evaluation treats it as current.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _coord(local):
    spec = importlib.util.spec_from_file_location(
        "coordinate", REPO / "tools" / "coordinate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOCAL = local
    return mod


class _Args:
    checkpoints = False


def _box(root, box, runs):
    d = root / f"box{box}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(
        {"runs": [{"name": r, "fingerprint": f"fp-{r}"} for r in runs]}))
    for r in runs:
        (d / r / "logs").mkdir(parents=True, exist_ok=True)
        (d / r / "logs" / "metrics.json").write_text("[]")
        (d / r / "run_complete.json").write_text("{}")
    return d


def _files(d):
    return sorted(p.name for p in d.rglob("*") if p.is_file())


def test_the_merge_is_the_union_of_both_halves(tmp_path, capsys):
    _box(tmp_path, "A", ["gru_a"])
    _box(tmp_path, "B", ["lstm_b"])
    m = _coord(tmp_path)
    m.merge(_Args())
    man = json.loads((tmp_path / "sweep" / "manifest.json").read_text())
    assert {e["name"] for e in man["runs"]} == {"gru_a", "lstm_b"}


def test_the_same_cell_completed_on_both_boxes_stops_the_merge(tmp_path):
    """Two provenances for one cell, and no principled way to say which the
    leaderboard reports. That must halt rather than resolve to whichever copy
    the loop happened to see last."""
    _box(tmp_path, "A", ["gru_a"])
    _box(tmp_path, "B", ["gru_a"])
    m = _coord(tmp_path)
    with pytest.raises(SystemExit) as e:
        m.merge(_Args())
    assert "BOTH boxes" in str(e.value)


def test_a_cell_declared_by_both_but_run_by_one_takes_the_finished_side(tmp_path):
    """Rebalancing makes declarations overlap on purpose: box B finishes its
    shard early and picks up a slice of box A's remaining runs, so both boxes
    declare the same test while only one executes each cell. Refusing here
    would make the rebalance unmergeable."""
    _box(tmp_path, "A", ["gru_a"])           # declared and complete on A
    b = _box(tmp_path, "B", ["gru_a"])       # declared on B, not run there
    (b / "gru_a" / "run_complete.json").unlink()
    m = _coord(tmp_path)
    m.merge(_Args())                          # must not raise

    entry = _manifest_entry(tmp_path, "gru_a")
    assert entry["fingerprint"] == "fp-gru_a"
    assert (tmp_path / "sweep" / "gru_a" / "run_complete.json").exists(), (
        "the finished side's completion record has to survive the merge")


def _manifest_entry(root, name):
    man = json.loads((root / "sweep" / "manifest.json").read_text())
    return next(e for e in man["runs"] if e["name"] == name)


def test_a_later_checkpoint_pull_reaches_the_merged_tree(tmp_path):
    """Evidence-only first, then --checkpoints. Treating an existing target
    directory as proof it is current left best.pth behind, and official
    evaluation then reported the checkpoint missing."""
    a = _box(tmp_path, "A", ["gru_a"])
    m = _coord(tmp_path)
    m.merge(_Args())
    merged = tmp_path / "sweep" / "gru_a"
    assert "best.pth" not in _files(merged)

    (a / "gru_a" / "best.pth").write_bytes(b"ckpt")
    m.merge(_Args())
    assert "best.pth" in _files(merged)


def test_a_cleared_completion_record_does_not_survive_the_merge(tmp_path):
    """sweep.py deletes run_complete.json before relaunching a run. If the
    rerun then fails, a surviving copy here would still pass provenance
    validation and the previous run would be scored as the current result."""
    a = _box(tmp_path, "A", ["gru_a"])
    m = _coord(tmp_path)
    m.merge(_Args())
    merged = tmp_path / "sweep" / "gru_a"
    assert "run_complete.json" in _files(merged)

    (a / "gru_a" / "run_complete.json").unlink()
    m.merge(_Args())
    assert "run_complete.json" not in _files(merged)


def test_the_pull_mirrors_deletions_and_protects_excluded_files():
    """--delete is what removes a cleared completion record locally. Excluded
    files are protected from it by default, so an evidence-only pull still
    cannot delete checkpoints it deliberately did not fetch."""
    source = (REPO / "tools" / "coordinate.py").read_text()
    assert '"--delete"' in source
    assert "--delete-excluded" not in source, (
        "that would let an evidence-only pull delete checkpoints")


def test_a_failed_transfer_is_not_reported_as_a_refresh():
    """Printing the count already on disk after a failed rsync makes a broken
    transfer read as success, and lets merge and verify run on stale data."""
    source = (REPO / "tools" / "coordinate.py").read_text()
    pull = source[source.index("def pull"):source.index("def merge")]
    assert "check=False" not in pull, "a failed rsync must not be ignored"
    assert "sys.exit(" in pull, "it has to stop rather than continue"
    assert "attempt" in pull, "and retry first -- AutoDL drops connections"


class _PullArgs:
    checkpoints = False
    force_delete = False


def _fake_rsync(deletions, calls):
    """Stand in for rsync: report `deletions` files as pending removal on the
    dry run, and record whether the real transfer was ever attempted."""
    class _R:
        returncode = 0
        stdout = "".join(f"*deleting   run{i}/run_complete.json\n"
                         for i in range(deletions))

    def run(cmd, **kw):
        calls.append(cmd)
        return _R()
    return run


def test_a_wiped_remote_does_not_take_the_local_copy_with_it(tmp_path, monkeypatch):
    """A released AutoDL instance can come back with the sweep path present and
    empty. Mirroring that would delete evidence this machine may hold alone."""
    m = _coord(tmp_path)
    calls = []
    monkeypatch.setattr(m.subprocess, "run", _fake_rsync(400, calls))
    monkeypatch.setenv("VMPW_A", "x")
    monkeypatch.setenv("VMPW_B", "x")

    with pytest.raises(SystemExit) as e:
        m.pull(_PullArgs())
    assert "only one left" in str(e.value)
    assert all("--dry-run" in c for c in calls), "it deleted before it counted"


def test_the_handful_of_deletions_a_relaunch_causes_still_goes_through(
        tmp_path, monkeypatch):
    m = _coord(tmp_path)
    calls = []
    monkeypatch.setattr(m.subprocess, "run", _fake_rsync(3, calls))
    monkeypatch.setenv("VMPW_A", "x")
    monkeypatch.setenv("VMPW_B", "x")

    m.pull(_PullArgs())
    assert any("--dry-run" not in c for c in calls), "the real pull never ran"


def test_force_delete_is_the_only_way_past_the_gate(tmp_path, monkeypatch):
    m = _coord(tmp_path)
    calls = []
    monkeypatch.setattr(m.subprocess, "run", _fake_rsync(400, calls))
    monkeypatch.setenv("VMPW_A", "x")
    monkeypatch.setenv("VMPW_B", "x")

    args = _PullArgs()
    args.force_delete = True
    m.pull(args)
    assert any("--dry-run" not in c for c in calls)
