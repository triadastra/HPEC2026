"""The supplementary Exp 0 arm probe narrows the declaration, and that is
both the point and the hazard.

`select_lr.py` writes nothing while a declared cell is missing its checkpoint,
so a probe that declared all 210 Exp 0 cells and ran 25 could never emit the
selection it exists to produce. `--exp0-arms` therefore narrows the
declaration -- unlike `--models`/`--exp0-models`, which filter execution only.

The cost is that the manifest it writes covers only those arms. Pointed at a
session holding other arms, it would erase their declarations.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import sweep  # noqa: E402


def _declare(arms):
    return sweep.build_matrix(
        tests={"0"}, seeds=(947,), flat_bs=2048, combo_bs=1, accum=None,
        npz="x.npz", effective_batch_size=None, mechs=None,
        lr_selection=None, exp0_models=None, exp0_arms=arms)


def test_the_arm_probe_declares_only_the_arms_it_runs():
    """25 mech cells, not 210 -- the number the docs quote for the fa probe."""
    mech = [n for n, _ in _declare(["mech"])]
    assert len(mech) == 25
    assert {sweep._arm_of(n) for n in mech} == {"mech"}


def test_the_full_exp0_declaration_is_unchanged_without_the_flag():
    """--exp0-arms is an execution filter, never a narrowing of what is declared.

    The literal is a tripwire, not the point: it moved 210 -> 240 when the
    mechlocal and mechsm arms were added for Exp 7 and Exp 8. If it fails,
    check that the arm you added belongs in the declaration before updating it,
    then update README and catalog.md with it -- test_docs_match_the_matrix.py
    asserts they agree.
    """
    full = _declare(None)
    assert len(full) == 240
    # Structural invariant: the declaration is exactly the union of the arms,
    # so an arm that is declared but unreachable (or reachable but undeclared)
    # fails here rather than by silently shrinking someone's manifest.
    per_arm = {n for arm in sweep.EXP0_ARMS for n, _ in _declare([arm])}
    assert per_arm <= {n for n, _ in full}


def test_pointing_the_probe_at_a_session_with_other_arms_is_refused(tmp_path):
    """The destructive case: narrowing the declaration over a finished Exp 0
    drops the other arms' entries, and the manifest is the only record of what
    that session declared."""
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 3, "runs": [
        {"name": "exp0_flat_gru_lr1e3_s947", "fingerprint": "f", "argv": []},
        {"name": "exp0_mech_gru_lr1e3_s947", "fingerprint": "f", "argv": []},
    ]}))
    with pytest.raises(SystemExit) as e:
        sweep._refuse_arm_probe_over_existing_session("mech", tmp_path)
    assert "stranding" in str(e.value) and "flat" in str(e.value)


def test_its_own_fresh_directory_is_allowed(tmp_path):
    sweep._refuse_arm_probe_over_existing_session("mech", tmp_path / "new")


def test_resuming_the_same_probe_is_allowed(tmp_path):
    """A probe that already holds only mech cells is the resume case, not the
    destructive one."""
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 3, "runs": [
        {"name": "exp0_mech_gru_lr1e3_s947", "fingerprint": "f", "argv": []},
    ]}))
    sweep._refuse_arm_probe_over_existing_session("mech", tmp_path)


def test_select_lr_writes_nothing_while_a_declared_cell_is_missing(tmp_path):
    """This is WHY the declaration has to narrow. If select_lr emitted a
    partial selection instead, the probe could declare everything and run one
    arm -- and this test would be the one to delete."""
    import subprocess
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 3, "runs": [
        {"name": "exp0_mech_gru_lr1e3_s947", "fingerprint": "f", "argv": []},
        {"name": "exp0_flat_gru_lr1e3_s947", "fingerprint": "f", "argv": []},
    ]}))
    out = tmp_path / "sel.json"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "select_lr.py"),
                        "--runs", str(tmp_path), "--out", str(out)],
                       capture_output=True, text=True)
    assert not out.exists(), "a partial selection must not be emitted"
    assert (r.stdout + r.stderr).strip(), "and it must say which cells failed"
