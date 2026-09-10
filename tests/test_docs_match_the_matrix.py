"""Check the public README against the live experiment generator."""
import re
import sys
from pathlib import Path
import pytest
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.sweep import (DEFAULT_SEEDS, MECHS, build_exp0_matrix,
                           build_exp7_matrix, build_exp8_matrix, build_matrix)
KW = dict(accum=15, effective_batch_size=30087)
LIVE_TESTS = ["1", "1.1", "2", "3", "4", "6"]


def _count(tests):
    return len(build_matrix(set(tests), DEFAULT_SEEDS, 2048, 1, **KW))


@pytest.fixture(scope="module")
def actual():
    return {t: _count([t]) for t in LIVE_TESTS}


def _table_counts(text, pattern):
    """{test id: documented count} from a markdown table."""
    out = {}
    for line in text.splitlines():
        m = re.match(pattern, line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def test_readme_test_table_matches_the_code(actual):
    text = (REPO / "README.md").read_text()
    documented = _table_counts(text, r"^\|\s*([\d.]+)\s*\|[^|]*\|\s*(\d+)\s*\|$")
    exp0 = documented.pop("0", None)
    assert exp0 == len(build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)), (
        "README's Exp 0 row disagrees with build_exp0_matrix")
    # Exp 7 and Exp 8 are popped for the same reason as Exp 0: each runs alone,
    # in its own --runs-dir, and none of them is among the declared runs
    # `actual` counts. The rows still have to track the code, or an arm's size
    # becomes another number nobody verified -- which is what this file exists
    # to prevent.
    for tid, builder in (("7", build_exp7_matrix), ("8", build_exp8_matrix)):
        documented_size = documented.pop(tid, None)
        assert documented_size == len(builder(DEFAULT_SEEDS, 1, "x.npz")), (
            f"README's Exp {tid} row disagrees with build_exp{tid}_matrix")
    assert documented == actual


@pytest.mark.parametrize("mech", ["aca", "fa_local"])
def test_optional_arms_cost_the_documented_amount(mech):
    base = _count(LIVE_TESTS)
    extended = len(build_matrix(set(LIVE_TESTS), DEFAULT_SEEDS, 2048, 1,
                                mechs=list(MECHS) + [mech], **KW))
    assert extended - base == 144, (
        f"{mech} now costs {extended - base} runs; README "
        "say +144")
    for doc in ("README.md",):
        assert "+144 runs" in (REPO / doc).read_text()


def test_retired_emitter_is_preserved_in_the_release_fixture():
    import hashlib
    import json
    fixture = REPO / "tests/fixtures/retired_sweep.py.txt"
    metadata = json.loads(fixture.with_name("retired_sweep.json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == metadata["sha256"]
    assert metadata["source_commit"] in (REPO / "README.md").read_text()
    assert "want_m3lr" in fixture.read_text()
    assert "want_m3lr" not in (REPO / "scripts/sweep.py").read_text()


def test_already_collected_test5_runs_remain_scoreable():
    """The stability finding is reported from runs already on disk, so
    evaluate.py must still parse their names."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate_for_docs", REPO / "scripts" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    cfg = evaluate.parse_run("mamba3_embeddings_1d_lr1e3_s947")
    assert cfg is not None, "old Test 5 checkpoints became unscoreable"
    assert cfg["variant"] == "embeddings_lr1e3"
    assert cfg["build_variant"] == "embeddings"
