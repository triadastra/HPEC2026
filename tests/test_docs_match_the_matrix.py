"""The documented run counts must equal what sweep.py actually emits.

Four documents carried four different totals for the same matrix -- catalog.md
said 765, RETRAIN.md said 657 and 513, base.yaml said 657, README said
something else again -- because each was written against a different roster and
none was checked against the code. A reader deciding whether they can afford
the sweep was reading a number nobody had verified in months.

So the numbers are asserted here rather than maintained by hand.
"""

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.sweep import (DEFAULT_SEEDS, MECHS, RETIRED_TESTS, RETRAIN_SETS,
                           build_exp0_matrix, build_exp7_matrix, build_exp8_matrix,
                           build_matrix,
                           select)

KW = dict(accum=14, effective_batch_size=28292)
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


# --------------------------------------------------------------------------
# catalog.md is the declared authority
# --------------------------------------------------------------------------

def test_catalog_matrix_summary_matches_the_code(actual):
    text = (REPO / "catalog.md").read_text()
    documented = _table_counts(
        text, r"^\|\s*Test ([\d.]+)[^|]*\|[^|]*\|\s*(\d+)\s*\|$")
    assert documented, "could not parse the catalog matrix summary"
    assert documented == actual, (
        f"catalog.md disagrees with sweep.py: documented={documented} "
        f"actual={actual}")


def test_catalog_total_matches_the_sum(actual):
    text = (REPO / "catalog.md").read_text()
    m = re.search(r"\*\*Total to train\*\*\s*\|\s*\|\s*\*\*(\d+)\*\*", text)
    assert m, "catalog.md has no parseable total"
    assert int(m.group(1)) == sum(actual.values()) == _count(LIVE_TESTS)


# --------------------------------------------------------------------------
# README repeats the same table and must not drift from it
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# the retrain slices
# --------------------------------------------------------------------------

def test_retrain_slice_sizes_are_documented_correctly():
    runs = build_matrix(set(LIVE_TESTS), DEFAULT_SEEDS, 2048, 1, **KW)
    sizes = {name: len(select(runs, pattern))
             for name, pattern in RETRAIN_SETS.items()}
    text = (REPO / "RETRAIN.md").read_text()
    for name, size in sizes.items():
        assert re.search(rf"\|\s*`{name}`\s*\|\s*{size}\s*\|", text), (
            f"RETRAIN.md does not document {name} = {size} runs")


# --------------------------------------------------------------------------
# the optional arms cost what the docs say
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mech", ["aca", "fa_local"])
def test_optional_arms_cost_the_documented_amount(mech):
    base = _count(LIVE_TESTS)
    extended = len(build_matrix(set(LIVE_TESTS), DEFAULT_SEEDS, 2048, 1,
                                mechs=list(MECHS) + [mech], **KW))
    assert extended - base == 144, (
        f"{mech} now costs {extended - base} runs; README and RETRAIN.md both "
        "say +144")
    for doc in ("README.md", "RETRAIN.md"):
        assert "+144 runs" in (REPO / doc).read_text()


def test_the_hybrid_identity_arm_costs_the_documented_amount():
    """The opt-in identity-aware hybrid arm (--hybrid-identity) mirrors
    Test 4 for the Test 6 mamba2/mamba3 hosts."""
    base = _count(LIVE_TESTS)
    extended = len(build_matrix(set(LIVE_TESTS), DEFAULT_SEEDS, 2048, 1,
                                hybrid_identity=True, **KW))
    assert extended - base == 72, (
        f"--hybrid-identity now costs {extended - base} runs; catalog.md and "
        "future_work.md both say +72")
    for doc in ("catalog.md", "future_work.md"):
        assert "+72 runs" in (REPO / doc).read_text(), (
            f"{doc} does not document the +72-run hybrid-identity arm")


def test_the_declared_and_extended_totals_in_retrain_are_right():
    base = _count(LIVE_TESTS)
    extended = len(build_matrix(set(LIVE_TESTS), DEFAULT_SEEDS, 2048, 1,
                                mechs=list(MECHS) + ["aca"], **KW))
    text = (REPO / "RETRAIN.md").read_text()
    assert f"{extended} -> {base}" in text, "the fa_local removal figure is stale"
    assert f"{base} -> {extended}" in text, "the aca addition figure is stale"


# --------------------------------------------------------------------------
# no document may still advertise the retired test
# --------------------------------------------------------------------------

DOCS = ["README.md", "catalog.md", "RETRAIN.md", "PLAN.md",
        "config/base.yaml", "scripts/hpec_pipeline.sh"]


@pytest.mark.parametrize("doc", DOCS)
def test_no_document_tells_the_reader_to_run_a_retired_test(doc):
    text = (REPO / doc).read_text()
    for retired in RETIRED_TESTS:
        assert f"TESTS=1,1.1,2,3,4,{retired},6" not in text
        assert f"--tests {retired}\n" not in text
        assert f"--tests {retired} " not in text


@pytest.mark.parametrize("doc", DOCS)
def test_no_document_still_quotes_a_stale_total(doc):
    """The specific wrong numbers that were in circulation."""
    text = (REPO / doc).read_text()
    for stale in ("765", "657"):
        assert stale not in text, f"{doc} still quotes the stale total {stale}"


def test_catalog_explains_the_retirement_rather_than_dropping_it():
    """Silently deleting the section would leave readers of the paper's
    stability finding with no idea where it came from."""
    text = (REPO / "catalog.md").read_text()
    assert "Test 5" in text and "retired" in text
    assert "future_work.md" in text


# --------------------------------------------------------------------------
# future_work.md makes checkable claims; check them
# --------------------------------------------------------------------------

FUTURE_WORK = REPO / "future_work.md"


def test_future_work_exists_and_covers_the_retired_test():
    text = FUTURE_WORK.read_text()
    assert "Test 5" in text
    assert "retired" in text.lower()


def test_retired_emitter_is_preserved_in_the_release_fixture():
    import hashlib
    import json
    fixture = REPO / "tests/fixtures/retired_sweep.py.txt"
    metadata = json.loads(fixture.with_name("retired_sweep.json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == metadata["sha256"]
    assert metadata["source_commit"] in FUTURE_WORK.read_text()
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


def test_the_nd_run_count_future_work_quotes_is_right():
    """It argues from the size of the unsearched arm; the number must hold."""
    runs = build_matrix(set(LIVE_TESTS), DEFAULT_SEEDS, 2048, 1, **KW)
    nd = [n for n, argv in runs if "--combo-encoder" in argv]
    assert f"**{len(nd)} multidimensional runs**" in FUTURE_WORK.read_text(), (
        f"future_work.md should quote {len(nd)} multidimensional runs")


def test_future_work_quotes_the_real_exp0_size():
    size = len(build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292))
    assert f"{size} truncated runs" in FUTURE_WORK.read_text()


def test_the_off_roster_arms_are_documented_with_their_real_cost():
    text = FUTURE_WORK.read_text()
    for mech in ("fa_local", "aca"):
        assert mech in text
        extended = len(build_matrix(set(LIVE_TESTS), DEFAULT_SEEDS, 2048, 1,
                                    mechs=list(MECHS) + [mech], **KW))
        assert f"+{extended - _count(LIVE_TESTS)} runs" in text
