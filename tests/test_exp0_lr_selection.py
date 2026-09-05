"""Exp 0 end-to-end: probe matrix -> lr_selection.json -> injection.

The learning rate is the one knob that decides whether the headline
flat-vs-multidimensional comparison is fair, so every stage here is tested for
what it REFUSES as much as for what it selects.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.sweep import (DEFAULT_SEEDS, DIVERGED_FILE, EXP0_ARMS,
                           EXP0_CHECK_BASE, EXP0_LR_GRID,
                           FLAT_MODELS, ND_BACKBONES, RETIRED_TESTS, _lr_args,
                           build_exp0_matrix, build_matrix, exp0_nd_variant,
                           lr_tag, rolling_aggregate_folds)
from src.utils.run_manifest import file_sha256


def _load_select_lr():
    spec = importlib.util.spec_from_file_location(
        "select_lr", REPO / "scripts" / "select_lr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select_lr = _load_select_lr()


# --------------------------------------------------------------------------
# fixture helpers: a synthetic Exp 0 session that passes validate_completion
# --------------------------------------------------------------------------

def _run(runs_dir, name, lr, curve=None, diverged=False):
    """Materialise one Exp 0 run directory and return its manifest entry."""
    d = runs_dir / name
    (d / "logs").mkdir(parents=True, exist_ok=True)
    entry = {"name": name, "fingerprint": f"fp-{name}",
             "argv": [sys.executable, "train.py", "--lr", lr]}
    if diverged:
        (d / DIVERGED_FILE).write_text(json.dumps({"loss": "nan", "epoch": 0}))
        return entry
    (d / "best.pth").write_bytes(b"ckpt-" + name.encode())
    (d / "logs" / "metrics.json").write_text(json.dumps([
        {"epoch": i, "phase": "train", "train_loss": v, "val_loss": v}
        for i, v in enumerate(curve)
    ]))
    (d / "run_complete.json").write_text(json.dumps({
        "fingerprint": entry["fingerprint"],
        "checkpoint_sha256": file_sha256(d / "best.pth"),
    }))
    return entry


def _session(tmp_path, spec):
    """spec: {(arm, model, seed): {lr: curve|"diverged"}} -> runs_dir."""
    runs_dir = tmp_path / "exp0"
    runs_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for (arm, model, seed), rates in spec.items():
        for lr, curve in rates.items():
            name = f"exp0_{arm}_{model}_{lr_tag(lr)}_s{seed}"
            entries.append(_run(runs_dir, name, lr,
                                curve=None if curve == "diverged" else curve,
                                diverged=curve == "diverged"))
    (runs_dir / "manifest.json").write_text(json.dumps({"runs": entries}))
    return runs_dir


SLOW = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.50, 0.50, 0.50]
FAST = [0.9, 0.7, 0.5, 0.4, 0.35, 0.30, 0.28, 0.26, 0.25, 0.25]   # better at
                                                                  # every prefix


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def test_selects_the_rate_with_the_best_validation_curve(tmp_path, capsys):
    runs = _session(tmp_path, {
        ("flat", "gru", 947): {"1e-4": SLOW, "1e-3": FAST},
        ("nd", "gru", 947): {"1e-4": FAST, "1e-3": SLOW},
    })
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert payload["selected"] == {"flat": {"gru": "1e-3"}, "nd": {"gru": "1e-4"}}
    assert payload["official"] is True
    assert payload["schema_version"] == select_lr.SCHEMA_VERSION


def test_selection_ignores_test_set_artifacts_lying_next_to_the_curve(tmp_path):
    """Selecting on test metrics is the failure this stage exists to prevent.

    Every run directory is given a test-set result that contradicts its
    validation curve. The selection must not move.
    """
    runs = _session(tmp_path, {("flat", "gru", 947): {"1e-4": SLOW, "1e-3": FAST}})
    for name, better in (("exp0_flat_gru_lr1e4_s947", 0.001),
                         ("exp0_flat_gru_lr1e3_s947", 0.999)):
        (runs / name / "results.csv").write_text(f"model,test_mse\ngru,{better}\n")
        (runs / name / "logs" / "test_metrics.json").write_text(
            json.dumps({"test_loss": better}))
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    # 1e-4 has the (planted) better test score; val_loss still says 1e-3.
    assert payload["selected"]["flat"]["gru"] == "1e-3"
    assert "val_loss" in payload["rule"]


def test_select_lr_does_not_import_the_test_set_scorer():
    source = (REPO / "scripts" / "select_lr.py").read_text()
    assert "import evaluate" not in source
    assert "from scripts.evaluate" not in source


def test_multi_seed_curves_are_averaged_not_overwritten(tmp_path):
    """Per-seed winners disagree; the mean decides. Keying by (arm,model,lr)
    without accumulating would silently let the last seed read win."""
    # seed 947 prefers 1e-4, seed 732 prefers 1e-3, and the MEAN prefers 1e-3.
    a947 = [0.30] * 10          # 1e-4
    b947 = [0.40] * 10          # 1e-3
    a732 = [0.90] * 10          # 1e-4  -> mean 0.60
    b732 = [0.42] * 10          # 1e-3  -> mean 0.41  (winner)
    runs = _session(tmp_path, {
        ("flat", "gru", 947): {"1e-4": a947, "1e-3": b947},
        ("flat", "gru", 732): {"1e-4": a732, "1e-3": b732},
    })
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert payload["selected"]["flat"]["gru"] == "1e-3"
    assert payload["seeds"] == [732, 947]
    best = payload["diagnostics"]["flat/gru"]["best_val"]
    assert best["1e-4"] == pytest.approx(0.60)
    assert best["1e-3"] == pytest.approx(0.41)


def test_ties_break_toward_the_smaller_rate(tmp_path):
    runs = _session(tmp_path, {("flat", "gru", 947): {"1e-4": SLOW, "1e-3": SLOW}})
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert payload["selected"]["flat"]["gru"] == "1e-4"


# --------------------------------------------------------------------------
# what it refuses
# --------------------------------------------------------------------------

def test_a_cell_missing_a_candidate_rate_is_refused(tmp_path, capsys):
    runs = _session(tmp_path, {
        ("flat", "gru", 947): {"1e-4": SLOW, "1e-3": FAST},
        ("flat", "lstm", 947): {"1e-4": SLOW},        # lost 1e-3 to a crash
    })
    assert select_lr.main(["--runs", str(runs)]) == 1
    assert "missing rates" in capsys.readouterr().out
    assert not (runs / "lr_selection.json").exists()
    assert (runs / "lr_selection_failures.json").exists()


def test_uneven_seed_coverage_is_refused(tmp_path, capsys):
    runs = _session(tmp_path, {
        ("flat", "gru", 947): {"1e-4": SLOW, "1e-3": FAST},
        ("flat", "gru", 732): {"1e-4": SLOW},          # 1e-3 has one fewer seed
    })
    assert select_lr.main(["--runs", str(runs)]) == 1
    assert "uneven seed coverage" in capsys.readouterr().out


def test_a_run_without_a_completion_record_is_refused(tmp_path, capsys):
    runs = _session(tmp_path, {("flat", "gru", 947): {"1e-4": SLOW, "1e-3": FAST}})
    (runs / "exp0_flat_gru_lr1e3_s947" / "run_complete.json").unlink()
    assert select_lr.main(["--runs", str(runs)]) == 1
    assert "run_complete.json" in capsys.readouterr().out


def test_a_tampered_checkpoint_is_refused(tmp_path, capsys):
    runs = _session(tmp_path, {("flat", "gru", 947): {"1e-4": SLOW, "1e-3": FAST}})
    (runs / "exp0_flat_gru_lr1e3_s947" / "best.pth").write_bytes(b"different")
    assert select_lr.main(["--runs", str(runs)]) == 1
    assert "checkpoint hash does not match" in capsys.readouterr().out


def test_an_unstable_ranking_is_refused_unless_explicitly_allowed(tmp_path, capsys):
    # crossover: 1e-4 leads early, 1e-3 overtakes it late.
    early_leader = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
    late_leader = [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.05, 0.05, 0.05]
    spec = {("flat", "gru", 947): {"1e-4": early_leader, "1e-3": late_leader}}
    runs = _session(tmp_path, spec)
    assert select_lr.main(["--runs", str(runs)]) == 1
    assert "unstable ranking" in capsys.readouterr().out

    runs2 = _session(tmp_path / "again", spec)
    assert select_lr.main(["--runs", str(runs2), "--allow-unstable"]) == 0
    payload = json.loads((runs2 / "lr_selection.json").read_text())
    assert payload["official"] is False, "unofficial output must say so"
    assert payload["diagnostics"]["flat/gru"]["ranking_stable"] is False


def test_the_stability_check_is_not_vacuous_on_short_curves(tmp_path):
    """A window longer than the curve must still compare a real midpoint.

    Unclamped, `mid` would land past the end of every curve, the midpoint and
    endpoint rankings would see identical data, and every cell would pass the
    stability check for free.
    """
    early_leader = [0.10] * 8
    late_leader = [0.90] * 6 + [0.05, 0.05]
    runs = _session(tmp_path, {
        ("flat", "gru", 947): {"1e-4": early_leader, "1e-3": late_leader}})
    assert select_lr.main(["--runs", str(runs), "--window", "25"]) == 1


# --------------------------------------------------------------------------
# divergence is a result, not a gap
# --------------------------------------------------------------------------

def test_a_diverged_rate_counts_as_covered_and_is_reported(tmp_path, capsys):
    runs = _session(tmp_path, {
        ("flat", "gru", 947): {"1e-4": SLOW, "1e-3": FAST, "1e-2": "diverged"},
    })
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert payload["selected"]["flat"]["gru"] == "1e-3"
    assert payload["diverged"] == {"flat/gru": ["1e-2"]}
    assert "diverged (tried, did not train)" in capsys.readouterr().out


def test_a_grid_endpoint_winner_is_flagged(tmp_path, capsys):
    runs = _session(tmp_path, {("flat", "gru", 947): {"1e-4": FAST, "1e-3": SLOW}})
    assert select_lr.main(["--runs", str(runs)]) == 0
    assert "grid endpoints won" in capsys.readouterr().out
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert payload["diagnostics"]["flat/gru"]["at_grid_endpoint"] is True


# --------------------------------------------------------------------------
# the dimensionality-transfer assumption
# --------------------------------------------------------------------------

def test_dim_probes_are_reported_but_never_selected(tmp_path):
    runs = _session(tmp_path, {
        ("nd", "s4nd", 947): {"1e-4": SLOW, "1e-3": FAST},
        ("dim2", "s4nd", 947): {"1e-4": SLOW, "1e-3": FAST},
        ("dim4", "s4nd", 947): {"1e-4": SLOW, "1e-3": FAST},
    })
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert set(payload["selected"]) == {"nd"}, "dim probes are not an arm"
    assert payload["dim_transfer"]["models"]["s4nd"]["consistent"] is True


def test_disagreeing_dim_probes_raise_the_transfer_warning(tmp_path):
    runs = _session(tmp_path, {
        ("nd", "s4nd", 947): {"1e-4": SLOW, "1e-3": FAST},
        ("dim2", "s4nd", 947): {"1e-4": SLOW, "1e-3": FAST},   # picks 1e-3
        ("dim4", "s4nd", 947): {"1e-4": FAST, "1e-3": SLOW},   # picks 1e-4
    })
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert payload["dim_transfer"]["models"]["s4nd"]["consistent"] is False
    assert any("dim-transfer WARNING" in n
               for n in payload["dim_transfer"]["notes"])


def test_dim_probes_are_compared_against_the_nd_rate_they_transfer(tmp_path):
    """dim2 == dim4 != nd is a FAILED transfer. The rate applied at 2-D and
    4-D is the 3-D selection, so the spot checks agreeing with each other must
    not read as "ok" while both disagree with the rate they will train at."""
    runs = _session(tmp_path, {
        ("nd", "s4nd", 947): {"1e-4": FAST, "1e-3": SLOW},     # selects 1e-4
        ("dim2", "s4nd", 947): {"1e-4": SLOW, "1e-3": FAST},   # prefers 1e-3
        ("dim4", "s4nd", 947): {"1e-4": SLOW, "1e-3": FAST},   # prefers 1e-3
    })
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    verdict = payload["dim_transfer"]["models"]["s4nd"]
    assert verdict["consistent"] is False
    assert verdict["rates"]["nd"] == "1e-4", "the 3-D selection must be in the comparison"
    assert any("dim-transfer WARNING" in n
               for n in payload["dim_transfer"]["notes"])


# --------------------------------------------------------------------------
# injection into the main matrix
# --------------------------------------------------------------------------

def _full_selection(rate="7e-4"):
    return {"schema_version": 1,
            "selected": {"flat": {m: rate for m in FLAT_MODELS},
                         "agg": {m: rate for m in FLAT_MODELS},
                         "roll": {m: rate for m in FLAT_MODELS},
                         "nd": {m: rate for m in ND_BACKBONES},
                         "mech": {m: rate for m in ND_BACKBONES}}}


def test_every_run_of_the_main_matrix_carries_the_selected_rate():
    runs = build_matrix({"1", "1.1", "2", "3", "4", "6"}, DEFAULT_SEEDS, 2048, 1,
                        accum=14, effective_batch_size=28292,
                        lr_selection=_full_selection())
    assert runs, "matrix must not be empty"
    for name, argv in runs:
        assert argv.count("--lr") == 1, f"{name}: expected exactly one --lr"
        assert argv[argv.index("--lr") + 1] == "7e-4", name


def test_the_aggregate_and_rolling_arms_are_covered_too():
    """Test 1.1 is the arm most easily forgotten: it is built in its own loop."""
    runs = build_matrix({"1.1"}, [947], 2048, 1, accum=14,
                        effective_batch_size=28292,
                        lr_selection=_full_selection("2e-4"))
    assert runs
    assert all(a[a.index("--lr") + 1] == "2e-4" for _, a in runs)


def test_the_rolling_arm_draws_its_rate_from_the_rolling_probe():
    """Not from the fixed-split aggregate arm. Fold 2020 trains on about half
    the history the fixed split gets, so a rate chosen there does not transfer
    for free."""
    selection = {"schema_version": 1,
                 "selected": {"agg": {m: "9e-9" for m in FLAT_MODELS},
                              "roll": {m: "2e-4" for m in FLAT_MODELS}}}
    rolling = build_matrix({"1.1"}, [947], 2048, 1, accum=14,
                           effective_batch_size=28292, lr_selection=selection)
    assert rolling
    assert all(a[a.index("--lr") + 1] == "2e-4" for _, a in rolling)

    fixed = build_matrix({"1"}, [947], 2048, 1, accum=14,
                         effective_batch_size=28292, lr_selection=selection)
    assert all(a[a.index("--lr") + 1] == "9e-9" for _, a in fixed)


def test_a_selection_without_the_rolling_arm_refuses_test_1_1():
    """Before the rolling probe existed, Test 1.1 silently inherited the
    fixed-split rate. It must now fail closed instead."""
    partial = {"schema_version": 1,
               "selected": {"agg": {m: "1e-3" for m in FLAT_MODELS}}}
    with pytest.raises(SystemExit) as excinfo:
        build_matrix({"1.1"}, [947], 2048, 1, accum=14,
                     effective_batch_size=28292, lr_selection=partial)
    assert "roll/" in str(excinfo.value)


def test_the_rolling_probe_reproduces_the_fold_contract():
    """A rate chosen under a different data contract than the runs it selects
    for does not transfer -- the mistake the inherited fixed-split rate made."""
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    roll = [a for n, a in probe if n.startswith("exp0_roll_")]
    assert roll, "Exp 0 declares no rolling probe"
    folds = rolling_aggregate_folds()
    for argv in roll:
        for flag in ("--aggregate", "--refit-normalization",
                     "--fresh-model-session"):
            assert flag in argv, f"rolling probe missing {flag}"
        assert argv[argv.index("--train-end") + 1] == str(folds[0]["train_end"])
        assert argv[argv.index("--test-end") + 1] == str(folds[0]["test_end"])


def test_the_rolling_probe_uses_the_shortest_fold():
    """The hardest case: fewest optimizer steps for a too-high rate to recover
    from. Probing the easiest fold would flatter every rate."""
    folds = rolling_aggregate_folds()
    assert folds[0]["train_end"] == min(f["train_end"] for f in folds)
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    roll = [a for n, a in probe if n.startswith("exp0_roll_")][0]
    assert roll[roll.index("--train-end") + 1] == str(folds[0]["train_end"])


def test_the_fold_length_transfer_check_uses_the_longest_fold():
    folds = rolling_aggregate_folds()
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    check = [a for n, a in probe if n.startswith("exp0_rollend_")]
    assert check, "no fold-length transfer check declared"
    for argv in check:
        assert argv[argv.index("--train-end") + 1] == str(folds[-1]["train_end"])


def test_the_rolling_probe_uses_the_same_five_rates_as_every_other_arm():
    """The whole point of Exp 0: one grid, every arm, no exceptions."""
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    rates = {a[a.index("--lr") + 1] for n, a in probe if n.startswith("exp0_roll_")}
    assert rates == set(EXP0_LR_GRID)


def test_the_transfer_check_arms_never_produce_a_selected_rate(tmp_path):
    """rollend and dim* exist to answer "does it transfer?", not to select."""
    runs = _session(tmp_path, {
        ("roll", "gru", 947): {"1e-4": SLOW, "1e-3": FAST},
        ("rollend", "gru", 947): {"1e-4": SLOW, "1e-3": FAST},
    })
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert set(payload["selected"]) == {"roll"}
    assert payload["transfer_checks"]["rollend"]["models"]["gru"]["consistent"] is True


def test_a_rate_that_does_not_survive_the_longest_fold_raises_a_warning(tmp_path):
    runs = _session(tmp_path, {
        ("roll", "mamba3", 947): {"1e-4": SLOW, "1e-3": FAST},    # picks 1e-3
        ("rollend", "mamba3", 947): {"1e-4": FAST, "1e-3": SLOW},  # picks 1e-4
    })
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    check = payload["transfer_checks"]["rollend"]
    transfer = check["models"]["mamba3"]
    assert transfer["consistent"] is False
    assert transfer["base"] == "1e-3", "the shortest fold selected this"
    assert transfer["check"] == "1e-4", "the longest fold disagreed"
    assert check["base_arm"] == "roll"
    assert any("rollend-transfer WARNING" in n for n in check["notes"])


def test_without_a_selection_no_run_gets_an_lr_flag():
    runs = build_matrix({"1", "1.1", "2", "3", "4", "6"}, [947], 2048, 1,
                        accum=14, effective_batch_size=28292)
    assert runs
    assert not any("--lr" in argv for _, argv in runs)


def test_injection_fails_closed_on_an_uncovered_cell():
    """A partial selection must stop the sweep, not silently mix searched and
    unsearched rates into one leaderboard."""
    partial = {"schema_version": 1, "selected": {"flat": {"gru": "1e-3"}}}
    with pytest.raises(SystemExit) as excinfo:
        build_matrix({"2"}, [947], 2048, 1, accum=14,
                     effective_batch_size=28292, lr_selection=partial)
    assert "covers no" in str(excinfo.value)


def test_exp0_covers_every_model_the_main_matrix_will_ask_for():
    """The two rosters must agree, or the fail-closed check above becomes a
    guaranteed crash halfway through a real sweep."""
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    covered = {arm: set() for arm in EXP0_ARMS}
    for name, _ in probe:
        parsed = select_lr.parse_exp0_name(name)
        if parsed["arm"] in covered:
            covered[parsed["arm"]].add(parsed["model"])
    assert covered["flat"] == set(FLAT_MODELS)
    assert covered["agg"] == set(FLAT_MODELS)
    assert covered["roll"] == set(FLAT_MODELS)
    assert covered["nd"] == set(ND_BACKBONES)


def test_exp0_probes_each_backbone_on_a_variant_it_actually_runs():
    """A rate probed on a variant the backbone never runs would be selected for
    the wrong optimisation problem."""
    for model in ND_BACKBONES:
        variant = exp0_nd_variant(model)
        main_runs = build_matrix({"3", "6"}, [947], 2048, 1, accum=14,
                                 effective_batch_size=28292)
        emitted = {a[a.index("--variant") + 1] for n, a in main_runs
                   if a[a.index("--model") + 1] == model}
        assert variant in emitted, (
            f"{model} is probed on {variant}, which the main matrix never runs "
            f"(it runs {sorted(emitted)})")


# --------------------------------------------------------------------------
# Test 5 retirement
# --------------------------------------------------------------------------

def test_test5_is_refused_with_an_explanation():
    with pytest.raises(SystemExit) as excinfo:
        build_matrix({"5"}, [947], 2048, 1, accum=14, effective_batch_size=28292)
    message = str(excinfo.value)
    assert "retired" in message
    assert "--tests 0" in message, "must point at the replacement"


def test_test5_is_refused_even_when_mixed_with_live_tests():
    with pytest.raises(SystemExit):
        build_matrix({"3", "5"}, [947], 2048, 1, accum=14,
                     effective_batch_size=28292)


def test_no_run_name_carries_a_test5_lr_tag():
    runs = build_matrix({"1", "1.1", "2", "3", "4", "6"}, DEFAULT_SEEDS, 2048, 1,
                        accum=14, effective_batch_size=28292)
    assert not any(f"_{lr_tag(lr)}_" in name
                   for name, _ in runs
                   for lr in ("1e-4", "3e-4", "5e-4", "1e-3"))


def test_retired_tests_is_the_single_source_of_truth():
    assert set(RETIRED_TESTS) == {"5"}


# --------------------------------------------------------------------------
# every axis on which the probe differs from the runs it selects for
# --------------------------------------------------------------------------

@pytest.mark.parametrize("check_arm,base_arm", sorted(EXP0_CHECK_BASE.items()))
def test_every_check_arm_is_declared_and_maps_to_a_real_base_arm(check_arm, base_arm):
    assert base_arm in EXP0_ARMS
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    assert any(n.startswith(f"exp0_{check_arm}_") for n, _ in probe), (
        f"{check_arm} is declared but never emitted")


def test_the_one_hot_flat_check_matches_the_test_2_flat_protocol():
    """Test 2's flat runs use --variant onehot; the flat probe uses
    embeddings. Everything else about the two must still agree."""
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    check = [a for n, a in probe if n.startswith("exp0_encflat_gru_")][0]
    base = [a for n, a in probe if n.startswith("exp0_flat_gru_")][0]
    assert check[check.index("--variant") + 1] == "onehot"
    assert base[base.index("--variant") + 1] == "embeddings"
    for flag in ("--batch-size", "--grad-accum", "--effective-batch-size"):
        assert check[check.index(flag) + 1] == base[base.index(flag) + 1]


def test_the_one_hot_nd_check_matches_the_nd_protocol():
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    check = [a for n, a in probe if n.startswith("exp0_encnd_gru_")][0]
    base = [a for n, a in probe if n.startswith("exp0_nd_gru_")][0]
    assert check[check.index("--combo-encoder") + 1] == "onehot"
    assert base[base.index("--combo-encoder") + 1] == "embeddings"
    assert check[check.index("--variant") + 1] == base[base.index("--variant") + 1]
    assert check[check.index("--batch-size") + 1] == base[base.index("--batch-size") + 1]


def test_the_identity_check_adds_only_axis_identity():
    """Test 4 is the N-D grid plus learned identity on the promoted axes --
    a different model, so a shared rate is an assumption."""
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    check = [a for n, a in probe if n.startswith("exp0_axid_gru_")][0]
    base = [a for n, a in probe if n.startswith("exp0_nd_gru_")][0]
    assert "--axis-identity" in check and "--axis-identity" not in base
    assert check[check.index("--variant") + 1] == base[base.index("--variant") + 1]
    assert check[check.index("--combo-encoder") + 1] == \
        base[base.index("--combo-encoder") + 1]


def test_the_mech_check_swaps_only_the_mixer():
    """The fa cells train at the rate probed on asa -- a swap of the whole
    grid-mixing operator, so a shared rate is an assumption. The check varies
    exactly that one axis against the N-D probe."""
    probe = build_exp0_matrix([947], 2048, 1, 14, "x.npz", 28292)
    check = [a for n, a in probe if n.startswith("exp0_mech_gru_")][0]
    base = [a for n, a in probe if n.startswith("exp0_nd_gru_")][0]
    assert check[check.index("--variant") + 1] == "fa_3d"
    assert base[base.index("--variant") + 1] == "asa_3d"
    assert check[check.index("--combo-encoder") + 1] == \
        base[base.index("--combo-encoder") + 1]
    assert check[check.index("--batch-size") + 1] == \
        base[base.index("--batch-size") + 1]


def test_the_mech_check_models_actually_run_fa_in_the_main_matrix():
    """The check is only evidence if its cells exist in the declared matrix:
    every mech-check model must have fa cells to select for, at the probed
    dimensionality."""
    from scripts.sweep import EXP0_MECH_CHECK_MODELS, EXP0_PROBE_DIM
    main_runs = build_matrix({"2", "3", "4", "6"}, [947], 2048, 1, accum=14,
                             effective_batch_size=28292)
    for model in EXP0_MECH_CHECK_MODELS:
        emitted = {a[a.index("--variant") + 1] for n, a in main_runs
                   if a[a.index("--model") + 1] == model}
        assert f"fa_{EXP0_PROBE_DIM}d" in emitted, (
            f"{model} is mech-checked on fa_{EXP0_PROBE_DIM}d, which the main "
            f"matrix never runs for it (it runs {sorted(emitted)})")


def test_check_arms_never_appear_in_the_selection(tmp_path):
    spec = {("flat", "gru", 947): {"1e-4": SLOW, "1e-3": FAST},
            ("nd", "gru", 947): {"1e-4": SLOW, "1e-3": FAST},
            ("roll", "gru", 947): {"1e-4": SLOW, "1e-3": FAST}}
    for arm in EXP0_CHECK_BASE:
        spec[(arm, "gru", 947)] = {"1e-4": SLOW, "1e-3": FAST}
    runs = _session(tmp_path, spec)
    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert set(payload["selected"]) == {"flat", "nd", "roll"}
    assert set(payload["transfer_checks"]) == set(EXP0_CHECK_BASE)
    for check in payload["transfer_checks"].values():
        assert check["models"]["gru"]["consistent"] is True


def test_the_arm_regex_is_not_spelled_out_twice():
    """sweep.py declares the arms; select_lr.py imports the pattern. Two
    copies of the ALTERNATION is how an arm gets added on one side and
    silently unparseable on the other. (Arm names still appear in
    select_lr.py's human-readable labels, which is fine -- a stale label
    misreads, an unparseable name refuses the whole selection.)"""
    source = (REPO / "scripts" / "select_lr.py").read_text()
    assert "exp0_arm_pattern()" in source
    assert '"|".join(' not in source, "select_lr.py rebuilds the alternation"
    assert "dim\\d+" not in source, "select_lr.py re-spells the dim pattern"


def test_a_new_arm_reaches_select_lr_without_touching_it():
    """The point of sharing the pattern: adding an arm in sweep.py must be
    enough. If this ever needs a select_lr.py edit, the coupling is back."""
    import re as _re
    from scripts.sweep import exp0_arm_pattern
    pattern = _re.compile(r"^exp0_(" + exp0_arm_pattern() + r")_")
    for arm in set(EXP0_ARMS) | set(EXP0_CHECK_BASE) | {"dim2", "dim4"}:
        assert pattern.match(f"exp0_{arm}_gru_lr1e4_s947"), arm
        assert select_lr.parse_exp0_name(f"exp0_{arm}_gru_lr1e4_s947")["arm"] == arm


# --------------------------------------------------------------------------
# the fa mixer gets its own rate, folded in as an additive patch
# --------------------------------------------------------------------------
#
# The nd rate is probed on asa_3d and 144 of the 477 declared runs use the fa
# mixer instead. Measured on the real sweep, that inheritance reverses a cell:
# mamba3 at 3-D reaches 0.634 on fa at its own 1e-3, but 1.024 at the 1e-4 the
# asa probe chose -- worse than asa's own 0.879. So fa gets its own arm, and
# the probe that produces it is folded in without disturbing the session that
# produced the base selection.

def test_an_fa_variant_asks_the_fa_arm_for_its_rate():
    from scripts.sweep import nd_rate_arm
    assert nd_rate_arm("fa_2d") == "mech"
    assert nd_rate_arm("fa_4d") == "mech"
    assert nd_rate_arm("asa_3d") == "nd"
    assert nd_rate_arm("grid_4d") == "nd"


def _sel(**arms):
    base = {"flat": {m: "1e-3" for m in FLAT_MODELS},
            "agg": {m: "1e-3" for m in FLAT_MODELS},
            "roll": {m: "1e-3" for m in FLAT_MODELS},
            "nd": {m: "3e-4" for m in ND_BACKBONES}}
    base.update(arms)
    return {"schema_version": 1, "selected": base}


def _rates_by_mixer(selection):
    runs = build_matrix({"1", "1.1", "2", "3", "4", "6"}, [947], 2048, 1,
                        accum=15, effective_batch_size=30087,
                        lr_selection=selection)
    out = {}
    for name, argv in runs:
        if "--lr" not in argv:
            continue
        rate = argv[argv.index("--lr") + 1]
        for key, mark in (("fa", "_fa_"), ("asa", "_asa_"), ("grid", "_grid_")):
            if mark in name:
                out.setdefault(key, set()).add(rate)
    return out


def test_without_an_fa_probe_the_fa_cells_are_refused():
    """A base selection from before the fa probe existed used to fall back
    silently to the nd (asa-probed) rate for every fa cell — the exact
    inheritance the mech arm exists to remove, because it reverses a measured
    cell. It must fail closed like any other uncovered cell, with a pointer
    at the supplementary probe."""
    with pytest.raises(SystemExit) as excinfo:
        _rates_by_mixer(_sel())
    assert "mech/" in str(excinfo.value)
    assert "--lr-selection-patch" in str(excinfo.value)


def test_with_an_fa_arm_the_two_mixers_diverge():
    by = _rates_by_mixer(_sel(mech={m: "1e-2" for m in ND_BACKBONES}))
    assert by["fa"] == {"1e-2"}
    assert by["asa"] == {"3e-4"}


def test_the_hybrid_identity_arm_is_identity_aware_and_rate_correct():
    """The opt-in --hybrid-identity runs must carry --axis-identity and draw
    their rate from the same arm as their identity-blind twins: fa cells from
    mech, asa cells from nd. A wrong arm here would rerun the asa->fa
    inheritance mistake inside the opt-in arm."""
    runs = build_matrix({"6"}, [947], 2048, 1, accum=15,
                        effective_batch_size=30087, hybrid_identity=True,
                        lr_selection=_sel(mech={m: "1e-2" for m in ND_BACKBONES}))
    id_runs = [(n, a) for n, a in runs if "_id_" in n]
    assert len(id_runs) == 24     # 2 models x 2 mixers x 3 dims x 2 encoders
    for name, argv in id_runs:
        assert "--axis-identity" in argv, name
        rate = argv[argv.index("--lr") + 1]
        assert rate == ("1e-2" if "_fa_" in name else "3e-4"), name


def test_the_grid_native_ssms_are_not_swept_into_the_fa_arm():
    """mamba_nd and s4nd run grid_*, not fa_*. Their run site has no `variant`
    of its own, and reading the axial block's leaked loop variable there gave
    them fa_4d's arm -- and so the fa rate -- for every grid run."""
    by = _rates_by_mixer(_sel(mech={m: "1e-2" for m in ND_BACKBONES}))
    assert by["grid"] == {"3e-4"}, "grid-native SSMs must take the nd rate"


def test_one_arm_can_be_probed_alone_without_redeclaring_exp0():
    """A supplementary probe runs in its own --runs-dir. It must emit only the
    arm asked for, so it does not re-run a finished session's 189 cells."""
    only = build_exp0_matrix([947], 2048, 1, 15, "x.npz", 30087,
                             only_arms=["mech"])
    assert only
    assert all(n.startswith("exp0_mech_") for n, _ in only)
    from scripts.sweep import EXP0_PROBE_DIM as PROBE_DIM
    variants = {a[a.index("--variant") + 1] for _, a in only}
    assert variants == {f"fa_{PROBE_DIM}d"}
    full = build_exp0_matrix([947], 2048, 1, 15, "x.npz", 30087)
    assert len(only) < len(full)


def test_the_fa_probe_covers_every_backbone_the_matrix_runs_on_fa():
    """Otherwise _lr_args refuses the uncovered ones and a real sweep stops
    halfway through declaring its matrix."""
    from scripts.sweep import EXP0_MECH_CHECK_MODELS
    probe = build_exp0_matrix([947], 2048, 1, 15, "x.npz", 30087,
                              only_arms=["mech"])
    probed = {select_lr.parse_exp0_name(n)["model"] for n, _ in probe}
    runs = build_matrix({"2", "3", "4", "6"}, [947], 2048, 1, accum=15,
                        effective_batch_size=30087)
    on_fa = {a[a.index("--model") + 1] for n, a in runs if "_fa_" in n}
    assert on_fa <= probed, f"unprobed fa backbones: {sorted(on_fa - probed)}"
    # and nothing beyond them: a grid-native SSM probed on fa_3d would select
    # a rate for cells the matrix never runs.
    assert probed == on_fa, f"probed but never run on fa: {sorted(probed - on_fa)}"


# --------------------------------------------------------------------------
# the diagnostic mixer arms (Exp 7 / Exp 8) go through the same three stages
# --------------------------------------------------------------------------

MIXER_ARMS = [("mechlocal", "7", "fa_local"), ("mechsm", "8", "fa_sm")]


@pytest.mark.parametrize("arm,tid,mixer", MIXER_ARMS)
def test_a_mixer_probe_selects_and_reaches_its_experiment(tmp_path, arm, tid,
                                                          mixer):
    """probe -> lr_selection.json -> the rate on the Exp 7/8 command line.

    The whole point of giving fa_local and fa_sm their own arms is that they
    stop inheriting the rate probed on the authors' fa_3d. That only holds if
    all three stages agree on the arm name, so the chain is tested end to end
    rather than at either end.
    """
    # A different winner per model, so a mix-up between models or between arms
    # shows up as a wrong rate rather than an accidentally-right one.
    winners = {"gru": "1e-2", "lstm": "3e-4", "transformer": "1e-3"}
    spec = {(arm, model, 947): {lr: (FAST if lr == win else SLOW)
                                for lr in EXP0_LR_GRID}
            for model, win in winners.items()}
    runs = _session(tmp_path, spec)

    assert select_lr.main(["--runs", str(runs)]) == 0
    payload = json.loads((runs / "lr_selection.json").read_text())
    assert payload["official"] is True
    assert payload["selected"] == {arm: winners}

    matrix = build_matrix({tid}, [947], 2048, 1, lr_selection=payload)
    assert matrix
    for name, argv in matrix:
        assert f"_{mixer}_" in name
        model = name.split("_", 1)[0]
        assert argv[argv.index("--lr") + 1] == winners[model], name


@pytest.mark.parametrize("arm,tid", [(a, t) for a, t, _ in MIXER_ARMS])
def test_an_uncovered_mixer_cell_is_refused_with_the_recipe(arm, tid):
    """Refusing is right; refusing without saying how to fix it is not.

    Only "mech" used to get the run-your-own-probe hint, which left the
    operator of a fa_local or fa_sm sweep at the same dead end under a
    different label.
    """
    base = {"schema_version": 1, "selected": {"nd": {"gru": "1e-4"}}}
    with pytest.raises(SystemExit) as excinfo:
        build_matrix({tid}, [947], 2048, 1, lr_selection=base)
    message = str(excinfo.value)
    assert f"covers no {arm}/" in message
    assert f"--exp0-arms {arm}" in message
    assert "--lr-selection-patch" in message


@pytest.mark.parametrize("arm,tid,mixer", MIXER_ARMS)
def test_the_patch_flow_carries_a_supplementary_probe_into_the_sweep(
        tmp_path, arm, tid, mixer):
    """The box workflow: a base selection plus a later arm, merged at launch.

    This is how the mech arm was added after its Exp 0 session had already
    finished (artifacts/lr_selection_fa.json), and it is how these two will be
    added. Exercised through the CLI because the merge lives in __main__.
    """
    import subprocess

    winners = {"gru": "1e-2", "lstm": "3e-4", "transformer": "1e-3"}
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "schema_version": 1, "official": True,
        "selected": {"nd": {m: "1e-4" for m in winners}},
    }))
    patch = tmp_path / "patch.json"
    patch.write_text(json.dumps({
        "schema_version": 1, "official": True, "selected": {arm: winners},
    }))

    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "sweep.py"),
         "--tests", tid, "--runs-dir", str(tmp_path / "session"),
         "--lr-selection", str(base), "--lr-selection-patch", str(patch),
         "--dry-run"],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert f"{arm}: gru=1e-2,lstm=3e-4,transformer=1e-3" in proc.stdout
    # and the base arm survives the merge rather than being replaced
    assert "nd: gru=1e-4" in proc.stdout
    assert proc.stdout.count(f"_{mixer}_") == 27
