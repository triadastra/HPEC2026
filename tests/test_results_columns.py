"""results.csv must say which task a row belongs to, and what it was selected on.

Two columns, both about not comparing incomparable things:

* ``panel`` -- the file holds the national aggregate series, the per-year
  aggregate folds, and the 28k-series panel. Their errors live in different
  target spaces, so a reader who sorts the file by MSE gets a ranking that is
  meaningless across panels unless the panel is in the row.
* ``val_loss`` -- with Exp 0 choosing learning rates on validation loss, the
  selection signal has to be visible next to the test metric it must not have
  been chosen from.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_evaluate():
    spec = importlib.util.spec_from_file_location(
        "evaluate_mod", REPO / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = _load_evaluate()

_CHANNEL = {"MAE": 0.1, "MSE": 0.01, "RMSE": 0.1, "sMAPE": 5.0, "MASE": 0.8}
# All three channels: _pooled_rolling_rows averages value_* and weight_* too.
METRICS = {c: dict(_CHANNEL) for c in ("both", "value", "weight")}


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------

@pytest.mark.parametrize("run,expected", [
    ("gru_aggregate_s947", "aggregate"),
    ("s4_aggregate_s732", "aggregate"),
    ("gru_aggregate_roll_y2024_s947", "aggregate_rolling"),
    ("gru_embeddings_1d_s947", "series"),
    ("gru_onehot_1d_s947", "series"),
    ("gru_asa_3d_embeddings_s947", "series"),
    ("transformer_fa_4d_id_onehot_s619", "series"),
    ("s4nd_grid_2d_embeddings_s947", "series"),
])
def test_every_run_name_lands_in_the_right_panel(run, expected):
    cfg = evaluate.parse_run(run)
    assert cfg is not None, f"{run} did not parse"
    assert evaluate._panel(cfg) == expected


def test_the_panel_column_is_written_for_every_row():
    row = evaluate._flatten_result("gru_aggregate_s947",
                                   evaluate.parse_run("gru_aggregate_s947"),
                                   METRICS)
    assert row["panel"] == "aggregate"


def test_the_three_panels_are_distinguishable_in_one_file(tmp_path):
    """The concrete hazard: an aggregate row and a series row with the same
    model/seed, side by side, at wildly different error scales."""
    rows = [
        evaluate._flatten_result(name, evaluate.parse_run(name), METRICS)
        for name in ("gru_aggregate_s947",
                     "gru_aggregate_roll_y2024_s947",
                     "gru_embeddings_1d_s947")
    ]
    out = tmp_path / "results.csv"
    evaluate._write_csv(out, rows)
    header = out.read_text().splitlines()[0].split(",")
    assert "panel" in header
    assert {r["panel"] for r in rows} == {
        "aggregate", "aggregate_rolling", "series"}


def test_baselines_are_placed_in_the_series_panel():
    """Baselines are scored on the per-series panel, so they must compare
    against the per-series models and not against the aggregate arm."""
    cfg = dict(model="persistence", variant="baseline", enc="none", seed=0)
    assert evaluate._panel(cfg) == "series"


# --------------------------------------------------------------------------
# val_loss
# --------------------------------------------------------------------------

def _write_curve(runs_dir, run, values):
    d = runs_dir / run / "logs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps([
        {"epoch": i, "phase": "train", "train_loss": v, "val_loss": v}
        for i, v in enumerate(values)
    ]))


def test_val_loss_is_the_best_epoch_not_the_last(tmp_path):
    """Training keeps going after the best epoch; the checkpoint that gets
    scored is the best one, so the reported val_loss must match it."""
    _write_curve(tmp_path, "gru_embeddings_1d_s947", [0.9, 0.4, 0.2, 0.6, 0.8])
    row = evaluate._flatten_result(
        "gru_embeddings_1d_s947",
        evaluate.parse_run("gru_embeddings_1d_s947"), METRICS,
        runs_dir=tmp_path)
    assert row["val_loss"] == pytest.approx(0.2)


def test_non_finite_epochs_are_ignored(tmp_path):
    run = "gru_embeddings_1d_s947"
    d = tmp_path / run / "logs"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps([
        {"epoch": 0, "val_loss": 0.5},
        {"epoch": 1, "val_loss": None},
        {"epoch": 2, "val_loss": "nan"},
    ]))
    row = evaluate._flatten_result(run, evaluate.parse_run(run), METRICS,
                                   runs_dir=tmp_path)
    assert row["val_loss"] == pytest.approx(0.5)


def test_a_missing_training_log_omits_the_column_instead_of_guessing(tmp_path):
    run = "gru_embeddings_1d_s947"
    row = evaluate._flatten_result(run, evaluate.parse_run(run), METRICS,
                                   runs_dir=tmp_path)
    assert "val_loss" not in row


def test_analytic_baselines_carry_no_val_loss():
    """They were never trained, so an empty cell is the honest value."""
    cfg = dict(model="persistence", variant="baseline", enc="none", seed=0)
    row = evaluate._flatten_result("persistence", cfg, METRICS)
    assert "val_loss" not in row


def test_a_corrupt_training_log_does_not_break_evaluation(tmp_path):
    run = "gru_embeddings_1d_s947"
    d = tmp_path / run / "logs"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text("{not json")
    row = evaluate._flatten_result(run, evaluate.parse_run(run), METRICS,
                                   runs_dir=tmp_path)
    assert "val_loss" not in row


def test_rows_with_and_without_val_loss_share_one_header(tmp_path):
    """DictWriter is built from the union of keys; a baseline row missing
    val_loss must not truncate the column for the trained rows."""
    _write_curve(tmp_path, "gru_embeddings_1d_s947", [0.3, 0.2])
    baseline = evaluate._flatten_result(
        "persistence",
        dict(model="persistence", variant="baseline", enc="none", seed=0),
        METRICS)
    trained = evaluate._flatten_result(
        "gru_embeddings_1d_s947",
        evaluate.parse_run("gru_embeddings_1d_s947"), METRICS,
        runs_dir=tmp_path)
    out = tmp_path / "results.csv"
    evaluate._write_csv(out, [baseline, trained])          # baseline first
    lines = out.read_text().splitlines()
    header = lines[0].split(",")
    assert "val_loss" in header
    assert lines[1].split(",")[header.index("val_loss")] == ""
    assert float(lines[2].split(",")[header.index("val_loss")]) == pytest.approx(0.2)


# --------------------------------------------------------------------------
# the headline leaderboard, not just the per-run rows
# --------------------------------------------------------------------------
#
# results.csv is the table people sort by MSE. The panel started out only on
# the per-run rows in metrics.json, which left the one table that actually gets
# read unable to say which of its rows are comparable.

def test_the_leaderboard_carries_the_panel():
    rows = [
        evaluate._flatten_result(n, evaluate.parse_run(n), METRICS)
        for n in ("gru_aggregate_s947", "gru_embeddings_1d_s947")
    ]
    averaged = evaluate._result_averages(rows)
    assert all("panel" in r for r in averaged)
    assert {r["panel"] for r in averaged} == {"aggregate", "series"}


def test_the_leaderboard_never_averages_across_panels():
    """Two rows that agree on (model, variant, encoder) but sit in different
    target spaces must not collapse into one leaderboard row."""
    a = evaluate._flatten_result("gru_aggregate_s947",
                                 evaluate.parse_run("gru_aggregate_s947"), METRICS)
    b = dict(a)                      # same model/variant/enc, different panel
    b["panel"] = "series"
    b["both_MSE"] = 9.0
    averaged = evaluate._result_averages([a, b])
    assert len(averaged) == 2, "panels were averaged together"
    by_panel = {r["panel"]: r for r in averaged}
    assert by_panel["aggregate"]["MSE"] != by_panel["series"]["MSE"]


def test_the_leaderboard_carries_val_loss_where_it_exists(tmp_path):
    _write_curve(tmp_path, "gru_embeddings_1d_s947", [0.5, 0.3])
    trained = evaluate._flatten_result(
        "gru_embeddings_1d_s947",
        evaluate.parse_run("gru_embeddings_1d_s947"), METRICS, runs_dir=tmp_path)
    averaged = evaluate._result_averages([trained])
    assert averaged[0]["val_loss"] == pytest.approx(0.3)


def test_analytic_baselines_leave_the_leaderboard_val_loss_empty():
    cfg = dict(model="persistence", variant="baseline", enc="none", seed=0)
    row = evaluate._flatten_result("persistence", cfg, METRICS)
    averaged = evaluate._result_averages([row])
    assert "val_loss" not in averaged[0], "a baseline never trained"


# --------------------------------------------------------------------------
# the rolling analytic anchors
# --------------------------------------------------------------------------

def test_rolling_baselines_are_labelled_as_rolling_aggregate():
    """They score one aggregate fold. Labelled `series`, their error sat in the
    leaderboard beside the 28k-series models as if it were comparable."""
    identity = dict(model="seasonal_naive", variant="aggregate_roll_y2024",
                    enc="aggregate", seed=0, test_year=2024,
                    aggregate=True, rolling=True)
    assert evaluate._panel(identity) == "aggregate_rolling"


def test_the_rolling_baseline_identity_in_evaluate_declares_its_panel():
    """Pins the call site: the identity dict must carry the keys _panel reads,
    which a plain `enc="aggregate"` does not supply."""
    source = (REPO / "scripts" / "evaluate.py").read_text()
    block = source[source.index("Analytic anchors for every annual aggregate fold"):]
    block = block[:block.index("_flatten_result")]
    assert "aggregate=True" in block and "rolling=True" in block


def test_fixed_baselines_stay_in_the_series_panel():
    """They are scored on the per-series lattice, so `series` is correct --
    the fix for the rolling anchors must not have moved these."""
    cfg = dict(model="seasonal_naive", variant="baseline", enc="none", seed=0)
    assert evaluate._panel(cfg) == "series"


# --------------------------------------------------------------------------
# the invariant, not the instances
# --------------------------------------------------------------------------
#
# Three separate row-producing paths have now shipped without declaring their
# panel: the rolling analytic anchors, the seed-averaged leaderboard, and the
# pooled Test 1.1 summary. Each was fixed individually and the next one broke
# the same way, because the panel is derived from keys a builder has to
# remember to set. This pins the rule instead: whatever a row's `variant`
# says it is, its panel has to agree.

def _panel_implied_by_name(run):
    """What the RUN NAME says the row is.

    Not the variant: a fixed-aggregate run is `gru_aggregate_s947`, whose
    parsed variant is "embeddings" and whose aggregate-ness lives in `enc`.
    The name is the one field that always carries the task.
    """
    if "_aggregate_roll" in run:
        return "aggregate_rolling"
    if "_aggregate" in run:
        return "aggregate"
    return "series"


def test_pooled_rolling_rows_declare_the_rolling_panel():
    """rolling_pooled.csv averages annual AGGREGATE folds. Labelled `series`,
    a pooled aggregate error sat beside the 28k-series models."""
    rows = [
        evaluate._flatten_result(f"gru_aggregate_roll_y{y}_s947",
                                 evaluate.parse_run(f"gru_aggregate_roll_y{y}_s947"),
                                 METRICS)
        for y in (2020, 2021, 2022)
    ]
    pooled = evaluate._pooled_rolling_rows(rows)
    assert pooled, "fixture produced no pooled rows"
    averaged = evaluate._result_averages(pooled)
    assert {r["panel"] for r in averaged} == {"aggregate_rolling"}


def test_the_pooled_builder_sets_the_keys_panel_reads():
    """_pooled_rolling_rows does not go through _flatten_result, so it has to
    declare the markers itself or _panel silently falls back to `series`."""
    rows = [
        evaluate._flatten_result(f"gru_aggregate_roll_y{y}_s947",
                                 evaluate.parse_run(f"gru_aggregate_roll_y{y}_s947"),
                                 METRICS)
        for y in (2020, 2021)
    ]
    pooled = evaluate._pooled_rolling_rows(rows)[0]
    assert evaluate._panel(pooled) == "aggregate_rolling"


@pytest.mark.parametrize("run", [
    "gru_aggregate_s947",
    "gru_aggregate_roll_y2024_s947",
    "gru_embeddings_1d_s947",
    "s4nd_grid_4d_onehot_s619",
])
def test_a_rows_panel_always_agrees_with_its_name(run):
    row = evaluate._flatten_result(run, evaluate.parse_run(run), METRICS)
    assert row["panel"] == _panel_implied_by_name(run)


def test_no_aggregate_variant_ever_lands_in_the_series_panel():
    """The invariant, across every row-producing path at once: if the variant
    names an aggregate task, the panel must not say `series`."""
    per_fold = [
        evaluate._flatten_result(f"gru_aggregate_roll_y{y}_s947",
                                 evaluate.parse_run(f"gru_aggregate_roll_y{y}_s947"),
                                 METRICS)
        for y in (2020, 2021)
    ]
    fixed = [evaluate._flatten_result("gru_aggregate_s947",
                                      evaluate.parse_run("gru_aggregate_s947"),
                                      METRICS)]
    pooled = evaluate._pooled_rolling_rows(per_fold)
    everything = (per_fold + fixed + pooled
                  + evaluate._result_averages(per_fold + fixed)
                  + evaluate._result_averages(pooled))
    offenders = [
        (r.get("run"), r["variant"], r.get("panel") or evaluate._panel(r))
        for r in everything
        if (str(r.get("variant", "")).startswith("aggregate")
            or "_aggregate" in str(r.get("run", "")))
        and (r.get("panel") or evaluate._panel(r)) == "series"
    ]
    assert not offenders, f"aggregate rows filed under `series`: {offenders}"


# --------------------------------------------------------------------------
# The fixed aggregate split needs its own analytic floor
# --------------------------------------------------------------------------

def test_a_fixed_split_baseline_lands_in_test_1s_panel():
    """The identity the fixed-split baselines are emitted with has to put them
    beside Test 1's trained rows -- not in the series panel, where an error
    computed on one aggregated series would sit next to the 28k-series models
    as if comparable, and not in the rolling panel, which is a different task.
    """
    identity = dict(model="seasonal_naive", variant="aggregate",
                    enc="aggregate", seed=0, aggregate=True,
                    model_session_id="analytic_baseline")
    assert evaluate._panel(identity) == "aggregate"

    trained = dict(model="gru", variant="aggregate", enc="embeddings", seed=947,
                   aggregate=True)
    assert evaluate._panel(trained) == evaluate._panel(identity), (
        "the floor must be scored in the same target space as what it floors")


def test_the_rolling_and_fixed_aggregate_panels_stay_distinct():
    fixed = dict(model="seasonal_naive", aggregate=True)
    rolled = dict(model="seasonal_naive", aggregate=True, rolling=True,
                  test_year=2024)
    assert evaluate._panel(fixed) != evaluate._panel(rolled)


def test_every_scored_task_gets_an_analytic_floor():
    """results.csv holds three tasks. Each one needs a persistence reference
    computed on ITS OWN split, or its leaderboard can only say which trained
    model won, never whether training beat doing nothing.

    The 28k-series panel and the rolling folds had one; the fixed aggregate
    split did not, and catalog.md claimed otherwise. If this ever fails, a
    panel has lost its floor again.
    """
    source = (REPO / "scripts" / "evaluate.py").read_text()
    body = source[source.index("if not args.no_baselines:"):
                  source.index("for name in sorted(configs):")]
    assert "eval_baseline(name, cl, dev)" in body, "series panel floor"
    assert "aggregate_roll_y" in body, "rolling fold floors"
    assert "eval_baseline(baseline, cl_agg, dev)" in body, (
        "fixed aggregate split has no analytic floor -- Test 1's rows, the "
        "paper's T-1 anchor, would have nothing to be measured against")


def test_the_catalog_claim_about_baselines_is_true():
    """catalog.md contrasted the fixed split with the Test-1.1 folds, which
    was true of the SPLIT SCHEME -- the series-panel anchors do use the fixed
    boundaries. What it did not say is which panel, and the aggregate panel
    under that same fixed split had no anchor at all. Panel and split scheme
    are separate axes, and the sentence only pinned one of them."""
    catalog = (REPO / "catalog.md").read_text()
    assert "fixed split" in catalog
    source = (REPO / "scripts" / "evaluate.py").read_text()
    assert "eval_baseline(baseline, cl_agg, dev)" in source
