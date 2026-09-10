#!/usr/bin/env python
"""Strict evaluation for Census sweep checkpoints.

Normalized MAE/MSE/RMSE remain directly comparable to the training objective.
sMAPE is computed after the per-series transform is inverted to real units,
while MASE is a log-space ratio against the training seasonal-naive MAE;
subgroup files expose performance by state, commodity, flow, training scale,
and training density. Official result files are not written when a planned run,
required seed, checkpoint load, or baseline evaluation fails.
"""
import argparse
import csv
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.census_loader import CensusLattice, census_config_from_config
from src.models import create_model
from src.utils import (argv_value, compose_config, compose_data_config,
                       model_kwargs_from_config,
                       run_input_fingerprint, validate_completion)

MODELS = ("gru", "lstm", "transformer", "gpt", "mamba_nd", "mamba3",
          "mamba2", "mamba", "s4nd", "s4")
BASELINES = ("seasonal_naive", "random_walk", "moving_average")
METRIC_KEYS = ("MAE", "MSE", "RMSE", "sMAPE", "MASE")


def parse_run(name):
    """Parse a sweep directory name into its model construction fields."""
    m = re.match(r"^(" + "|".join(MODELS) + r")_(.+)_s(\d+)$", name)
    if not m:
        return None
    model, mid, seed = m.group(1), m.group(2), int(m.group(3))
    # Hyperparameter tags are stripped off the middle so the run still builds
    # from its base variant, then re-appended to `variant` so each setting is a
    # distinct leaderboard row with its own required seed cohort.
    tags = []
    for pattern in (r"^(.+)_(dh\d+)$", r"^(.+)_(lr\d+e\d+)$"):
        m_tag = re.match(pattern, mid)
        if m_tag:
            mid, tag = m_tag.group(1), m_tag.group(2)
            tags.insert(0, tag)
    lr_tag = "_".join(tags) if tags else None
    rolling = re.fullmatch(r"aggregate_roll_y(\d{4})", mid)
    if mid == "aggregate":
        cfg = dict(model=model, variant="embeddings", enc="aggregate", seed=seed,
                   combo=False, aggregate=True)
    elif rolling:
        year = int(rolling.group(1))
        cfg = dict(
            model=model, variant=f"aggregate_roll_y{year}", enc="aggregate",
            seed=seed, combo=False, aggregate=True, rolling=True,
            test_year=year, build_variant="embeddings",
        )
    elif mid.endswith("_1d"):
        cfg = dict(model=model, variant=mid[:-3], enc=mid[:-3], seed=seed,
                   combo=False, aggregate=False)
    else:
        variant, enc = mid.rsplit("_", 1)
        axis_id = variant.endswith("_id")
        cfg = dict(model=model, variant=variant, enc=enc, seed=seed, combo=True,
                   aggregate=False, axis_id=axis_id,
                   build_variant=variant[:-3] if axis_id else variant)
    if lr_tag:
        cfg["build_variant"] = cfg.get("build_variant", cfg["variant"])
        cfg["variant"] = f"{cfg['variant']}_{lr_tag}"
    return cfg


def _metrics(pred, true, scale=None, smape_pred=None, smape_true=None):
    """Normalized error metrics plus raw-unit sMAPE and log-space MASE.

    ``scale`` is the training seasonal-naive MAE in the same normalized space
    as ``pred``/``true`` (see ``seasonal_scale``), so MASE is scale-free per
    series but still a log-space ratio.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    err = pred - true
    mse = float(np.mean(err ** 2))
    sp = pred if smape_pred is None else np.asarray(smape_pred, dtype=np.float64).reshape(-1)
    st = true if smape_true is None else np.asarray(smape_true, dtype=np.float64).reshape(-1)
    smape = float(np.mean(2 * np.abs(sp - st) / (np.abs(sp) + np.abs(st) + 1e-8)) * 100)
    out = dict(MAE=float(np.mean(np.abs(err))), MSE=mse, RMSE=float(np.sqrt(mse)),
               sMAPE=smape)
    if scale is not None:
        sc = np.asarray(scale, dtype=np.float64).reshape(-1)
        ok = sc > 1e-12
        out["MASE"] = float(np.mean(np.abs(err[ok]) / sc[ok])) if ok.any() else float("nan")
    return out


def seasonal_scale(cl):
    """Training-only seasonal-naive MAE for each series and target channel.

    Computed on the NORMALIZED panel, i.e. in log1p space. The per-series
    MinMax factor cancels in the MASE ratio but ``log1p`` does not, so the
    reported MASE is a log-space MASE: "MASE < 1 beats seasonal persistence"
    holds on log errors, not on dollar/tonne errors. Raw-unit accuracy is
    reported separately as sMAPE. State this in the paper alongside the
    number.
    """
    burn = cl.config.input_len + cl.config.lag_count
    lo, hi = burn + 12, cl.config.train_end
    tv, tw = cl.target_ch
    out = np.zeros((cl.panel.shape[0], 2), dtype=np.float64)
    for j, ch in enumerate((tv, tw)):
        cur = cl.panel[:, lo:hi, ch].astype(np.float64)
        prev = cl.panel[:, lo - 12:hi - 12, ch].astype(np.float64)
        out[:, j] = np.abs(cur - prev).mean(axis=1)
    return out


def sample_series_ids(cl, combo):
    """Series index of every emitted test observation, in loader order."""
    n_t = cl.test_end - cl.config.val_end
    n_series = cl.panel.shape[0]
    return (np.tile(np.arange(n_series), n_t) if combo
            else np.repeat(np.arange(n_series), n_t))


def _score_arrays(P, T, sc, raw_p, raw_t):
    return dict(
        value=_metrics(P[:, 0], T[:, 0], sc[:, 0], raw_p[:, 0], raw_t[:, 0]),
        weight=_metrics(P[:, 1], T[:, 1], sc[:, 1], raw_p[:, 1], raw_t[:, 1]),
        both=_metrics(P, T, sc, raw_p, raw_t),
    )


def score(P, T, cl, combo):
    """Score normalized errors and raw-unit sMAPE on the same observations."""
    if not np.isfinite(P).all() or not np.isfinite(T).all():
        raise ValueError("predictions or targets contain NaN/Inf")
    sid = sample_series_ids(cl, combo)
    if len(sid) != len(P):
        raise ValueError(f"sample/series length mismatch ({len(sid)} vs {len(P)})")
    sc = seasonal_scale(cl)[sid]
    with np.errstate(over="ignore", invalid="ignore"):
        raw_p, raw_t = cl.inverse_targets(P, sid), cl.inverse_targets(T, sid)
    if not np.isfinite(raw_p).all() or not np.isfinite(raw_t).all():
        raise ValueError("inverse-transformed predictions or targets contain NaN/Inf")
    return _score_arrays(P, T, sc, raw_p, raw_t)


@torch.no_grad()
def eval_baseline(name, cl, dev):
    _, _, test = cl.get_dataloaders(batch_size=512, num_workers=0, combo=False,
                                    shuffle_train=False)
    tv, tw = cl.target_ch
    model = create_model(
        name, "onehot", num_numeric_features=cl.features_per_group,
        num_states=cl.num_states, num_commodities=cl.num_commodities,
        num_flows=cl.num_flows, value_idx=tv, weight_idx=tw,
    ).to(dev).eval()
    P, T = [], []
    for b in test:
        out = model(b["x_numeric"].to(dev))
        tgt = torch.stack([b["target_value"], b["target_weight"]], dim=-1)
        P.append(out.cpu().numpy().reshape(-1, 2))
        T.append(tgt.numpy().reshape(-1, 2))
    return np.concatenate(P), np.concatenate(T)


@torch.no_grad()
def eval_ckpt(ckpt, cfg, cl, dev):
    combo = cfg["combo"]
    _, _, test = cl.get_dataloaders(batch_size=(1 if combo else 512),
                                    num_workers=0, combo=combo, shuffle_train=False)
    build_variant = cfg.get("build_variant", cfg["variant"])
    composed = compose_config(cfg["model"], build_variant, extra=["config/census.yaml"])
    kw = model_kwargs_from_config(composed)
    kw.update(num_numeric_features=cl.features_per_group, num_states=cl.num_states,
              num_commodities=cl.num_commodities, num_flows=cl.num_flows)
    if combo:
        kw.update(num_combos=cl.num_combos, features_per_group=cl.features_per_group,
                  combo_coords=cl.combo_coords, lattice_dims=cl.lattice_dims,
                  combo_encoder=cfg["enc"], axis_identity=cfg.get("axis_id", False))
    model = create_model(cfg["model"], build_variant, **kw).to(dev).eval()
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    P, T = [], []
    for b in test:
        x = b["x_numeric"].to(dev)
        out = (model(x) if combo else model(
            x, b["state_ids"].to(dev), b["comm_ids"].to(dev), b["flow_ids"].to(dev)))
        tgt = torch.stack([b["target_value"], b["target_weight"]], dim=-1)
        P.append(out.cpu().numpy().reshape(-1, 2))
        T.append(tgt.numpy().reshape(-1, 2))
    return np.concatenate(P), np.concatenate(T)


def _rank_quartiles(values):
    """Value-boundary quartiles that never split identical profiles."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.empty(0, dtype=np.int64)
    cuts = np.quantile(values, (0.25, 0.5, 0.75))
    # ``searchsorted`` applies the same boundary to every tied value. Repeated
    # cuts intentionally produce empty bins instead of inventing disparities
    # between series with identical training density/scale.
    return np.searchsorted(cuts, values, side="left").astype(np.int64)


def subgroup_rows(P, T, cl, combo, identity):
    """Return state/commodity/flow/scale/density fairness audit rows."""
    if cl.N == 1:
        return []
    sid = sample_series_ids(cl, combo)
    sc = seasonal_scale(cl)[sid]
    raw_p, raw_t = cl.inverse_targets(P, sid), cl.inverse_targets(T, sid)
    tv = cl.target_ch[0]
    train_value = cl.panel_raw[:, :cl.config.train_end, tv].astype(np.float64)
    scale_q = _rank_quartiles(np.mean(train_value, axis=1))
    density_q = _rank_quartiles(np.mean(train_value > 0, axis=1))
    state_names = cl.metadata.get("states", [])
    commodity_names = cl.metadata.get("commodities", [])
    flow_names = cl.metadata.get("flows", [])

    dimensions = [
        ("state", cl.series_idx[:, 1], state_names),
        ("commodity", cl.series_idx[:, 0], commodity_names),
        ("flow", cl.series_idx[:, 2], flow_names),
        ("training_value_scale_quartile", scale_q,
         ["Q1_low", "Q2", "Q3", "Q4_high"]),
        ("training_trade_density_quartile", density_q,
         ["Q1_low", "Q2", "Q3", "Q4_high"]),
    ]
    rows = []
    for dimension, per_series, labels in dimensions:
        for group_id in np.unique(per_series):
            members = np.flatnonzero(per_series == group_id)
            n_t = cl.test_end - cl.config.val_end
            if combo:
                obs = (np.arange(n_t)[:, None] * cl.N + members[None, :]).reshape(-1)
            else:
                obs = (members[:, None] * n_t + np.arange(n_t)[None, :]).reshape(-1)
            metrics = _score_arrays(P[obs], T[obs], sc[obs], raw_p[obs], raw_t[obs])
            label = labels[int(group_id)] if int(group_id) < len(labels) else str(int(group_id))
            flat = {f"{channel}_{key}": value for channel, vals in metrics.items()
                    for key, value in vals.items()}
            rows.append(dict(**identity, dimension=dimension, group=str(label),
                             n_series=len(members), n_observations=len(obs), **flat))
    return rows


def _panel(cfg):
    """Which target space this row was scored in.

    results.csv holds three tasks whose errors are not comparable: the single
    national aggregate series (Test 1), one aggregate fold per year (Test 1.1),
    and the 28k-series panel (Tests 2/3/4/6). They have different targets,
    different normalization fits, and different numbers of scored points, so
    sorting the file by MSE without this column silently ranks an aggregate row
    against a per-series row. Every downstream grouping already splits them;
    this makes the split visible in the artifact itself.
    """
    if cfg.get("rolling"):
        return "aggregate_rolling"
    if cfg.get("aggregate"):
        return "aggregate"
    return "series"


def _val_loss(runs_dir, run):
    """Best validation loss this run reached, from its own training log.

    The selection signal, carried next to the test metric it is NOT allowed to
    be chosen from. With Exp 0 choosing learning rates on validation loss, a
    reader has to be able to see both columns to check that the reported
    ranking was not tuned on test -- and a row whose val_loss is far better
    than its neighbours' while its test MSE is not is exactly what overfitting
    the selection looks like.
    """
    path = Path(runs_dir) / run / "logs" / "metrics.json"
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    values = [row["val_loss"] for row in rows
              if isinstance(row, dict) and isinstance(row.get("val_loss"), (int, float))
              and math.isfinite(row["val_loss"])]
    return min(values) if values else None


def _flatten_result(run, cfg, metrics, runs_dir=None):
    flat = {f"{channel}_{key}": value for channel, vals in metrics.items()
            for key, value in vals.items()}
    row = dict(run=run, panel=_panel(cfg),
               **{k: cfg[k] for k in ("model", "variant", "enc", "seed")})
    for key in (
        "test_year", "train_end", "val_end", "test_end", "n_test_months",
        "model_session_id",
    ):
        if key in cfg:
            row[key] = cfg[key]
    if runs_dir is not None:
        value = _val_loss(runs_dir, run)
        if value is not None:
            row["val_loss"] = value
    return dict(**row, **flat)


def _planned_entries(runs_dir):
    manifest = runs_dir / "manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text())
    entries = data.get("runs", data) if isinstance(data, dict) else data
    return {entry["name"]: entry for entry in entries if isinstance(entry, dict)}


def _planned_names(runs_dir):
    entries = _planned_entries(runs_dir)
    return None if entries is None else set(entries)


def rolling_census_config(entry, npz, data_cfg=None):
    """Reconstruct one Test-1.1 data contract from its manifest command.

    Fold boundaries come from the declared command; everything else (window
    length, lag mode) comes from the composed ``data:`` block, so a rolling
    fold cannot silently score under different feature construction than the
    run that produced its checkpoint.
    """
    argv = entry.get("argv", [])
    values = {}
    for flag, key in (
        ("--train-end", "train_end"),
        ("--val-end", "val_end"),
        ("--test-end", "test_end"),
        ("--input-len", "input_len"),
        ("--lag-count", "lag_count"),
    ):
        value = argv_value(argv, flag)
        if value is not None:
            values[key] = int(value)
    required = {"train_end", "val_end", "test_end"}
    if not required.issubset(values):
        missing = ", ".join(sorted(required - set(values)))
        raise ValueError(f"{entry.get('name')}: rolling manifest lacks {missing}")
    if "--aggregate" not in argv or "--refit-normalization" not in argv:
        raise ValueError(
            f"{entry.get('name')}: rolling run must declare aggregate refit normalization"
        )
    return census_config_from_config(
        data_cfg or compose_data_config(),
        npz=str(npz), aggregate=True, refit_normalization=True, **values,
    )


def seed_matrix_errors(configs, expected_seeds, planned_names=None):
    """Validate planned checkpoints and complete seed cohorts."""
    errors = []
    names = set(configs)
    if planned_names is not None:
        missing = sorted(set(planned_names) - names)
        if missing:
            errors.append(f"missing {len(missing)} planned checkpoints: {', '.join(missing)}")
    groups = defaultdict(set)
    for cfg in configs.values():
        groups[(cfg["model"], cfg["variant"], cfg["enc"])].add(cfg["seed"])
    wanted = set(expected_seeds)
    for group, found in sorted(groups.items()):
        if found != wanted:
            errors.append(f"incomplete seeds for {'/'.join(group)}: expected "
                          f"{sorted(wanted)}, found {sorted(found)}")
    return errors


def rolling_score_errors(configs, rows):
    """Require exactly one bounded test-score row per declared rolling run."""
    expected = {name for name, cfg in configs.items() if cfg.get("rolling")}
    actual = defaultdict(list)
    for row in rows:
        if row.get("run") in expected:
            actual[row["run"]].append(row)
    errors = []
    missing = sorted(expected - set(actual))
    if missing:
        errors.append("rolling runs missing test scores: " + ", ".join(missing))
    for name, matches in sorted(actual.items()):
        if len(matches) != 1:
            errors.append(f"{name}: expected one rolling test score, found {len(matches)}")
        elif matches[0].get("n_test_months") != 12:
            errors.append(
                f"{name}: rolling test score covers "
                f"{matches[0].get('n_test_months')} months instead of 12"
            )
        elif (
            matches[0].get("test_end") is None
            or matches[0].get("val_end") is None
            or matches[0]["test_end"] - matches[0]["val_end"] != 12
        ):
            errors.append(f"{name}: saved rolling score boundaries are not 12 months")
        elif not matches[0].get("model_session_id"):
            errors.append(f"{name}: rolling test score is not bound to a model session")
    return errors


def _problem_summary(headline, problems, failure_file, limit=12):
    """Readable refusal: the first few reasons plus a pointer to the full list."""
    shown = problems[:limit]
    lines = [f"{headline}: {len(problems)} problem(s)"]
    lines += [f"  - {problem}" for problem in shown]
    if len(problems) > limit:
        lines.append(f"  ... {len(problems) - limit} more")
    lines.append(f"  full list: {failure_file}")
    return "\n".join(lines)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as fh:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_average(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_fields)].append(row)
    out = []
    for group, items in sorted(grouped.items()):
        row = dict(zip(group_fields, group))
        row["n_seeds"] = len({x["seed"] for x in items})
        for key in items[0]:
            if any(key.endswith("_" + metric) for metric in METRIC_KEYS):
                row[key] = float(np.nanmean([x[key] for x in items]))
        out.append(row)
    return out


def _result_averages(rows):
    """Compact backward-compatible leaderboard columns.

    `panel` is part of the grouping key, not decoration: results.csv is the
    table people sort by MSE, and it holds the national aggregate series, the
    per-year aggregate folds and the 28k-series panel at once. Carrying the
    panel only on the per-run rows in metrics.json -- where it started -- left
    the one table that actually gets read unable to say which rows are
    comparable. Grouping by it also means that if two rows ever disagreed on
    the panel within one (model, variant, encoder), they separate here instead
    of being silently averaged together.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[(_panel(row) if "panel" not in row else row["panel"],
                 row["model"], row["variant"], row["enc"])].append(row)
    out = []
    for (panel, model, variant, enc), items in sorted(grouped.items()):
        def mean(key):
            return float(np.nanmean([item[key] for item in items]))
        mse_values = [item["both_MSE"] for item in items]
        row = dict(
            panel=panel, model=model, variant=variant, enc=enc, n=len(items),
            MSE=mean("both_MSE"), MSE_std=float(np.nanstd(mse_values)),
            MAE=mean("both_MAE"), RMSE=mean("both_RMSE"),
            sMAPE=mean("both_sMAPE"), MASE=mean("both_MASE"),
        )
        # The selection signal, next to the test metric it must not have been
        # chosen from. Absent for the analytic baselines, which never trained.
        val = [item["val_loss"] for item in items if item.get("val_loss") is not None]
        if val:
            row["val_loss"] = float(np.nanmean(val))
        out.append(row)
    return out


def _pooled_rolling_rows(rows):
    """Pool equal-length annual Test-1.1 folds within each model and seed."""
    grouped = defaultdict(list)
    for row in rows:
        if str(row.get("variant", "")).startswith("aggregate_roll_y"):
            grouped[(row["model"], row["enc"], row["seed"])].append(row)
    out = []
    for (model, enc, seed), items in sorted(grouped.items()):
        row = dict(
            run=f"{model}_aggregate_roll_pooled_s{seed}", model=model,
            variant="aggregate_roll_pooled", enc=enc, seed=seed,
            n_folds=len(items),
            # Declared, not inferred. These rows are built here rather than by
            # _flatten_result, so without them _panel() sees no aggregate or
            # rolling marker and files the pooled Test 1.1 result under
            # "series" -- putting a pooled aggregate error next to the
            # 28k-series models, which is the exact comparability problem the
            # panel column exists to prevent, one level up.
            aggregate=True, rolling=True,
        )
        for channel in ("value", "weight", "both"):
            for metric in METRIC_KEYS:
                key = f"{channel}_{metric}"
                if metric == "RMSE":
                    row[key] = float(np.sqrt(np.mean([
                        item[f"{channel}_MSE"] for item in items
                    ])))
                else:
                    row[key] = float(np.mean([item[key] for item in items]))
        out.append(row)
    return out


def _disparity_rows(subgroup_averages):
    """Best/worst subgroup gaps for every reported error metric."""
    grouped = defaultdict(list)
    fields = ("model", "variant", "enc", "dimension")
    for row in subgroup_averages:
        grouped[tuple(row[k] for k in fields)].append(row)
    out = []
    for group, items in sorted(grouped.items()):
        base = dict(zip(fields, group))
        for key in items[0]:
            if not any(key.endswith("_" + metric) for metric in METRIC_KEYS):
                continue
            values = np.asarray([item[key] for item in items], dtype=np.float64)
            finite = np.isfinite(values)
            if not finite.any():
                continue
            valid_items = [item for item, keep in zip(items, finite) if keep]
            valid_values = values[finite]
            lo, hi = int(np.argmin(valid_values)), int(np.argmax(valid_values))
            minimum, maximum = float(valid_values[lo]), float(valid_values[hi])
            out.append(dict(
                **base, metric=key, best_group=valid_items[lo]["group"],
                worst_group=valid_items[hi]["group"], minimum=minimum,
                maximum=maximum, absolute_gap=maximum - minimum,
                worst_to_best_ratio=(maximum / minimum if minimum > 0 else float("nan")),
            ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/sweep")
    ap.add_argument("--npz", default="data/census_port/processed/census_lattice_9ch.npz")
    ap.add_argument("--seeds", default="947,732,619",
                    help="required seed cohort (default: 947,732,619)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="diagnostic only: write results despite missing/failed runs")
    args = ap.parse_args()
    runs_dir = Path(args.runs)
    runs_dir.mkdir(parents=True, exist_ok=True)
    expected_seeds = [int(x) for x in args.seeds.split(",") if x]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Bind the evaluation window/splits to the same composed data config the
    # sweep trains under, instead of CensusConfig's dataclass defaults.
    data_cfg = compose_data_config()
    cl = CensusLattice(census_config_from_config(data_cfg, npz=args.npz))
    cl_agg = None
    rolling_lattices = {}
    print(f"eval on {dev} | test months = {cl.test_end - cl.config.val_end}")

    checkpoints = {Path(p).parent.name: p for p in glob.glob(str(runs_dir / "*" / "best.pth"))}
    configs = {name: parse_run(name) for name in checkpoints}
    unrecognized = sorted(name for name, cfg in configs.items() if cfg is None)
    configs = {name: cfg for name, cfg in configs.items() if cfg is not None}
    if unrecognized:
        print("ignoring non-sweep checkpoint directories: " + ", ".join(unrecognized))
    planned_entries = _planned_entries(runs_dir)
    planned = None if planned_entries is None else set(planned_entries)
    provenance_problems = []
    if planned_entries is None:
        provenance_problems.append(
            "missing manifest.json; official evaluation requires a declared run matrix"
        )
    else:
        expected_npz = Path(args.npz).resolve()
        for name, entry in sorted(planned_entries.items()):
            error = validate_completion(runs_dir, entry)
            if error:
                provenance_problems.append(error)
            # Only compare fingerprints for runs that actually produced a
            # checkpoint. A never-trained cell would otherwise report both
            # "missing best.pth" and a fingerprint mismatch, doubling the
            # report and burying the failures that need reading.
            if (runs_dir / name / "best.pth").exists():
                try:
                    current_fingerprint = run_input_fingerprint(
                        entry.get("argv", []), Path(__file__).resolve().parent.parent
                    )
                except OSError as exc:
                    current_fingerprint = None
                    provenance_problems.append(
                        f"{name}: cannot fingerprint current config/lattice: {exc}"
                    )
                if (current_fingerprint is not None
                        and entry.get("input_fingerprint") != current_fingerprint):
                    provenance_problems.append(
                        f"{name}: current data/config/source fingerprint differs "
                        "from manifest"
                    )
            declared_npz = argv_value(entry.get("argv", []), "--npz")
            if declared_npz is None:
                provenance_problems.append(f"{name}: manifest command lacks --npz")
            else:
                declared_path = Path(declared_npz)
                if not declared_path.is_absolute():
                    declared_path = Path(__file__).resolve().parent.parent / declared_path
                if declared_path.resolve() != expected_npz:
                    provenance_problems.append(
                        f"{name}: trained on {declared_path.resolve()}, evaluation uses {expected_npz}"
                    )
    if planned is not None:
        extras = sorted(set(configs) - planned)
        if extras:
            print("ignoring checkpoints outside the saved manifest: " + ", ".join(extras))
        configs = {name: cfg for name, cfg in configs.items() if name in planned}
    problems = provenance_problems + seed_matrix_errors(configs, expected_seeds, planned)
    if not configs:
        problems.append("no recognized model checkpoints found")
    if problems and not args.allow_partial:
        failure_file = runs_dir / "evaluation_failures.json"
        failure_file.write_text(json.dumps(problems, indent=2))
        print(_problem_summary("evaluation refused", problems, failure_file))
        return 1

    def get_rolling_lattice(run_name):
        entry = planned_entries.get(run_name) if planned_entries else None
        if entry is None:
            raise ValueError(f"{run_name}: rolling evaluation requires its manifest entry")
        rolling_config = rolling_census_config(entry, args.npz, data_cfg)
        key = (
            rolling_config.train_end, rolling_config.val_end,
            rolling_config.test_end,
        )
        if key not in rolling_lattices:
            rolling_lattices[key] = CensusLattice(rolling_config)
        return rolling_lattices[key]

    rows, subgroups = [], []
    if not args.no_baselines:
        for name in BASELINES:
            try:
                P, T = eval_baseline(name, cl, dev)
                metrics = score(P, T, cl, combo=False)
                cfg = dict(model=name, variant="baseline", enc="none", seed=0)
                rows.append(_flatten_result(name, cfg, metrics))
                subgroups.extend(subgroup_rows(P, T, cl, False, cfg))
                print(f"  {name:46} MSE {metrics['both']['MSE']:.5f}  "
                      f"raw-sMAPE {metrics['both']['sMAPE']:5.1f}")
            except Exception as exc:
                problems.append(f"{name}: {type(exc).__name__}: {exc}")
                # Say so on stdout, exactly like a failed checkpoint. The whole
                # point of the baselines is to anchor the leaderboard; dropping
                # one silently leaves a table that looks complete and has no
                # persistence reference. (F6)
                print(f"  {name:46} BASELINE-FAIL {type(exc).__name__}: {exc}")

        # Analytic anchors for every annual aggregate fold. These are scored
        # but are not training experiments and therefore do not enter the
        # manifest or the declared training count.
        rolling_representatives = {}
        for run_name, cfg in configs.items():
            if cfg.get("rolling"):
                rolling_representatives.setdefault(cfg["test_year"], run_name)
        for year, run_name in sorted(rolling_representatives.items()):
            try:
                use_cl = get_rolling_lattice(run_name)
                for baseline in BASELINES:
                    P, T = eval_baseline(baseline, use_cl, dev)
                    metrics = score(P, T, use_cl, combo=False)
                    identity = dict(
                        model=baseline, variant=f"aggregate_roll_y{year}",
                        enc="aggregate", seed=0, test_year=year,
                        # Without these the row lands in the "series" panel and
                        # its error -- computed on one aggregate fold -- sits in
                        # the leaderboard beside the 28k-series models as if it
                        # were comparable.
                        aggregate=True, rolling=True,
                        train_end=use_cl.config.train_end,
                        val_end=use_cl.config.val_end,
                        test_end=use_cl.test_end,
                        n_test_months=use_cl.test_end - use_cl.config.val_end,
                        model_session_id="analytic_baseline",
                    )
                    rows.append(_flatten_result(
                        f"{baseline}_aggregate_roll_y{year}", identity, metrics,
                    ))
                    print(f"  {baseline + '_aggregate_roll_y' + str(year):46} "
                          f"MSE {metrics['both']['MSE']:.5f}  "
                          f"raw-sMAPE {metrics['both']['sMAPE']:5.1f}")
            except Exception as exc:
                problems.append(
                    f"rolling baselines y{year}: {type(exc).__name__}: {exc}"
                )

        # Analytic anchors for the FIXED aggregate split -- Test 1's task, and
        # the paper's T-1 row.
        #
        # Panel and split scheme are separate axes, and the anchors covered
        # only three of the four cells: series/fixed and aggregate/rolling and
        # series/rolling, but not aggregate/fixed. So Test 1's leaderboard
        # could say which trained model won, never whether training beat doing
        # nothing on the same series under the same boundaries.
        if any(c.get("aggregate") and not c.get("rolling")
               for c in configs.values()):
            try:
                if cl_agg is None:
                    cl_agg = CensusLattice(census_config_from_config(
                        data_cfg, npz=args.npz, aggregate=True))
                for baseline in BASELINES:
                    P, T = eval_baseline(baseline, cl_agg, dev)
                    metrics = score(P, T, cl_agg, combo=False)
                    identity = dict(
                        model=baseline, variant="aggregate", enc="aggregate",
                        seed=0,
                        # aggregate=True keeps the row out of the series panel,
                        # where an error computed on one aggregated series would
                        # sit beside the 28k-series models as if comparable.
                        # rolling is absent on purpose: this is the fixed split.
                        aggregate=True,
                        train_end=cl_agg.config.train_end,
                        val_end=cl_agg.config.val_end,
                        test_end=cl_agg.test_end,
                        n_test_months=cl_agg.test_end - cl_agg.config.val_end,
                        model_session_id="analytic_baseline",
                    )
                    rows.append(_flatten_result(
                        f"{baseline}_aggregate", identity, metrics,
                    ))
                    print(f"  {baseline + '_aggregate':46} "
                          f"MSE {metrics['both']['MSE']:.5f}  "
                          f"raw-sMAPE {metrics['both']['sMAPE']:5.1f}")
            except Exception as exc:
                problems.append(
                    f"aggregate baselines: {type(exc).__name__}: {exc}"
                )

    for name in sorted(configs):
        cfg = configs[name]
        use_cl = cl
        if cfg.get("rolling"):
            try:
                use_cl = get_rolling_lattice(name)
            except Exception as exc:
                problems.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
        elif cfg.get("aggregate"):
            if cl_agg is None:
                cl_agg = CensusLattice(census_config_from_config(
                    data_cfg, npz=args.npz, aggregate=True))
            use_cl = cl_agg
        try:
            P, T = eval_ckpt(checkpoints[name], cfg, use_cl, dev)
            metrics = score(P, T, use_cl, cfg["combo"])
            if cfg.get("rolling"):
                session = json.loads(
                    (runs_dir / name / "model_session.json").read_text()
                )
                cfg.update(
                    train_end=use_cl.config.train_end,
                    val_end=use_cl.config.val_end,
                    test_end=use_cl.test_end,
                    model_session_id=session["session_id"],
                )
                cfg["n_test_months"] = use_cl.test_end - use_cl.config.val_end
            identity = {k: cfg[k] for k in ("model", "variant", "enc", "seed")}
            rows.append(_flatten_result(name, cfg, metrics, runs_dir=runs_dir))
            subgroups.extend(subgroup_rows(P, T, use_cl, cfg["combo"], identity))
            print(f"  {name:46} MSE {metrics['both']['MSE']:.5f}  "
                  f"raw-sMAPE {metrics['both']['sMAPE']:5.1f}")
        except Exception as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  {name:46} EVAL-FAIL {type(exc).__name__}: {exc}")

    problems.extend(rolling_score_errors(configs, rows))

    if problems:
        failure_file = runs_dir / "evaluation_failures.json"
        failure_file.write_text(json.dumps(problems, indent=2))
        if not args.allow_partial:
            print(_problem_summary("evaluation failed; official result files were "
                                   "not updated", problems, failure_file))
            return 1
        print(f"{len(problems)} problem(s) recorded in {failure_file}")
    else:
        failure_file = runs_dir / "evaluation_failures.json"
        if failure_file.exists():
            failure_file.unlink()

    suffix = ".partial" if args.allow_partial else ""
    (runs_dir / f"metrics{suffix}.json").write_text(json.dumps(rows, indent=2))
    averaged = _result_averages(rows)
    _write_csv(runs_dir / f"results{suffix}.csv", averaged)
    rolling_rows = [
        row for row in rows
        if str(row.get("variant", "")).startswith("aggregate_roll_y")
    ]
    _write_csv(runs_dir / f"rolling_test_scores{suffix}.csv", rolling_rows)
    _write_csv(
        runs_dir / f"rolling_annual{suffix}.csv",
        _result_averages(rolling_rows),
    )
    _write_csv(
        runs_dir / f"rolling_pooled{suffix}.csv",
        _result_averages(_pooled_rolling_rows(rolling_rows)),
    )
    _write_csv(runs_dir / f"subgroups{suffix}.csv", subgroups)
    subgroup_avg = _seed_average(
        subgroups, ("model", "variant", "enc", "dimension", "group"))
    _write_csv(runs_dir / f"subgroups_seed_averaged{suffix}.csv", subgroup_avg)
    _write_csv(runs_dir / f"subgroup_disparities{suffix}.csv",
               _disparity_rows(subgroup_avg))
    print(f"wrote {len(rows)} run metrics and {len(subgroups)} subgroup rows to {runs_dir}"
          + (" (diagnostic partial outputs)" if suffix else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
