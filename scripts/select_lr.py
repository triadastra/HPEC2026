#!/usr/bin/env python
"""Exp 0: pick each (arm, model) learning rate from VALIDATION loss.

Reads an Exp-0 sweep session (``scripts/sweep.py --tests 0``) and writes
``lr_selection.json``, which ``sweep.py --lr-selection`` then injects into every
run of the main matrix.

Why selection lives in its own stage
------------------------------------
The learning rate has to be chosen on validation data. Test metrics for all
candidate rates are printed side by side by ``scripts/evaluate.py``, and
choosing the best row of that table would be selecting on the test set. The
per-epoch validation curve each run already writes to ``logs/metrics.json`` is
the only selection signal used here; no checkpoint is evaluated on test.

What this script refuses to do
------------------------------
The rest of the pipeline fails closed -- the lattice loader rejects an artifact
that cannot prove train-only normalization, the manifest rejects a fingerprint
mismatch, evaluation rejects an incomplete seed matrix. The learning-rate
choice is the one input that decides cross-arm fairness, so it fails closed too:

* a cell missing any candidate rate is refused, not silently selected from the
  rates that happen to be present;
* a run without a validated completion record is refused;
* a cell whose ranking is not stable inside the truncated window is refused,
  because a rate chosen from a truncated curve that has not settled is not a
  selection, it is a coin flip (``--allow-unstable`` downgrades this to a
  warning for diagnostic use, and marks the output);
* a winner sitting on an endpoint of the grid is reported as such: the grid was
  too narrow and the true optimum may lie outside it.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.utils import validate_completion
# Imported, not re-declared: the arm names and the divergence sentinel are the
# contract between the two scripts, and a second copy here would drift silently.
from scripts.sweep import (DIVERGED_FILE, EXP0_CHECK_ARMS, EXP0_CHECK_BASE,
                          exp0_arm_pattern)

SCHEMA_VERSION = 1

# exp0_<arm>_<model>_<lrtag>_s<seed>; model may contain underscores (mamba_nd),
# so the arm set and the lr tag anchor the split rather than a plain split("_").
# The alternation comes from sweep.py, which declares the arms. Spelling it
# out here as well is how an arm added on one side and missing on the other
# would make its runs unparseable -- and an unparseable run refuses the whole
# selection.
RUN_RE = re.compile(
    r"^exp0_(" + exp0_arm_pattern() + r")_(.+)_(lr[0-9]+e[0-9]+)_s(\d+)$")


def is_check_arm(arm):
    """True for arms that only TEST a transfer assumption.

    They are ranked and reported like any other cell, but never contribute a
    selected rate: their whole purpose is to answer "does the rate chosen
    elsewhere still win here?".
    """
    return any(arm.startswith(prefix) for prefix in EXP0_CHECK_ARMS)


def parse_exp0_name(name):
    m = RUN_RE.match(name)
    if not m:
        return None
    arm, model, tag, seed = m.groups()
    return dict(arm=arm, model=model, tag=tag, seed=int(seed))


def manifest_entries(runs_dir):
    path = runs_dir / "manifest.json"
    if not path.exists():
        sys.exit(f"{path} is required: Exp 0 must be run through scripts/sweep.py")
    data = json.loads(path.read_text())
    entries = data.get("runs", data) if isinstance(data, dict) else data
    return {e["name"]: e for e in entries if isinstance(e, dict)}


def argv_lr(entry):
    """The declared rate for a run, from the manifest command itself."""
    argv = entry.get("argv", [])
    if "--lr" not in argv:
        return None
    return argv[argv.index("--lr") + 1]


def val_curve(run_dir):
    """[(epoch, val_loss)] from the trainer's per-epoch log, epoch-ordered."""
    path = run_dir / "logs" / "metrics.json"
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    out = []
    for row in rows:
        if not isinstance(row, dict) or "val_loss" not in row:
            continue
        value = row["val_loss"]
        if value is None or value != value:      # drop NaN/None epochs
            continue
        out.append((int(row.get("epoch", len(out))), float(value)))
    return sorted(out) or None


def best_val(curves, upto):
    """Mean over seeds of each seed's best validation loss before ``upto``.

    ``curves`` is a LIST of per-seed curves, not one curve. Exp 0 defaults to a
    single seed, but ``--seeds`` widens it, and keying only by (arm, model, lr)
    would then let the last seed read silently overwrite the others -- a
    selection that claims to use N seeds while using one.
    """
    per_seed = []
    for curve in curves:
        values = [v for e, v in curve if e < upto]
        if values:
            per_seed.append(min(values))
    return sum(per_seed) / len(per_seed) if per_seed else None


def rank_key(candidates, upto):
    """(best_val, lr) per candidate -- lr breaks ties toward the smaller rate.

    A tie means the data cannot separate the two rates; taking the smaller one
    is the more conservative choice and, unlike "first in dict order", it does
    not depend on how the candidates happened to be enumerated.
    """
    ranked = {}
    for lr, curves in candidates.items():
        value = best_val(curves, upto)
        if value is not None:
            ranked[lr] = (value, float(lr))
    return ranked


def pick(candidates, window, mid):
    """Winner within the window, plus whether the ranking had settled by ``mid``."""
    end_rank = rank_key(candidates, window)
    mid_rank = rank_key(candidates, mid)
    if not end_rank or not mid_rank:
        return None, None, False
    winner = min(end_rank, key=lambda k: end_rank[k])
    early = min(mid_rank, key=lambda k: mid_rank[k])
    return winner, early, winner == early


def _dim_transfer(diagnostics):
    """Does the rate selected at the 3-D probe still win at 2-D and 4-D?

    Exp 0 probes the N-D arm at one dimensionality and applies the result to
    all of them; the dim cells exist to check that transfer. The 3-D selection
    is therefore part of the comparison -- dim2 and dim4 agreeing with each
    other is not agreement with the rate they will actually train at, and
    reporting it as "ok" would silence the warning precisely when the transfer
    fails.
    """
    by_model = defaultdict(dict)
    for key, d in diagnostics.items():
        arm, model = key.split("/", 1)
        if arm.startswith("dim"):
            by_model[model][arm] = d["selected"]
    notes, agree = [], {}
    for model, dims in sorted(by_model.items()):
        chosen = diagnostics.get(f"nd/{model}")
        if chosen is None:
            notes.append(f"dim-transfer: no nd selection for {model} to check "
                         "against")
            continue
        arms = {"nd": chosen["selected"], **dims}
        rates = set(arms.values())
        agree[model] = dict(rates=arms, consistent=len(rates) == 1)
        if len(rates) > 1:
            notes.append(
                f"dim-transfer WARNING: {model} selects different rates by "
                f"dimensionality ({arms}); a single N-D rate is not supported "
                "for it -- probe that backbone per dimension."
            )
        else:
            notes.append(f"dim-transfer ok: {model} selects "
                         f"{next(iter(rates))} at the 3-D probe and every "
                         "checked dimensionality")
    return dict(models=agree, notes=notes)


def _transfer_checks(diagnostics):
    """For each check arm: does the rate selected on its base arm still win?

    Every check arm corresponds to one axis along which the probe differs from
    the runs it selects for -- fold length, encoder, axis identity. A
    disagreement does not invalidate the selection; it says that axis needs its
    own probe rather than a shared rate, and it says so in the artifact instead
    of leaving the assumption implicit.
    """
    # No "mech" entry: mech is a SELECTING arm (EXP0_ARMS) since its check-arm
    # version showed the asa->fa inheritance reverses a cell, so its runs get
    # their own rate rather than testing a transfer.
    LABEL = {
        "rollend": ("the shortest fold", "the longest fold"),
        "encflat": ("the embeddings encoder", "the one-hot encoder"),
        "encnd": ("the embeddings combo encoder", "the one-hot combo encoder"),
        "axid": ("no axis identity", "--axis-identity"),
    }
    report = {}
    for check_arm, base_arm in sorted(EXP0_CHECK_BASE.items()):
        models, notes = {}, []
        for key, d in sorted(diagnostics.items()):
            arm, model = key.split("/", 1)
            if arm != check_arm:
                continue
            chosen = diagnostics.get(f"{base_arm}/{model}")
            if chosen is None:
                notes.append(f"{check_arm}: no {base_arm} selection for {model} "
                             "to check against")
                continue
            base_label, check_label = LABEL.get(check_arm, (base_arm, check_arm))
            consistent = chosen["selected"] == d["selected"]
            models[model] = dict(base=chosen["selected"], check=d["selected"],
                                 consistent=consistent)
            if consistent:
                notes.append(f"{check_arm}-transfer ok: {model} selects "
                             f"{d['selected']} under both {base_label} and "
                             f"{check_label}")
            else:
                notes.append(
                    f"{check_arm}-transfer WARNING: {model} selects "
                    f"{chosen['selected']} under {base_label} but "
                    f"{d['selected']} under {check_label}; a shared rate across "
                    f"that axis is not supported for it -- probe it separately."
                )
        if models or notes:
            report[check_arm] = dict(base_arm=base_arm, models=models, notes=notes)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "outputs/exp0"),
                    help="Exp 0 session directory (sweep.py --runs-dir)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <runs>/lr_selection.json)")
    ap.add_argument("--window", type=int, default=None,
                    help="select on epochs [0, window). Default: every logged "
                         "epoch. Truncating is fine for ranking rates, but it "
                         "mildly favours larger rates -- state it in the paper.")
    ap.add_argument("--stability-fraction", type=float, default=0.6,
                    help="ranking must already hold at this fraction of the "
                         "window for the cell to be accepted")
    ap.add_argument("--allow-unstable", action="store_true",
                    help="diagnostic only: keep cells whose ranking had not "
                         "settled, and mark the output as unofficial")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs).resolve()
    entries = manifest_entries(runs_dir)

    problems, curves = [], defaultdict(dict)   # (arm,model) -> {lr: [curves]}
    sources, seeds_seen = {}, set()
    diverged_cells = {}
    for name, entry in sorted(entries.items()):
        parsed = parse_exp0_name(name)
        if parsed is None:
            problems.append(f"{name}: not an Exp 0 run name")
            continue
        lr_from_argv = argv_lr(entry)
        if lr_from_argv is None:
            problems.append(f"{name}: manifest command carries no --lr")
            continue
        if (runs_dir / name / DIVERGED_FILE).exists():
            # Attempted and answered: excluded from ranking, but the cell still
            # counts as having covered this rate.
            curves[(parsed["arm"], parsed["model"])].setdefault(lr_from_argv, [])
            diverged_cells.setdefault(
                f"{parsed['arm']}/{parsed['model']}", []).append(lr_from_argv)
            seeds_seen.add(parsed["seed"])
            continue
        error = validate_completion(runs_dir, entry)
        if error:
            problems.append(error)      # already prefixed with the run name
            continue
        curve = val_curve(runs_dir / name)
        if curve is None:
            problems.append(f"{name}: no usable val_loss curve in logs/metrics.json")
            continue
        curves[(parsed["arm"], parsed["model"])].setdefault(
            lr_from_argv, []).append(curve)
        sources[name] = entry.get("fingerprint")
        seeds_seen.add(parsed["seed"])

    if not curves:
        print("no Exp 0 runs found: " + ("; ".join(problems) or "empty manifest"))
        return 1

    # A rate must train on every declared seed. Never rank only its survivors.
    for (arm, model), candidates in curves.items():
        for lr in diverged_cells.get(f"{arm}/{model}", []):
            candidates[lr] = []

    # Every cell must offer the same candidate grid. A cell that lost one rate
    # to a crash would otherwise be "selected" from a smaller search than its
    # peers -- reintroducing, inside Exp 0, the asymmetry Exp 0 exists to remove.
    grid = sorted({lr for cells in curves.values() for lr in cells}, key=float)
    for (arm, model), cells in sorted(curves.items()):
        missing = [lr for lr in grid if lr not in cells]
        if missing:
            problems.append(f"{arm}/{model}: missing rates {missing}")
        # Uneven seed coverage is the same failure in a subtler form: a rate
        # averaged over more seeds is estimated more precisely than its rivals,
        # so the comparison is no longer like-for-like. Diverged rates carry no
        # curves by design and are excluded, or every divergence would read as
        # uneven coverage.
        counts = {lr: len(runs) for lr, runs in sorted(cells.items()) if runs}
        if len(set(counts.values())) > 1:
            problems.append(f"{arm}/{model}: uneven seed coverage across rates "
                            f"{counts}")

    longest = max((len(curve) for cells in curves.values()
                   for runs in cells.values() for curve in runs), default=0)
    window = args.window if args.window is not None else longest
    if args.window is not None and args.window > longest:
        print(f"note: --window {args.window} exceeds the longest logged curve "
              f"({longest} epochs); windows are clamped per cell")

    selected = defaultdict(dict)
    diagnostics, unstable = {}, []
    for (arm, model), cells in sorted(curves.items()):
        if any(lr not in cells for lr in grid):
            continue
        # Clamp the window to what this cell actually logged. Early stopping
        # routinely ends a run before the requested window, and an unclamped
        # window makes the stability check vacuous: if every curve is shorter
        # than `mid`, the midpoint and the endpoint see identical data and the
        # ranking "agrees" for free. Clamping keeps at least the longest curve
        # in the cell contributing epochs between mid and the end.
        cell_len = max((len(curve) for runs in cells.values() for curve in runs), default=0)
        if not cell_len:
            problems.append(f"{arm}/{model}: no eligible rate; every candidate diverged")
            continue
        cell_window = min(window, cell_len)
        cell_mid = max(1, int(round(cell_window * args.stability_fraction)))
        winner, early, stable = pick(cells, cell_window, cell_mid)
        if winner is None:
            problems.append(f"{arm}/{model}: no validation data inside the window")
            continue
        at_endpoint = winner in (grid[0], grid[-1])
        diagnostics[f"{arm}/{model}"] = dict(
            selected=winner, winner_at_epoch_fraction=early, ranking_stable=stable,
            at_grid_endpoint=at_endpoint,
            window_epochs=cell_window, stability_epoch=cell_mid,
            best_val={lr: best_val(cells[lr], cell_window) for lr in grid},
        )
        if not stable:
            unstable.append(f"{arm}/{model} (winner over epochs [0,{cell_window}) "
                            f"is {winner}, but [0,{cell_mid}) says {early})")
            if not args.allow_unstable:
                continue
        if is_check_arm(arm):
            continue                     # transfer check only; not a selection
        selected[arm][model] = winner

    dim_report = _dim_transfer(diagnostics)
    transfer_report = _transfer_checks(diagnostics)

    if unstable and not args.allow_unstable:
        problems.extend(f"unstable ranking: {u}" for u in unstable)
    if problems:
        print("lr selection refused:\n  " + "\n  ".join(problems))
        (runs_dir / "lr_selection_failures.json").write_text(
            json.dumps(problems, indent=2))
        return 1

    payload = dict(
        schema_version=SCHEMA_VERSION,
        rule="min val_loss over epochs [0, window); ties -> smaller lr",
        window_epochs_requested=window,
        stability_fraction=args.stability_fraction,
        grid=grid, seeds=sorted(seeds_seen),
        official=not args.allow_unstable,
        selected={arm: dict(sorted(cells.items()))
                  for arm, cells in sorted(selected.items())},
        # Rates that were tried and did not train. Reporting these is the point
        # of probing a wide grid: an endpoint that diverges is what shows the
        # grid actually brackets the optimum.
        diverged={k: sorted(v) for k, v in sorted(diverged_cells.items())},
        dim_transfer=dim_report,
        transfer_checks=transfer_report,
        diagnostics=diagnostics,
        sources=dict(sorted(sources.items())),
    )
    out = Path(args.out) if args.out else runs_dir / "lr_selection.json"
    out.write_text(json.dumps(payload, indent=2))

    for arm, cells in sorted(selected.items()):
        print(f"{arm:5} " + "  ".join(f"{m}={lr}" for m, lr in sorted(cells.items())))
    endpoints = [k for k, d in diagnostics.items() if d["at_grid_endpoint"]]
    if endpoints:
        print(f"\ngrid endpoints won in {len(endpoints)} cell(s): "
              + ", ".join(sorted(endpoints)))
        print("the optimum may lie outside the grid -- extend it one step that "
              "way and re-probe those cells.")
    if diverged_cells:
        print("\ndiverged (tried, did not train):")
        for cell, rates in sorted(diverged_cells.items()):
            print(f"  {cell}: {', '.join(sorted(rates))}")
    for line in dim_report.get("notes", []):
        print(line)
    for check in transfer_report.values():
        for line in check["notes"]:
            print(line)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
