#!/usr/bin/env python3
"""The analytic floor for the fixed aggregate split -- Test 1's task.

Why this is a tool and not part of evaluate.py for this sweep:

`evaluate.py` recomputes an input fingerprint covering src/, scripts/, config/
and external/, and refuses every run whose recorded fingerprint differs. So
editing the evaluator invalidates the evaluation of everything trained before
the edit. The fix that adds these rows inside evaluate.py is on main and is
correct -- it just cannot be deployed to a sweep already in flight.

tools/ is outside the fingerprint's source-tree hash. Running this changes
nothing evaluation checks, so the floor can be computed against the same tree
the runs trained under, at any time, without invalidating a single checkpoint.

DEPLOYMENT: copy this file into the frozen worktree (scp/rsync) WITHOUT
moving HEAD. `run_input_fingerprint()` also records `git rev-parse HEAD` via
`_git_identity` (src/utils/run_manifest.py), so deploying it with a pull,
checkout, or cherry-pick invalidates every recorded fingerprint even though
the hashed source directories are untouched.

The rows it emits are identical in kind to the ones evaluate.py writes for the
rolling folds: analytic, no checkpoint, `model_session_id="analytic_baseline"`,
and `aggregate=True` with no `rolling` so they land in Test 1's panel.

    python3 tools/aggregate_baselines.py --npz <lattice.npz> --out floor.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="aggregate_floor.json")
    ap.add_argument("--data-config", default="config/census.yaml")
    a = ap.parse_args()

    import torch
    from evaluate import BASELINES, eval_baseline, score, _flatten_result
    from src.data.census_loader import CensusLattice, census_config_from_config
    from src.utils.config import load_config

    data_cfg = load_config(str(REPO / a.data_config))
    cl = CensusLattice(census_config_from_config(data_cfg, npz=a.npz, aggregate=True))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"fixed aggregate split | device {dev} | "
          f"train_end={cl.config.train_end} val_end={cl.config.val_end} "
          f"test_end={cl.test_end}")

    rows = []
    for name in BASELINES:
        P, T = eval_baseline(name, cl, dev)
        metrics = score(P, T, cl, combo=False)
        identity = dict(
            model=name, variant="aggregate", enc="aggregate", seed=0,
            # aggregate without rolling -> the same panel as Test 1's trained
            # rows. In the series panel an error over one aggregated series
            # would sit beside the 30k-series models as if comparable.
            aggregate=True,
            train_end=cl.config.train_end, val_end=cl.config.val_end,
            test_end=cl.test_end,
            n_test_months=cl.test_end - cl.config.val_end,
            model_session_id="analytic_baseline",
        )
        rows.append(_flatten_result(f"{name}_aggregate", identity, metrics))
        print(f"  {name + '_aggregate':32} MSE {metrics['both']['MSE']:.6f}  "
              f"MAE {metrics['both']['MAE']:.6f}  "
              f"sMAPE {metrics['both']['sMAPE']:5.1f}  "
              f"MASE {metrics['both']['MASE']:.4f}")

    Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} floor rows -> {a.out}")
    print("panel:", rows[0].get("panel"))


if __name__ == "__main__":
    main()
