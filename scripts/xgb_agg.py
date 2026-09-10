#!/usr/bin/env python
"""XGBoost on the aggregate test (Test 1), protocol-matched.

Same data pipeline as the neural runs (CensusLattice aggregate=True -> the
loader's feature panel: 2 target channels + 12 lags each + month sin/cos = 28
features per month, log1p + train-only MinMax). Tabular framing flattens the
same 36-month ``[T-36,T)`` window consumed by the neural models, so every model
in the aggregate experiment receives identical history. One XGBRegressor per target channel,
early-stopped on the validation window. Splits identical to the benchmark:
targets in [48,144) train, [144,168) val, [168,192) test. Seeds 947/732/619.

    CUDA_VISIBLE_DEVICES="" python scripts/xgb_agg.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data.census_loader import (  # noqa: E402
    CensusLattice,
    census_config_from_config,
)
from src.utils import compose_data_config  # noqa: E402

SEEDS = (947, 732, 619)


def window_xy(feat, targ, lo, hi, input_len):
    """Flatten the exact [t-input_len, t) window consumed by neural models."""
    times = range(lo, hi)
    x = np.stack([feat[t - input_len:t].reshape(-1) for t in times])
    return x, targ[lo:hi]


def metrics(P, Y, cl):
    e = (P - Y).astype(np.float64)
    mse = float((e ** 2).mean())
    sid = np.zeros(len(P), dtype=np.int64)
    raw_p, raw_y = cl.inverse_targets(P, sid), cl.inverse_targets(Y, sid)
    sm = float((2 * np.abs(raw_p - raw_y) /
                (np.abs(raw_p) + np.abs(raw_y) + 1e-8)).mean() * 100)
    return mse, float(np.abs(e).mean()), float(mse ** 0.5), sm


def main():
    # Imported here, not at module scope. xgboost is only needed to RUN the
    # tabular arm, but tests/test_review_regressions.py imports window_xy from
    # this module -- and a hard top-level import made the whole test module
    # fail to collect on any box without xgboost, silently taking ~30
    # unrelated regression tests (including the declared-matrix size
    # assertions) out of the run. src/models/__init__.py already treats
    # xgboost as optional; this brings the script in line.
    import xgboost as xgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(
        REPO / "data/census_port/processed/census_lattice_9ch.npz"))
    ap.add_argument("--out", default=str(REPO / "outputs/sweep/xgboost_aggregate.json"))
    args = ap.parse_args()

    cl = CensusLattice(census_config_from_config(
        compose_data_config(), npz=args.npz, aggregate=True))
    feat = cl.feat[0]
    targ = cl.panel[0]
    burn = cl.config.input_len + cl.config.lag_count
    Xtr, Ytr = window_xy(feat, targ, burn, cl.config.train_end, cl.config.input_len)
    Xva, Yva = window_xy(
        feat, targ, cl.config.train_end, cl.config.val_end, cl.config.input_len
    )
    Xte, Yte = window_xy(feat, targ, cl.config.val_end, cl.T, cl.config.input_len)

    all_p = []
    for seed in SEEDS:
        preds = []
        for ch in range(2):
            model = xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=4,
                subsample=0.9, colsample_bytree=0.9,
                early_stopping_rounds=30, random_state=seed,
                tree_method="hist", n_jobs=8,
            )
            model.fit(Xtr, Ytr[:, ch], eval_set=[(Xva, Yva[:, ch])], verbose=False)
            preds.append(model.predict(Xte))
        all_p.append(np.stack(preds, axis=1))

    per_seed = [metrics(prediction, Yte, cl) for prediction in all_p]
    avg = tuple(float(np.mean([row[i] for row in per_seed])) for i in range(4))
    std = float(np.std([row[0] for row in per_seed]))
    print(f"XGBoost agg (3 seeds): MSE {avg[0]:.6f} +- {std:.6f}  "
          f"MAE {avg[1]:.6f}  RMSE {avg[2]:.6f}  raw-sMAPE {avg[3]:.2f}")
    for seed, row in zip(SEEDS, per_seed):
        print(f"  seed {seed}: MSE {row[0]:.6f} MAE {row[1]:.6f}")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "model": "xgboost", "experiment": "aggregate",
        "input_len": cl.config.input_len,
        "features_per_timestep": cl.features_per_group,
        "flattened_features": int(Xtr.shape[1]),
        "seeds": list(SEEDS),
        "per_seed": [dict(seed=seed, MSE=row[0], MAE=row[1], RMSE=row[2], sMAPE=row[3])
                     for seed, row in zip(SEEDS, per_seed)],
        "average": dict(MSE=avg[0], MSE_std=std, MAE=avg[1], RMSE=avg[2], sMAPE=avg[3]),
    }, indent=2))
    print(f"-> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
