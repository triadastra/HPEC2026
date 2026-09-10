# Artifacts

Derived results small enough to version, kept here because losing them costs
GPU time to reproduce and because `outputs/` is not tracked.

## `lr_selection_fa.json`

The fa (`mech`) arm of Exp 0 — the learning rate every fa cell in the matrix
trains at. Produced 2026-08-28 by

```bash
python scripts/sweep.py --tests 0 --exp0-arms mech --epochs 20 \
  --runs-dir outputs/sweep/exp0_fa --gpus 0,1,2,3,4
python scripts/select_lr.py --runs outputs/sweep/exp0_fa
```

and folded into a run with `--lr-selection-patch`, which never rewrites the
base selection. `official: true` — no cell's ranking was unsettled, unlike the
base selection (see future_work.md).

It exists because fa does **not** inherit the N-D rate. Every one of the five
backbones selects a different rate than the `nd` arm does:

| backbone | `nd` rate | `mech` rate | factor |
|---|---|---|---|
| gru | 3e-4 | 1e-2 | 33x |
| lstm | 1e-4 | 1e-3 | 10x |
| mamba2 | 3e-3 | 1e-4 | 1/30 |
| mamba3 | 1e-4 | 1e-3 | 10x |
| transformer | 1e-3 | 3e-4 | 1/3.3 |

25 cells, one seed (947), grid `1e-4 … 1e-2`. The `sources` block records the
fingerprint of every run the selection was read from.
