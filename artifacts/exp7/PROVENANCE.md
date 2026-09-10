# Exp 7 (fa_local) — rescued record

Trained on the AutoDL box `region-42.seetacloud.com:23629` (single H20), which
became unreachable during a switch to a 5-card machine. These are the numbers
as evaluated ON that box; the checkpoints and per-run logs stayed there.

## Status of the run

- 27/27 complete, 0 failed, 0 diverged. Chain log: `Exp 7 complete: 27/27` at
  07:00:14 on 2026-09-02.
- Evaluated CLEANLY: no `evaluation_failures.json`, `--allow-partial` NOT used.
- All 27 `input_fingerprint`s verified against the manifest (27/27) after
  restoring the tree to `5e93b59`, the commit the manifest was written under.
  The box had been synced to `30efd49` mid-sweep; only `scripts/sweep.py`
  differs between those commits and no run imports it.

## Learning rates (Exp 0 `mechlocal` arm, 15 runs, 20 epochs)

    gru 1e-4    lstm 1e-2    transformer 1e-4

CAVEAT the paper must carry: select_lr reported `grid endpoints won in 3
cell(s)` — all three rates sit at the boundary of the {1e-4 … 1e-2} grid, so
the optimum may lie outside it. Extending the grid one step and re-probing is
the stated remedy.

## What this file is NOT

The checkpoints (`best.pth`), per-run `cost.json` / `logs/metrics.json`, the
manifest, and the 40920 subgroup fairness rows were left on the old box. If the
AutoDL data disk did not migrate, they are gone, and significance testing for
these cells (dump_errors.py is machine-bound) would need Exp 7 re-run on the
new machine.

## Headline

Against the Test 3 cells (embeddings, blind, 3 seeds, same protocol):

- `fa_local` beats `asa` in **5/9** cells. The old workbook, same pairing
  (`cafa_*` vs `cross_attention_*`), had CaFA winning 12/12 GRU, 12/12 LSTM,
  3/12 Transformer. The draft's advantage did not survive the protocol change,
  and three of the five surviving wins sit in high-variance cells
  (lstm 3d ±0.167, lstm 4d ±0.204, transformer 4d's comparator ±0.103).
- `fa_local` beats `fa` (the authors' released operator) in **9/9** cells,
  by up to 29% (gru 2d: 0.6458 vs 0.8357) — while the authors' operator carries
  5.5–11x the mixer parameters.
