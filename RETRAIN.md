# RETRAIN.md — protocol fix, code fixes, and what has to be recomputed

**Status:** code changed, nothing re-run yet. The WCTR workbooks and figures
have been removed from this repository; they are not HPEC evidence. Any existing
HPEC sweep metrics or figures were produced under the old cohort/protocol and
must not be reused.

**Superseding retraining decision:** rebuild the Census lattice, then retrain
the entire declared HPEC matrix. The cohort is now frozen from training months
only, the normalization artifact now stores its exact floored range,
validation checkpoint selection is observation-weighted, and the default cap
is 200 epochs. These changes make selective checkpoint reuse indefensible,
even for model families unaffected by the architecture-specific findings
below.

Findings are referenced as **F1–F13** from the code review of the model roster
and setup (2026-08-23).

### Original pipeline-review fixes completed after F1–F13

- Cohort density, state ranking, and commodity pruning use 2010–2021 only.
- The lattice stores `norm_range`; inversion uses the same range floor as the
  forward transform.
- Evaluation computes sMAPE in inverted real units, keeps normalized MSE as
  the selection/reporting objective, and uses pooled two-channel RMSE.
- Evaluation writes no official leaderboard unless its saved sweep manifest
  and every required seed are complete and every checkpoint loads.
- Fairness files report state, commodity, flow, training-scale quartile, and
  training-density quartile performance for every run and seed average.
- Validation loss is weighted by valid target observations, not batch count.
- The dormant basic categorical attention uses three tokens; its query/key
  projections now learn. Dormant FiLM projections are constructed before
  optimizer/device setup. Neither variant is in the HPEC sweep roster.
- The broken predecessor Predictor API and old shared metric implementation
  have been deleted; the Census entry point is
  `scripts/evaluate.py` plus `CensusLattice.inverse_targets()`.
- `requirements-cuda.txt` and `scripts/check_environment.py` define and verify
  the full CUDA/Mamba dependency contract.

---

## 1. The protocol change (F1) — step-matched training

### What was wrong

The flat and combo paths define a training *sample* differently. The counts
below are the old 28,292-series cohort example; the rebuilt sidecar supplies
the corrected cohort size dynamically:

| path | one sample is | train samples/epoch |
|---|---|---|
| `CensusDataset` (flat) | one series at one month | 28,292 × 96 = 2,716,032 |
| `CensusComboDataset` (combo/axial/grid) | one month, **all** series | 96 |

Both consume the same 2,716,032 target values per epoch — there was never a
data-exposure gap. What differed was how those values were chopped into
optimizer steps:

| arm | micro-batch | steps/epoch | steps over 30 epochs | effective batch |
|---|---|---|---|---|
| flat (old) | 512 | 5,304 | ~159,000 | 512 |
| combo (old & new) | 1 month | 96 | 2,880 | 28,292 |
| **flat (new)** | **13×2,048 + 1×1,668** | **96** | **2,880** | **28,292** |

The grid models were getting ~55× fewer weight updates than the flat models, at
a ~55× larger effective batch, on the same `lr=1e-3`. A "multidimensional
structure doesn't help" conclusion drawn from that comparison is partly a
statement about update budget, not architecture.

### The fix

Gradient accumulation (`Trainer.grad_accum`, `train.py --grad-accum`,
`sweep.py --step-match`, on by default). The flat arm accumulates
`ceil(G / flat_bs)` micro-batches per optimizer step, shortening the final
micro-batch so every group contains exactly G observations. This gives it the
same effective batch and the same step count as the combo arm without
materialising a 28k-sample batch or dropping observations.

The combo arm is the one that **cannot** move: one sample is one month with
every group present, and the axial/grid models need all groups simultaneously
to attend across the lattice. So matching happens by moving the flat arm down
to 96 steps, not the combo arm up.

`--flat-bs` is now a pure memory/speed knob (default 2,048); the effective
batch is pinned to G x `--combo-bs` regardless of what you set it to (raising
`--combo-bs` divides the combo arm's step count, so the flat arm accumulates
proportionally more). `--no-step-match` reproduces the old unmatched protocol;
pass `--flat-bs 512` with it to reproduce the pre-retrain runs exactly, since
the `--flat-bs` default moved 512 -> 2,048.

Resume (the default) requires a provenance completion record matching the
declared command, batch protocol, data, configuration, source tree, submodules,
training runtime, and checkpoint hash. Old unmatched runs therefore do not
count as finished and are re-run without `--force`.

### What this costs

Batch 28,292 at `lr=1e-3` is a large-batch regime the flat models were not
tuned for. **They will very likely score worse than the current numbers.**
That is the intended effect — both arms now sit in the same regime — but it
means the step-matched table is a *comparability* result, not a
best-achievable one.

Recommended reporting: two tables. Primary = step-matched (supports
architectural claims). Secondary = per-arm-tuned, i.e. the existing flat
numbers at batch 512, labelled as each family's unconstrained best. The
secondary table costs nothing because those runs already exist.

### Epoch-cap decision (now fixed)

The old 30-epoch runs hit the cap before early stopping. `base.yaml` and
`sweep.py` now default to 200 epochs so validation, rather than a short fixed
cap, decides when training ends.

---

## 2. Model/config fixes that change results

### F2 — Mamba-ND had no `LeftoverEncoder` *(critical, fairness)*

`MambaNDModel` folds un-scanned categorical axes into the batch exactly as
`AxialComboSA` does, but unlike every other combo model it never encoded them.
At `dims=2` it could not tell commodities apart; at `dims=3`, flows. Its
competitors (`gru`/`lstm`/`transformer`/`s4nd` axial) all could. Fixed by
wiring the same `LeftoverEncoder` + `combo_encoder` plumbing; no-op at
`dims=4`. **Mamba-ND 2d/3d rows are invalid and must be re-run.**

### F3 — Mamba-3 ran at `headdim=64` vs Mamba-2's `32` *(high)*

`config/models/mamba3.yaml` was still the placeholder and set
`hidden_size`/`num_layers`, which `Mamba3Model` does not read (it reads
`d_model`/`n_layers`). Every key in the file was ignored, including the
implicit head count: Mamba-3 ran 4 SSD heads against Mamba-2's 8 in a roster
that claims to be capacity-matched. The config now uses the right key names and
pins `headdim: 32`, so the two differ only in SSM formulation. Set it back to
64 to run Mamba-3 at its upstream default. **All Mamba-3 rows must be re-run.**

### F4 — nested `cross_attention:` / `film:` YAML blocks were never read *(high)*

`config/variants/*.yaml` nests `num_heads`, `d_k`, `d_v`, `dropout` under
`cross_attention:`, and `hidden_dim`/`num_layers` under `film:`. These reached
`create_model` as raw dicts and were swallowed by `**kwargs`; the code read
flat names (`cross_attn_num_heads`, `attn_dropout`, `film_hidden_dim`, …) that
no config ever supplied. `flatten_nested_model_cfg` now lifts them.

Numerically this is a no-op **today** — every nested value happens to equal the
code default it was silently falling back to — so it does not by itself
invalidate results. It does mean those files previously controlled nothing.
`d_v` is still not read by any module (the CA hops are single-head with
`d_v == hidden_dim`); it is passed through as `cross_attn_d_v` and ignored.

### F5 — asymmetric LR tuning *(medium; fixed with a symmetric grid)*

Transformer previously ran `lr=1e-4` and only Mamba-3 received an LR bracket,
while every other model ran `1e-3` untuned. That "tuned only if it broke"
protocol could favor or penalize particular models -- and Test 5, the first
attempt at a fix, searched only the FLAT arm, leaving the multidimensional arm
it is compared against with no search at all. Exp 0 replaces it: the identical
five-rate grid `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` for every cell in every
arm, selected on validation loss only (`scripts/select_lr.py`), injected into
every run by `sweep.py --lr-selection`. A selection that does not cover a cell
refuses the sweep rather than letting that cell fall back to an unsearched
config default.

### F6 — no persistence anchor in the leaderboard *(medium)*

`evaluate.py` scored only learned models, so nothing showed whether any of them
beat doing nothing on a near-random-walk monthly panel. It now scores
`seasonal_naive`, `random_walk` and `moving_average` on the same test window
and reports **MASE** (scaled by in-sample seasonal-naive MAE, computed on the
train window only). MASE < 1 beats seasonal persistence. Note the space: the
scale is measured on the normalized panel, so the per-series MinMax factor
cancels but `log1p` does not — this is a **log-space MASE**, and the paper must
say so. Raw-unit accuracy is the separately reported sMAPE. `--no-baselines`
disables. This is a *reporting* change: no retraining needed, just re-run
`evaluate.py`.

---

## 3. Latent fixes (no current results affected)

| ID | Fix | Why it did not bite |
|---|---|---|
| F7 | `evaluate.py` / `model_cost.py` now build from the composed config (`compose_config` + `model_kwargs_from_config`) instead of a hardcoded `hidden_size=128, num_layers=4` | Both built GPT at its code default `d_model=256` while it trains at 128 → `load_state_dict` raised → the run was silently dropped by the `except` in `main()`. GPT is not in the census roster, so nothing was lost yet. Would have bitten the moment GPT was added. |
| F8 | `--num-layers` now writes both `num_layers` and `n_layers`; `normalize_model_kwargs` exposes width/depth under both spellings | The flag was a no-op for `s4`/`mamba*`/`mamba_nd`/`gpt`. The sweep never passes it. |
| F9 | `MODEL_CONFIG_ALIAS` maps `mamba2 → mamba.yaml`; added `config/models/s4nd.yaml`; a model with no resolvable config is now a hard error | `--model mamba2` (every flat SSM run) and `--model s4nd` loaded no model YAML and ran on code defaults. Those defaults happened to equal the intended values, so numbers are unaffected — but nothing enforced it. |
| F10 | `input_len` (which lives under `data:`) is threaded into model kwargs | `S4NDModel` sizes its Time-axis DPLR kernels from `input_len` and defaulted to 36. Everything runs at 36, so no run was wrong; `--input-len 24` would have built length-36 kernels. |
| F11 | `Trainer` reads `scheduler_patience` / `scheduler_factor` from config; checkpoint selection honours `checkpointing.mode` | Hardcoded values coincided with the config; `mode` is always `min`. |
| F12 | Baselines take `value_idx` / `weight_idx` instead of hardcoding channels 0/1 | The Census 9-channel panel puts `agg_weight` at channel 5, not 1. Baselines had never been run on Census; doing so would have scored against a transport-mode channel. |
| F13 | Tree models seed `random_state` from the run's `--seed` | `subsample`/`colsample` make them genuinely stochastic, so a pinned 42 meant every "multi-seed" tree run repeated one draw. Trees are not in the census roster. |

---

## 4. What must be recomputed

### Retrain (checkpoints invalid or protocol-inconsistent)

| Runs | Reason | Cost |
|---|---|---|
| **Entire declared matrix** | train-only cohort rebuild, corrected normalization artifact, observation-weighted validation selection, step matching, 200-epoch cap, plus F2/F3 where applicable | all planned runs |

There is no supported "minimum re-run" subset. The sweep manifest is the
authority for completion, and strict evaluation will fail while any planned
run or required seed is missing.

The per-finding slices below are therefore diagnostic only — useful for
answering "which findings touched this run?" and for smoke-testing a single
family, NOT for assembling a partial official leaderboard. They are available
as `RETRAIN=f1|f2|f3|mandatory` (regexes in `sweep.py::RETRAIN_SETS`):

| Slice | Runs | What the finding invalidated |
|---|---|---|
| `f1` | 36 | flat 1-D runs: step-matched against the combo arm |
| `f2` | 6 | Mamba-ND 2d/3d: leftover encoder added, input width differs (no-op at `dims=4`) |
| `f3` | 63 | every Mamba-3 run (flat, two-mixer hybrid, fixed aggregate, and annual rolling aggregate): corrected head count reshapes the weights |
| `mandatory` | 99 | the union of the three |

Two points that stay true regardless of the full-matrix decision:

* **Test 1 (aggregate) is not affected by F1.** The aggregate arm keeps
  `--batch-size 32` with no accumulation, which is correct: every model in
  Test 1 runs the flat path over a single national series, so there is no
  flat-vs-grid asymmetry inside that test to correct. Its Mamba-3 runs are
  still invalidated by F3, and at ~3 steps/epoch the whole test was badly
  undertrained under the old 30-epoch cap.
* **Mamba-3's combo hybrids are model-invalidated, not protocol-invalidated.**
  F3 changes the architecture, so no "combo runs survive" argument ever
  covered them.

### After retraining

- Run `scripts/evaluate.py`; it adds the baselines, MASE, raw-unit sMAPE,
  strict seed validation, and subgroup fairness outputs.
- Run `scripts/model_cost.py` against the corrected model configs.

### Regenerate from the new metrics

Every HPEC table, significance test, and figure must be regenerated from the
new strict evaluation outputs. Predecessor-only workbook builders remain in
this repository and must not be used for the Census benchmark.

### Do **not** reuse

Any old HPEC checkpoint, table, cross-arm comparison, subgroup claim, or
cost-accuracy Pareto.

---

## 5. Running it

`scripts/hpec_pipeline.sh` drives the whole thing — train -> eval -> cost —
with resume, selective redo and a graceful stop. (`scripts/sweep.py`
still works standalone for training only; `scripts/run_all_experiments.sh`
drives the OLD WCTR experiments, not this lattice.)

```bash
# what state is every run in?  (done / stale / partial / failed / todo)
bash scripts/hpec_pipeline.sh --status

# confirm the matrix and the step-matched batch line before spending GPU time
TESTS=1,1.1,2,3,4,6 bash scripts/hpec_pipeline.sh --dry-run

# THE RETRAIN: full declared matrix at the 200-epoch cap, fresh manifest
GPUS=0,1,2,3 STAGES=exp0 bash scripts/hpec_pipeline.sh   # select the rates first
GPUS=0,1,2,3 TESTS=1,1.1,2,3,4,6 EPOCHS=200 RESET_MANIFEST=1 \
  bash scripts/hpec_pipeline.sh

# diagnostic only: the 99 runs touched by F1/F2/F3 (strict evaluation will
# reject this as an incomplete official matrix)
GPUS=0,1,2,3 RETRAIN=mandatory STAGES=train bash scripts/hpec_pipeline.sh

# or the full matrix with the raised epoch cap (early stopping then decides)
GPUS=0,1,2,3 TESTS=1,1.1,2,3,4,6 EPOCHS=200 bash scripts/hpec_pipeline.sh

# re-evaluate / rebuild only, over checkpoints that already exist
STAGES=eval,cost bash scripts/hpec_pipeline.sh

# halt cleanly mid-sweep: in-flight runs finish, nothing new launches
touch outputs/sweep/sessions/<session-id>/STOP  # delete it and re-run to continue
```

Named retrain slices (regex over run names, kept in `sweep.py::RETRAIN_SETS` so
they cannot drift from this document): `RETRAIN=f1` (36 flat cells), `f2` (6
Mamba-ND), `f3` (63 Mamba-3), `mandatory` (99). Anything else is reachable with
`ONLY='<regex>'`.

`RESET_MANIFEST=1` creates a new isolated directory under
`outputs/sweep/sessions/` and records it in `outputs/sweep/current_session`.
If training is stopped, rerunning without `RESET_MANIFEST` resumes that same
session; older checkpoints and statistics are never merged into it.
`sweep.py` skips a run only when its checkpoint and provenance completion
record validate against the current declaration. Requested pipeline stages
always execute, so changed data, seeds, source, or batch settings cannot be
hidden by stale timestamp markers. A failed or gracefully stopped training
stage aborts the pipeline rather than building tables from a half-finished sweep.

### The local FA arm was dropped

`fa_local_*` (formerly `cafa_*`) is a local reimplementation of the FA operator.
It differs from the authors' `fa_*` in at least four ways simultaneously, so no
comparison between them is attributable to a single cause. It left the declared
matrix (621 -> 477 runs) so that every FA claim rests on the authors' own
components. The module and its regression tests remain; `--mechs asa,fa,fa_local`
reruns it. If a kernel-nonlinearity claim is wanted later, the faithful form is
`LowRankKernel(softmax=True)` — the authors' own switch — not this arm.

### FA head geometry is pinned, with an opt-in sensitivity arm

`fa_heads 4 / fa_dim_head 32 / fa_kernel_multiplier 2`, fixed in
`config/variants/fa_*.yaml` before any run. The paper's claim is "FA geometry is
scaled for this benchmark's 128-dimensional residual width: four 32-dimensional
value heads give a 128-dimensional value space, while `kernel_multiplier=2`
gives 64-dimensional Q/K features per head" — a benchmark-specific scaling
choice, not head-geometry fidelity to the authors and not a match to
transformer/gpt (8 heads x 16).

For the record, the authors' processor is **768** wide, not 384: `latent_dim:
768` is the residual/channel width `FactFormerS2` hands to `FABlockS2`, while
the 384 is `processor.latent_dim`, the `PoolingReducer` bottleneck the axial
kernels live in. Their 16 x 64 = 1024 of value width is therefore ~1.33x the
residual width, not ~2.7x. See the table in `config/variants/fa_2d.yaml`.

`--fa-dim-head-ablation` adds 9 runs ({16,32,64} x 3 seeds on gru+fa_2d+embeddings)
to show 32 is not arbitrary. It is diagnostic: the main table stays 32, and the
ablation must not be used to pick a winner on test data.

### Optional arm: axial cross-attention

`--mechs asa,fa,aca` adds `aca_{2,3,4}d` (`src/models/aca.py`) — the
only arm in which queries and keys come from different sources; everything else
is self-attention. It costs **+144 runs** (477 -> 621) and is therefore NOT in
the declared matrix. Decide before launching: adding it later re-declares the
manifest, and strict evaluation requires the full declared matrix to complete.

### Freeze the source tree for the duration of a sweep

The provenance fingerprint covers the declared command, the data, the configs,
**every `.py`/`.yaml` under `src/`, `scripts/`, `config/` and `external/`**, the
submodule state and the training runtime. That is deliberate — it is what makes
"this checkpoint was produced by this code" checkable — but it has an operational
consequence worth stating plainly:

> Editing **any** source file mid-sweep invalidates every finished run. `--resume`
> will retrain them, and `evaluate.py` will refuse the manifest.

A docstring fix in `evaluate.py`, a new helper script under `scripts/`, or
initializing a submodule is enough. So: tag the commit you intend to run, do not
touch the tree until evaluation has written `results.csv`, and land code changes
between sweeps rather than during one. `sweep.py --status` lists every cell whose
provenance no longer matches (`stale`), and `--resume` now prints a warning naming
them instead of silently re-training.

Each run's log now carries an auditable step budget, so the protocol can be
verified from the logs rather than re-derived:

```
Batch budget: micro-batch 2048 x grad_accum 14 = effective batch 28292 observations |
  1344 micro-batches/epoch -> 96 optimizer steps/epoch | max 19200 steps over 200 epochs
...
Realized optimizer steps: 5568 (96/epoch x 58 epochs) at effective batch 28292
```

Sanity check before committing GPU time: run one flat and one combo config for
1 epoch and confirm the two `optimizer steps/epoch` numbers match.

---

## 6. Verification status

The original-review regression suite passes locally. Mamba execution cannot be
tested on this macOS host because Triton/CUDA are intentionally unavailable.
Before the full sweep, on the GPU box:

1. `python scripts/check_environment.py --require-mamba`.
2. `TESTS=3 EPOCHS=1 STAGES=train ONLY='^gru_' bash scripts/hpec_pipeline.sh`
   — flat + combo smoke through the real driver.
3. Confirm matched `optimizer steps/epoch` in both logs.
4. `python -m pytest tests/test_review_regressions.py tests/test_scan_schedule.py tests/test_fa_local_smoke.py tests/test_fa_smoke.py`.
5. One `mamba_nd asa_2d` run — the F2 change alters its input width
   (`features_per_group + leftover.out_dim`), so old `mamba_nd` checkpoints will
   not load; that is expected.
