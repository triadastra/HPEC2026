# Experiment catalog

This is the authoritative catalog for the declared Census HPEC training matrix.
An **experiment** is one independently trained cell: model × architecture or
variant × encoder × seed, plus learning rate or rolling fold when applicable.
Analytic baselines are evaluations, not training experiments.

## Matrix summary

| Area | Scientific question | Training experiments |
|---|---|---:|
| Test 1 — fixed aggregate | Can each temporal model forecast one national aggregate series on the fixed 2024–2025 holdout? | 18 |
| Test 1.1 — annual rolling aggregate | Does aggregate performance persist across six annual forecast origins rather than one fixed split? | 108 |
| Test 2 — one-hot N-D | Does promoted multidimensional structure help when leftover categoricals are one-hot encoded? | 72 |
| Test 3 — embedding N-D | Does promoted multidimensional structure help when leftover categoricals use learned embeddings? | 81 |
| Test 4 — identity-aware axial | Was the structured-model comparison confounded by promoted axes lacking explicit identity? | 108 |
| Test 6 — structured SSM completion | Do Mamba hybrids and genuine S4ND change the structured-model conclusion? | 90 |
| **Total to train** | | **477** |

Two mixers are declared (`asa`, `fa`), not three. `fa_local`, the local
reimplementation of the FA operator, left the roster so that every FA claim
rests on the authors' own components; `aca` (axial CROSS-attention) is opt-in.
Either one adds 144 runs. Test 5 is retired -- Exp 0 replaces it (below).
Neither diagnostic mixer is a third roster arm. `fa_local` is declared as
**Exp 7** and `fa_sm` — the same `fa` operator on the authors' softmax switch —
as **Exp 8** (both below), because each exists only to be paired against a
Test 3 cell one variable at a time, not to join the leaderboard.

**Exp 0 — learning-rate selection (240 runs).** Not part of the 477: it runs
first, in its own directory, and gates the rest. The same five rates
`{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` — log-spaced by √10 across two decades — for
every cell of every arm.

**Selecting arms** — each yields the rate its runs train at:

| Arm | Selects for | Cells | Runs |
|---|---|---:|---:|
| `flat` | the per-series 1-D baselines | 6 models | 30 |
| `nd` | every multidimensional backbone, probed on `asa_3d`/`grid_3d` | 7 models | 35 |
| `agg` | Test 1, the fixed aggregate split | 6 models | 30 |
| `roll` | Test 1.1, probed on the **shortest** fold | 6 models | 30 |
| `mech` | every **fa** cell (Tests 2/3/4/6), probed on `fa_3d` | 5 models | 25 |
| `mechlocal` | every **fa_local** cell (Exp 7), probed on `fa_local_3d` | 3 models | 15 |
| `mechsm` | every **fa_sm** cell (Exp 8), probed on `fa_sm_3d` | 3 models | 15 |

`mech` began as a two-model check arm and was promoted to a selecting arm once
that version showed the asa→fa inheritance reverses a measured cell; the 144
declared fa runs train at the mech rate, and a selection file without the arm
is refused rather than silently downgraded to the nd rate.

**Check arms** — reported, never selected from. Each corresponds to one axis
along which a probe's command line differs from the runs it selects for. Every
such difference is an assumption that the rate still transfers, and these are
the assumptions made explicit:

| Arm | Checks the rate survives | Cells | Runs |
|---|---|---:|---:|
| `dim2`/`dim4` | 2-D and 4-D, not just the 3-D probe | 2 models | 20 |
| `rollend` | the **longest** fold, not just the shortest | 2 models | 10 |
| `encflat` | the one-hot encoder (Test 2), not just embeddings | 2 models | 10 |
| `encnd` | the one-hot combo encoder (Test 2) | 2 models | 10 |
| `axid` | `--axis-identity` (Test 4), which changes the model | 2 models | 10 |

Each check arm probes one axis in isolation, on two backbones. Interactions
between axes — the fa mixer under `--axis-identity`, a hybrid under the
one-hot encoder — are not probed; the paper states this single-axis transfer
assumption explicitly.

One seed, 20-epoch budget, selected on validation loss only. `roll` is separate
from `agg` because fold 2020 trains on roughly half the history the fixed split
gets; a rate chosen on the full split is not automatically right for it, and
Exp 0 does not make that transfer silently.

`tests/test_exp0_protocol_integrity.py` enumerates every flag on which a probe
differs from the runs it selects for, and fails on any difference no check arm
covers — so this table cannot quietly fall out of step with the matrix. See
`scripts/select_lr.py`.

**Exp 7 — the fa_local replication (27 runs).** Not part of the 477: it runs
alone, in its own directory, and it is the arm that has to run FIRST.

The submitted draft's CaFA advantage — CaFA winning 12/12 on GRU and 12/12 on
LSTM against the axial arm — was produced by `fa_local_*`, our own build of the
FA operator from the paper (the workbook spells it `cafa_*`). The rerun never
ran that module. It runs `fa_*`, the authors' released components, which lose
0/12 and 1/12 on the same hosts. So the reversal the paper now reports
confounds **two** changes at once — the operator changed *and* the protocol
changed — and nothing published separates them.

Running `fa_local` under the current cohort, normalisation, step matching and
epoch cap separates them directly, and the two outcomes say different things:

* `fa_local` still beats `asa` → the draft's finding survives. What changed is
  that the authors' operator behaves differently at this scale, and the paper
  says exactly that.
* `fa_local` also loses → the advantage was the old cohort/protocol, not the
  operator, and the four-way difference between the two builds is moot.

*What "replication" means here, precisely.* Exp 7 is not a bit-exact replay of
the code that produced the workbook, and the paper should not claim it is. Two
things changed in `fa_local` after the draft's runs (which postdate `78ec003`,
since the workbook carries an `identity` column):

* `out = out * self.vmask6` was added between axial contractions (`2c312c4`) —
  invalid cells were carrying scratch into the next axis while the denominator
  still counted only originally valid keys.
* `torch.nan_to_num(out)` was removed — a no-op given `clamp_min`, except that
  it laundered real divergence into finite numbers.

Measured, not assumed: `fa_local_2d` is **bit-identical** to the workbook-era
forward (one contraction, nothing to carry). `3d`/`4d` differ, and the size of
that difference is a function of lattice density — 0.56% relative output change
at the old lattice's 99.4%, 3.2% at 85%, 3.9% at 80%. At the density the draft
ran on, the change is far inside 3-seed variance, so **the workbook's
`cafa_3d`/`cafa_4d` numbers are not called into question by it**. Exp 7
therefore reruns the same operator with one correctness fix, and the fix
matters more for the current cohort than for the old one.

The `nan_to_num` removal is the one thing that cannot be checked after the
fact: a draft-era run that diverged would have completed with finite garbage,
where the same run today raises `NonFiniteLossError`. The old run logs are
gone, so this is recorded as an unresolved caveat rather than a cleared one.

**Exp 8 — FA kernel nonlinearity (27 runs).** Only meaningful once Exp 7 has
shown the two operators disagree under one protocol. `fa_local` and `fa` differ
four ways at once (softmax vs LeakyReLU gating, gamma-MLP vs `PoolingReducer`,
single-head vs `LowRankKernel`, and a different sparse renormalisation), so
neither the RNN loss nor the Transformer win is attributable to any one of
them. `fa_sm_{2,3,4}d` moves exactly one: the authors' own
`LowRankKernel(softmax=True)` switch, with `PoolingReducer`, head geometry, the
channel mixer and Q/K RMS norm pinned to the `fa_*` values. `fa_sm` and `fa`
have identical parameter counts at every lattice size and byte-identical
initial weights under the same seed.

It is not a config-only flip: `fa.py` applies the LeakyReLU gate outside
`LowRankKernel` and then divides by a uniform categorical quadrature count,
and upstream `FABlockS2` retunes the kernel temperature when softmax is on. The
flag therefore drops the gate, swaps the divisor for a valid-mass renormaliser,
and takes the scaling that belongs to the mode — otherwise the arm would report
a ~1/line_count rescale, or a kernel 8x sharper than the authors intend, rather
than the nonlinearity.

Both arms mirror the shape of their comparator exactly:

| | cells | runs |
|---|---:|---:|
| gru / lstm / transformer × 2d/3d/4d × embeddings × 3 seeds | 9 | 27 |

**Each trains at its own Exp 0 arm** — `mechlocal` for Exp 7, `mechsm` for
Exp 8 — never at `mech`, which was probed on the authors' LeakyReLU `fa_3d`.
The arms are far enough apart for that to matter: `nd` vs `mech` is 3e-4 vs
1e-2 on gru (33x) and reverses direction on transformer (1e-3 vs 3e-4). A
prefix test that routed every `fa_*` variant to `mech` would have handed both
diagnostic mixers someone else's rate, which is the asymmetry F5 removed.
`tests/test_diagnostic_mixer_arms.py` pins the routing and every way either arm could
stop being single-variable.

**The two mixers are NOT capacity-matched, and the paper has to say so.** The
backbones are (hidden 128, 4 layers, uniform protocol), but `asa` and `fa` are
not, and nothing in this repository said it until now. Measured on the real
lattice, embeddings encoder:

| | mixer params | model params | FLOPs/forecast |
|---|---|---|---|
| gru 2d | 50,176 → 558,464 (**11.1x**) | 596,654 → 1,104,942 (1.85x) | 6.00x |
| gru 3d | 99,584 → 689,920 (6.9x) | 505,350 → 1,095,686 (2.17x) | 1.45x |
| gru 4d | 148,992 → 821,376 (5.5x) | 554,242 → 1,226,626 (2.21x) | 1.34x |

Measured at the mixer itself, with the backbone excluded, the four arms line
up like this (width 128, real lattice):

| dims | `asa` | `fa_local` | `fa` (authors) | `fa_sm` | local/asa | fa/asa |
|---|---:|---:|---:|---:|---:|---:|
| 2d | 50,176 | 83,200 | 558,464 | 558,464 | 1.66x | 11.13x |
| 3d | 99,584 | 148,992 | 689,920 | 689,920 | 1.50x | 6.93x |
| 4d | 148,992 | 214,784 | 821,376 | 821,376 | 1.44x | 5.51x |

That table carries the sharpest version of the problem. **The draft's mixer
comparison was near capacity-matched and the rerun's is not.** `cafa` vs
`cross_attention` — what the workbook actually ran — is `fa_local` vs `asa`,
1.44x–1.66x. `fa` vs `asa` is 5.5x–11.1x. So the reversal the paper reports
does not only swap the operator; it swaps a roughly matched mixer for one
carrying an order of magnitude more parameters, and the sign of the result
changes with it. That is a third confound stacked on the other two, and another
reason Exp 7 runs first: `fa_local` restores the operator AND the near-match at
once, leaving the protocol as the only thing that moved.

Model-level ratios run 1.51x–2.21x across all nine cells. 77% of the gap is one
module: the authors' `channel_mixer = MLP([dim, dim*6, heads*dim_head+dim*2])`,
393,216 parameters at width 128, which the axial baseline has no counterpart
for. It is inherent to their released block, not a tuning choice, so it cannot
be equalised without deviating from the operator being tested — which is
exactly why it must be reported rather than quietly fixed.

This cuts asymmetrically, and the two Test 3 findings do not inherit the same
confidence:

* **RNNs** — fa loses 0/12 (GRU) and 1/12 (LSTM) *while carrying 5–11x the
  mixer*. Capacity is working against the conclusion, so the conclusion is
  robust; if anything the margin understates it.
* **Transformer** — fa wins 10/12 *while carrying the same advantage*. "The
  factorization helps Transformers" is not separable here from "393k more
  parameters help Transformers". State it as confounded, or add a
  capacity-matched axial arm before claiming the mechanism.

The FLOPs column is the weaker of the two: per §4 of the cost accounting, the
fused recurrent kernels and the eval-mode Transformer encoder count as zero, so
these ratios are mixer-dominated. The parameter column is the one to quote.

Exp 7 is unaffected by any of this: `fa_sm` and `fa` have **identical**
parameter counts at every lattice size (softmax carries no weights), identical
buffers, and byte-identical initial weights under the same seed.

The three analytic anchors—seasonal naive, random walk/persistence, and moving
average—are scored once per task, so no leaderboard panel is without a floor:
on the 28k-series panel, on the fixed aggregate split, and on each Test-1.1
fold. They add zero training experiments. A row's `panel` column says which of
the three it belongs to; the errors are not comparable across them.

## Shared protocol

- Models: GRU, LSTM, Transformer, S4, Mamba-2, and Mamba-3, with Mamba-ND and
  S4ND included only in their applicable structured areas.
- Seeds: 947, 732, and 619.
- Targets: aggregate value and aggregate weight.
- Input: 36 months plus 12 explicit aggregate lags; the first usable target is
  therefore 2014-01 in the 2010-01…2025-12 panel.
- Every rolling run starts from a fresh model initialization. No checkpoint is
  warm-started from an earlier fold. Its completion record is bound to a unique
  `model_session.json` identity created by that training process.
- Split intervals are half-open: `[2020-01, 2021-01)` is exactly 12 months.
- Aggregate targets are summed from the full raw universe before the
  multidimensional benchmark cohort is selected.

## Test 1 — fixed aggregate

Area: aggregate sanity check and temporal-model baseline.

Six models × three seeds = **18 experiments**. All raw source series are summed
before cohort selection into one national `(value, weight)` series. The canonical boundaries are
train `[0,144)`, validation `[144,168)`, and test `[168,192)`, corresponding to
training through 2021, validation in 2022–2023, and testing in 2024–2025.

Run-name form: `<model>_aggregate_s<seed>`.

## Test 1.1 — annual rolling-origin aggregate

Area: temporal robustness and structural-break sensitivity.

One year defines the validation and test windows; training history expands.
Each fold fits aggregate log1p/MinMax normalization on that fold's training
months only, selects the checkpoint on the immediately preceding year, and
tests on the following year. Forecasts remain one-step walk-forward: prediction
for month `t` may use the observed history through `t-1`, including earlier
months in the same test year, but can never use month `t` or any later month.

| Test year | Training targets | Validation | Test indices | Experiments |
|---:|---|---|---|---:|
| 2020 | 2014–2018 | 2019 | `[120,132)` | 18 |
| 2021 | 2014–2019 | 2020 | `[132,144)` | 18 |
| 2022 | 2014–2020 | 2021 | `[144,156)` | 18 |
| 2023 | 2014–2021 | 2022 | `[156,168)` | 18 |
| 2024 | 2014–2022 | 2023 | `[168,180)` | 18 |
| 2025 | 2014–2023 | 2024 | `[180,192)` | 18 |
| **Total** | | | 72 out-of-sample months | **108** |

Each row is six models × three seeds. Run-name form:
`<model>_aggregate_roll_y<year>_s<seed>`.

Evaluation must produce one row per trained model/fold/seed in
`rolling_test_scores.csv`; official evaluation fails if any declared rolling
run lacks a test score. Each row records its train/validation/test boundaries
and the unique model-session ID whose checkpoint produced it. Evaluation also
writes seed-averaged annual summaries to
`rolling_annual.csv` and equal-window pooled summaries to `rolling_pooled.csv`.
Fold-normalized MSE is reported, but MASE and raw-unit sMAPE are the primary
cross-year comparisons.

The builder stores this national series before density filtering, state
ranking, or commodity pruning. The rolling loader then truncates its in-memory
panel at each fold's test end, so later months cannot enter features or targets.
Artifacts lacking this pre-cohort aggregate contract are rejected and must be
rebuilt before aggregate training.

## Test 2 — one-hot N-D

Area: categorical representation and dimensional promotion.

- Flat baselines: six models × three seeds = 18.
- Structured cells: GRU/LSTM/Transformer × {`asa`, `fa`} ×
  2-D/3-D/4-D × three seeds = 54.
- Total: **72 experiments**.

Leftover categoricals are one-hot encoded. At 4-D there are no leftover
categoricals, so one-hot and embedding architectures coincide.

## Test 3 — embedding N-D

Area: learned categorical representation and dimensional promotion.

- Flat baselines: six models × three seeds = 18.
- Structured GRU/LSTM/Transformer cells: 54.
- Mamba-ND grid scans: three dimensions × three seeds = 9.
- Total: **81 experiments**.

## Test 4 — identity-aware axial

Area: positional/identity confounding in promoted axes.

GRU/LSTM/Transformer × {`asa`, `fa`} × 2-D/3-D/4-D × two
leftover encoders × three seeds = **108 experiments**. Every promoted axis gets
learned identity embeddings. Mamba-ND is excluded because its ordered scan is
already position-aware.

## Test 5 — retired

Area: optimization sensitivity and fair hyperparameter selection.

Superseded by Exp 0 and no longer runnable (`--tests 5` is refused with a
pointer to the replacement). Test 5 searched four rates across one decade for
the FLAT arm only -- the multidimensional arm it was compared against got no
search at all, which is the asymmetry it was meant to address. Exp 0 searches
five rates across two decades in every selecting arm, so the old grid is a
strict subset of a symmetric one.

The Mamba-3 stability finding that came out of Test 5 stands on the runs
already collected and is reported from those. It is not re-trained. See
`future_work.md`.

## Test 6 — structured SSM completion

Area: structured state-space model coverage.

- Mamba-2 and Mamba-3 hybrids: two models × two mixers × three
  dimensions × two encoders × three seeds = 72.
- Genuine S4ND: three dimensions × two encoders × three seeds = 18.
- Total: **90 experiments**.

Like Tests 2 and 3, the hybrid cells run identity-blind along their promoted
axes; the identity confound is isolated on the attention hosts by Test 4. An
opt-in `--hybrid-identity` arm (+72 runs) extends the same A/B to the
mamba2/mamba3 hybrids — see `future_work.md`. S4ND and Mamba-ND need no such
arm: an LTI kernel or ordered scan along an axis is already position-aware.

## Launch and audit

Declare the complete matrix with:

```bash
# Exp 0 first: it selects the learning rate every run below trains at.
bash scripts/hpec_pipeline.sh   # STAGES=exp0
TESTS=1,1.1,2,3,4,6 bash scripts/hpec_pipeline.sh --dry-run
```

Start a fresh official session with:

```bash
GPUS=0,1,2,3 TESTS=1,1.1,2,3,4,6 EPOCHS=200 RESET_MANIFEST=1 \
  bash scripts/hpec_pipeline.sh
```

The saved `manifest.json` is the execution authority. A checkpoint counts as
complete only when its command, data/config/source fingerprint, completion
record, and checkpoint hash match that manifest.
