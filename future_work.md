# Future work

Work that is built, or was built and has been retired, but is deliberately
**not part of the declared HPEC benchmark**. Nothing here is reported as a
benchmark result in the current paper. Each entry records what it is, why it
was set aside, and exactly how to revive it.

---

## Test 5 — flat-model learning-rate bracket (retired 2026-08-26)

### What it was

A symmetric four-rate learning-rate grid `{1e-4, 3e-4, 5e-4, 1e-3}` over every
flat model, both encoders, all three seeds:

```
6 models x 4 rates x 2 encoders x 3 seeds = 144 runs
```

It served two purposes at once, which is why it was replaced:

1. **Learning-rate selection** for the flat arm.
2. **A stability study** — the endpoint behaviour of models that misbehave at
   the shared default rate.

### Why it was retired

Purpose (1) moved to **Exp 0**, which covers *every* arm (flat, N-D, aggregate)
rather than only the flat one. Test 5 selected rates for the flat arm while the
**315 multidimensional runs** in the declared matrix inherited unsearched config
defaults — the exact asymmetry Exp 0 exists to remove, and the one that matters
most here because the paper's headline is that the multidimensional arm does
*not* earn its compute. A negative result obtained under a smaller search
budget for the arm being argued against is not a result anyone has to accept.
See `catalog.md` for the full argument.

Purpose (2) is a genuine finding, but it is **not a headline result of this
paper**:

- The instability is specific to Mamba-3, a very new architecture.
- The paper's thesis is that multidimensional structure does not earn its
  compute for demand forecasting. Mamba-3's optimiser behaviour is orthogonal
  to that claim.
- IEEE HPEC full papers are capped at 6 pages excluding references; the
  stability material does not compete with the headline for that space.

Keeping 144 runs to support material that will not be reported was not a good
use of the budget. Exp 0 costs 240 truncated runs and replaces the part that
mattered.

**The finding itself is retained and reported from the runs already
collected.** It is not re-trained — see "Reusing the old observations".

### The finding it was built to support

From the Test 5 comment in `scripts/sweep.py`, preserved verbatim before
removal:

> Test 5: symmetric flat-model LR arm. EVERY run (Tests 1-4) already trains
> under grad-clip 1.0 (base.yaml training.grad_clip), yet mamba3's
> near-unit-circle rotational state diverges on the per-series task at the
> shared lr=1e-3 (2/3 seeds, finite-but-bad loss — clipped divergence) and
> underfits at the 1e-4 refit (untagged runs). Same remedy the protocol already
> grants the transformer (config/models/transformer.yaml: 1e-3 NaNs -> per-model
> lr 1e-4).

Note the phrase **"untagged runs"**: those observations predate the current
manifest/fingerprint scheme. See "Reusing the old observations" below.

Note also "finite-but-bad loss — clipped divergence". Grad-clip converts some
divergence into a finite but useless loss rather than a NaN. The divergence
detector added in `src/training/trainer.py` catches only the non-finite kind;
the clipped kind still reaches the leaderboard, honestly rather than scrubbed.
That distinction is the substance of the stability finding.

### The code

**Removed**, not merely excluded. `--tests 5` is refused with an explanation
pointing at Exp 0 (`scripts/sweep.py::RETIRED_TESTS`), because an unknown test
id that silently produced zero runs would look like a completed sweep while
quietly shrinking the seed matrix.

The emitter, the `--lr-bracket-models` flag and the `lr_bracket` parameter were
deleted in `f3a5011`; the last commit that contains them is `032c75f`:

```bash
git show 032c75f:scripts/sweep.py    # the Test 5 emitter, verbatim
```

Supporting machinery that stays live on purpose:

- `scripts/evaluate.py::parse_run` still parses `lrNeM` tags into their own
  leaderboard variant, so **already-collected Test 5 checkpoints remain
  scoreable** without reviving anything.
- `scripts/evaluate.py` now also emits a `val_loss` column, so the selection
  signal is visible next to the test metric it must not be chosen from.

### How to revive it

The honest revival is not to restore the old code — it is to run Exp 0 with a
wider grid and read the divergence report, which is strictly more informative:

```bash
bash scripts/hpec_pipeline.sh   # STAGES=exp0
# select_lr.py reports, per (arm, model), every rate that was tried and did
# not train, under "diverged (tried, did not train)".
```

Exp 0 probes `1e-2` precisely to find where training breaks down, and
`scripts/train.py` records each divergence in `run_diverged.json` rather than
crashing. A rate that diverges is a measurement, not a failed run.

If the *original* arm is genuinely wanted for a separate paper, restore the
emitter from `032c75f` into a scratch branch and give it its own `--runs-dir`.
Run fingerprints include the full command, so a Test-5 run can never be
mistaken for a main-matrix run.

### If it is revived for a separate paper

Two things to fix first, both found during the 2026-08-25 verification:

1. **Adopt Exp 0's grid.** `5e-4` is the dead trial: the old grid's ratios are
   ×3, ×1.67, ×2, and `5e-4` sits in the tightest gap. Exp 0 runs
   `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` — five rates, log-spaced by exactly √10,
   spanning two decades. The old grid covered only one decade and its ceiling
   was hit in all three arms during the 2026-08-25 probe, with validation loss
   still improving monotonically toward it. Extension goes up rather than down
   because `reduce_on_plateau` can only lower a rate: a too-low start is never
   recovered, a too-high one partly self-heals.

2. **Selection must be on validation.** `scripts/select_lr.py` reads only
   per-epoch `val_loss` from `logs/metrics.json` and never opens a results
   table. Do not pick the best row of the test table.

### Reusing the old observations

The pre-existing Mamba-3 divergence data **can** be cited as a qualitative
observation, but not as a benchmark result:

- It will not pass the manifest fingerprint gate — that gate covers command,
  batch protocol, data, configuration, source tree, submodules, and runtime,
  and those runs predate the step-matched protocol.
- `README.md` already states that the protocol changes "invalidate the earlier
  draft claims".

A crisp era test: **if the run's argv has no `--effective-batch-size`, it
predates the step-matched protocol.** The divergence evidence itself lives in
each run's `logs/metrics.json` (per-epoch `val_loss`), not in the checkpoint,
so it survives even where the checkpoint is unusable.

If cited, label it explicitly, e.g.:

> Under an earlier protocol (pre step-matching) we observed Mamba-3 diverging
> at the shared 1e-3 in 2 of 3 seeds. Those runs are not comparable to the
> reported results and are included only to motivate Exp 0.

---

## Off-roster model arms

All three are implemented, tested, and runnable; none is in the declared
matrix.

- **`--hybrid-identity` — identity-aware hybrids** (+72 runs). Test 4 isolates
  the permutation-equivariance confound on the attention hosts
  (gru/lstm/transformer) only; the Test 6 mamba2/mamba3 hybrids use the same
  identity-blind asa/fa mixers and have no identity-aware control. If Test 4
  shows identity matters, the hybrid conclusions inherit the confound until
  this arm runs. It is the exact hybrid grid × `--axis-identity`
  (two models × two mixers × three dims × two encoders × three seeds), rates
  drawn from the same nd/mech arms as the identity-blind twins, run names
  `<model>_<mixer>_<d>d_id_<enc>_s<seed>` (evaluate.py already parses them,
  and the cost panel's `COMBO_ID` measures the cells). Revive with
  `--tests ...,6 --hybrid-identity` (pipeline: `HYBRID_IDENTITY=1`). S4ND and
  Mamba-ND stay excluded: an LTI kernel or ordered scan along an axis is
  already position-aware.

- **`fa_local` — local reimplementation of the FA operator** (+144 runs).
  It differs from the authors' `fa_*` in at least four ways simultaneously —
  softmax vs LeakyReLU gating, gamma-MLP vs `PoolingReducer`, single-head vs
  `LowRankKernel`, and a different sparse renormalization — so no `fa` vs
  `fa_local` comparison is attributable to any single cause. It left the roster
  so that every FA claim rests on the authors' own components. If a
  kernel-nonlinearity claim is wanted later, the faithful form is
  `LowRankKernel(softmax=True)` — the authors' own switch — not this arm.
  Revive with `--mechs asa,fa,fa_local`.

- **`aca` — axial cross-attention** (+144 runs). The only arm in which queries
  and keys come from different sources; everything else on the roster is
  self-attention. Revive with `--mechs asa,fa,aca`. Decide before launching:
  adding it later re-declares the manifest, and strict evaluation requires the
  full declared matrix to complete.

---

## Deferred verification items

Found during the 2026-08-25 pipeline verification; none block the paper.

- **`seed`-mode equivalence margin is fragile at n=3.**
  `scripts/significance_tests.py --delta-mode seed` scales the margin by the
  per-seed SD, which is itself estimated on 2 degrees of freedom. Defensible as
  a robustness check next to the `frac` main table, not as the primary margin.
  Report the resolved delta value, never treat it as a known constant.

- **N-D "upper bound" reporting mode.** For a negative result, the strongest
  presentation is to take the N-D arm's *best* result across the Exp 0 grid —
  an optimistically biased estimate favouring N-D — and show the flat arm still
  matches it. This inverts the Dodge budget asymmetry into support rather than
  a confound. Exp 0 produces the runs; the reporting mode is not implemented in
  `evaluate.py`.

- **Per-dimension N-D rates.** Exp 0 probes the N-D arm at 3-D and transfers
  the result to 2-D and 4-D, spot-checking that transfer on two backbones
  (`EXP0_DIM_CHECK_MODELS`). If `select_lr.py` reports a dim-transfer warning,
  that backbone genuinely needs a per-dimension rate and the grid should be
  expanded for it.

- **The released LR selection is marked `official: false`, and the paper has to
  say why.** `select_lr.py` sets `official = not --allow-unstable`, and the
  selection in use was produced with that flag. 5 of its 39 cells had a ranking
  that had not settled by the stability fraction (0.6 of a 20-epoch window):
  `agg/s4`, `roll/mamba2`, `roll/s4` in selecting arms, `dim2/s4nd` and
  `rollend/mamba3` in check arms. Dropping the flag drops those cells, and the
  main sweep then refuses every run that needs them — so the flag is load-bearing,
  not cosmetic.

  Checked before accepting it: in no unstable cell is the selection materially
  wrong. Two are decisive at the end of the window despite an early crossover
  (`roll/mamba2` picks 1e-4 at 0.00209 against 0.00464 for the runner-up;
  `rollend/mamba3` picks 1e-3 at 0.000296 against 0.000573). The other three are
  near-ties between *adjacent* grid points — `agg/s4` 5.7%, `roll/s4` 2.6%,
  `dim2/s4nd` 1.3% — where either rate is defensible and the documented rule
  (min val_loss; ties to the smaller rate) is applied consistently. Re-running
  Exp 0 at a longer budget to chase this costs ~21 h of 5-card time with no
  guarantee the crossovers resolve. Report the caveat instead.

- **21 of 39 Exp 0 cells select a grid endpoint** (1e-4 or 1e-2 out of
  `1e-4 … 1e-2`), so for those the search is truncated rather than bracketed and
  the optimum may lie outside the grid. This is a property of the released
  artifact, visible per cell as `at_grid_endpoint` in `lr_selection.json`
  diagnostics. Widening the grid was considered and declined: above 1e-2 the
  cells that select the endpoint are the ones already stepping hard rather than
  converging, so a wider grid would reward instability. State the bracket and
  the endpoint count in the paper rather than implying a bracketed optimum.

## Standalone release history

Historical commits above refer to the research source repository, not this fresh
release history. The retired emitter from `032c75f` is preserved verbatim in
`tests/fixtures/retired_sweep.py.txt`, with its checksum and origin in the adjacent JSON.
