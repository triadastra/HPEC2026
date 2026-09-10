# PLAN — Multidimensional Long-Series Forecasting (IEEE HPEC 2026)

> **Status note (2026-08-23):** numeric cohort sizes and draft result claims
> below predate the train-only cohort and strict evaluation fixes. Treat them
> as planning history. The rebuilt lattice sidecar and new sweep manifest are
> authoritative; see `RETRAIN.md` before using any result.

Sequel to the WCTR 2026 study
([Multidimensional-Demand-Forecasting](https://github.com/triadastra/Multidimensional-Demand-Forecasting)).
WCTR finding: flat SSMs beat every multidimensional variant (all ND/CA/FiLM
significantly worse, DM p < 0.05). This project stress-tests that finding at
~30× scale with a new attention module (factorized attention), a restored recurrent baseline
(LSTM), and — new this cycle — a **transport-mode decomposition** of the
target, built from the US **Census** foreign-trade port HS6 files.

> ## ⏩ Current scope — 2026-07 cycle: **Census port HS6 lattice ONLY**
> Trains on **track 1 only** — `census_lattice_9ch.npz` (HS6 × state × flow,
> 28,292 series, 9 mode channels). Every other track is staged but **parked**.
> The **multi-track fusion / aggregation-expansion (P6)**, **HS10**, and the
> other loaders are **deferred to next cycle**. ~~Mamba-3 deferred~~ —
> **Mamba-3 is IN** (implemented + trained this cycle; see §4b).
> **See §4c — PROTOCOL FREEZE (2026-07-13): the run protocol and roster are
> final; all remaining work is analysis and writing.**

---

## 0. ⚠️ Domain framing — READ FIRST (paper-alignment blocker)

The data is **US Census foreign-trade port statistics** (`PORTHS6XM`/`PORTHS6MM`,
census.gov/foreign-trade). It is **import/export trade crossing US ports and
borders**, moving by **air, ocean vessel, and land (truck/rail/pipeline)** —
across 237 partner countries. It is **NOT barge, NOT inland waterway (IWT), NOT
the Mississippi River system, and NOT US Army Corps (USACE) data.** Barge is a
*domestic* mode tracked by USACE (track 3, still PENDING).

The mode decomposition below (air / vessel / breakbulk / land) is itself proof
the domain is **multimodal foreign trade**. Any "barge / IWT / Mississippi /
Army Corps" framing must be corrected to **multimodal international-trade
forecasting through US ports**, OR the study must switch to USACE data (not
built). **Decision required before the paper narrative is finalized.**

---

## 1. Model roster

Kept: **GRU, LSTM, Transformer, XGBoost, S4, S4ND, Mamba-2** (+ their
multidimensional axial / factorized variants). Moving Average = statistical baseline.
- **LSTM restored** (`src/models/lstm.py`, mirror of `gru.py` with `nn.LSTM`).
- **FA (local)** = factorized axial SELF-attention module (`src/models/fa_local.py`,
  `LocalFactorizedAttention`), **not** a standalone model — wired as the
  `{gru,lstm,transformer} × fa_local_{2d,3d,4d}` variants. OFF the declared
  roster as of the FA-naming pass: see RETRAIN.md.
- **FA (authors)** = their unchanged projection/kernel/channel components without
  sphere geometry, behind categorical sparse-lattice glue (`src/models/fa.py`),
  wired as `{gru,lstm,transformer} × fa_{2d,3d,4d}`. The local build remains as
  `fa_local_*`; `authors_cafa_*` is a compatibility alias only. NOTE: CaFA is
  the authors' weather MODEL (ForeCasting with Factorized Attention); the
  operator is FA, so no variant is named `cafa`.
- **S4ND** is natively N-D (no axial/FA wrapper needed); its `grid_*d` tag only
  selects how many axes are promoted — it has no attention.
- Out of headline tables: GPT, LightGBM, naive baselines (code retained).
- ⏸ **Mamba-3** — deferred to next cycle.

At G = 28,292 the *all-group* combo variants (`cross_attention` flat,
28K^2) are infeasible; only **flat** (`onehot`/`embeddings`) and
**axial/FA** (`*_2d/3d/4d`) run at this scale.

---

## 2. HS6 dataset & feature spec (this cycle) — the "series"

### 2.1 Lattice
- **Unit / series:** one `(HS6 commodity × state × flow)` combo.
- **Grid:** `1263 commodity × 14 state × 2 flow` = 35,364 cells; **28,292
  populated** (80% density, 20% masked). 192 monthly points each (2010-01 …
  2025-12).
- **States (14):** CA, FL, GA, IL, LA, MI, ND, NY, OH, PA, SC, TX, VA, WA —
  top-14 by dense-commodity fill; "state" = customs **port-state** (gateway
  geography, not economic origin/destination).
- **Filter:** keep a series only if its aggregate value is non-zero in ≥95% of
  months; then greedy-prune commodities to hold density ≥ 80%.
- **Ports & countries are aggregated away** (not axes): stacking
  port × country collapses density to ~0.02%. Final lattice is **3-D**
  (commodity × state × flow) — *not* 5-D.

### 2.2 Channels (per timestep) — 4 transport modes, kept SEPARATE
Value is reported all-modes; **weight is per-mode and mode-limited** — the file
weighs air + vessel only. Four **mutually-exclusive** modes partition the total:

| mode | value | weight |
|---|---|---|
| **aggregate** | `value_mo` | `air_swt + ves_swt` (recovered) |
| **air** | `air_val` | `air_swt` |
| **vessel — containerized** | `cnt_val` | `cnt_swt` |
| **vessel — breakbulk** (`ves − cnt`) | ✓ | ✓ |
| **land** (truck/rail/pipeline) | `value − air − ves` | — **not weighed** |

→ **5 value + 4 weight = 9 base channels** (land is value-only; that gap is the
signal that explains value/weight mismatch). Mode is thus **numeric**, not a
categorical encoder.

**Feature vector** = 9 channels + month `sin/cos` + lags. Two lag options
(OPEN): **(A) lag the 2 aggregates only → 35 features** (rec; the 36-window
carries per-mode history) or **(B) lag all 9 → 119**.

**Targets** (OPEN): **(A) `(agg value, agg weight)`** (rec; matches WCTR) or
**(B) all per-mode channels**.

### 2.3 Normalization — `log1p` then train-only MinMax (CHANGED)
Plain per-series MinMax is **broken here**: it's train-only (leakage-free, good)
but blind to post-2021 structural growth, so 2024–25 targets run 10–68× out of
range and **~50% of test-MSE comes from ~1% of series** (unforecastable breaks).
Fix (builder-only):
```
raw → log1p → per-series MinMax(fit on train months 0..143, range floored at 1.0)
```
- Value: `log1p` alone tames it (worst 68× → 3.5×; only 4 series >3×;
  MSE-concentration 50% → 22%).
- Weight: needs the **floor 1.0** to stop divide-by-≈0 on near-constant series
  (else max blows to 1e9).
Still MinMax (comparable to WCTR), still leakage-free.

### 2.4 Splits (by target-month index)
train `[0,144)` 2010–2021 · val `[144,168)` 2022–23 · test `[168,192)` 2024–25.
COVID's 2020 crash + 2021 rebound sit safely in **train**.

### 2.5 Data-integrity findings (document these — a contribution in themselves)
1. **`ves_swt` is vessel-only, mode-limited.** 34.5% of value-bearing cells had
   `weight=0` — non-vessel trade (air/land). Raw field is literal `0`, not blank.
2. **Weight recovery:** reading `air_swt` (which the old builder ignored) lifts
   weight coverage **35% → 84%**; 76% of the "missing" weight was air.
3. **Modes are NOT mutually exclusive per cell:** 76% pure, but ~20% carry both
   air and vessel → mode is a **composition**, which is why we keep per-mode
   channels rather than a single dominant-mode label.
4. **Structural breaks in val/test** are drift, not a macro shock (national
   aggregate is clean); handled by `log1p`, not masking.

---

## 3. Encoder × dimensionality — the integrated design (KEY)

Dimensionality = **how many categoricals are promoted from an encoded feature to
an attention axis.** Whatever is *not* an axis MUST be encoded (one-hot /
embeddings) or the model is blind to it. So the encoder acts on the leftovers:

| test | state | commodity | flow | encoder acts on |
|---|---|---|---|---|
| aggregate | summed | summed | summed | — |
| 1-D flat | encoded | encoded | encoded | state + comm + flow |
| 2-D axial | **axis** | encoded | encoded | comm + flow |
| 3-D axial | **axis** | **axis** | encoded | flow |
| 4-D axial | **axis** | **axis** | **axis** | — |

**Consequences:**
- One-hot vs embeddings matters **everywhere except 4-D and aggregate** — so
  Test 2 and Test 3 differ across 1-D/2-D/3-D (they only coincide at 4-D). The
  earlier "N-D columns identical" shortcut is **void**.
- **New code required:** the combo models currently set `encoder = nn.Identity()`
  for all dims. They must instead **encode the not-yet-an-axis categoricals**
  (look up each group's leftover ids from `combo_coords`, encode, concat to the
  per-group features), keyed on `dims`. Applies to GRU/LSTM/Transformer combo
  classes + the encoder classes.

**Encoders:**
- **One-hot:** append K bits; `input_proj` projects the wide vector to hidden.
  (One-hot + first Linear *is* a hidden-dim embedding.)
- **Embeddings:** `nn.Embedding` lookup, dims by fast.ai
  `d = min(600, round(1.6·n^0.56))` → **commodity 88, state 7, flow 2**.
- Embed dim is capped by hidden size (past it, can't flow through); category
  *count* is not (128 dims separate ≫128 categories).

**Sharpening (optional, elevates Objective 3):** map the one-hot→embedding
**crossover vs vocabulary size** (commodity top-N = 100/500/1263) — a scaling
curve, not a single point. Expect the encoder gap to **shrink as dims rise**
(each promotion removes a categorical from the encoder's job).

---

## 4. Experiments

- **Test 1 — Aggregate.** Collapse all series → one (value, weight) series; all
  models, no encoders. Sanity + baseline.
- **Test 1.1 — Annual rolling-origin aggregate.** Six expanding-origin folds:
  train through Y−2, validate Y−1, and test the 12 months of Y for
  Y=2020…2025. Refit the aggregate transform on each fold's training months;
  retrain every model from scratch in a fresh process. The aggregate is summed
  from the raw universe before cohort selection, and each in-memory fold is
  truncated at its test end. See `catalog.md` for the exact boundaries.
- **Test 2 — One-hot N-D.** 1-D vs 2-D(state) vs 3-D(+comm) vs 4-D(+flow), for
  **Axial self-attention and factorized attention**, one-hot on the leftovers.
- **Test 3 — Embeddings N-D.** Same, embeddings on the leftovers.

**Metrics:** MAE, MSE, RMSE, sMAPE%, **+ effect sizes**. ⚠️ At N=28,292 the DM
test saturates — nearly everything is "p<0.05," so significance stops
discriminating; **report effect size / practical delta / compute cost**, not
just p-values. Also report a **structural-break-robust cut** (metrics excluding
or down-weighting the ~1% breakout series) so ranking isn't dominated by
unforecastable cells.

### 4.1 Efficiency & FLOPs instrumentation (HPEC core)

Efficiency is half the thesis (Objectives 1–2), so **every run emits a cost row**
and the headline figure is a cost–accuracy Pareto, not a table.

**HW: 8× H20, NVLink.** Memory-rich (96 GB HBM3/GPU), **compute-modest** (BF16
peak well below H100 — fill exact TFLOP/s from the H20 spec for MFU), high
NVLink bandwidth. This profile makes **FLOP-efficiency the pointed metric**:
compute is the scarce resource, so "accuracy per FLOP" is the honest axis, and
the memory-heavy axial path is exactly what this box is built to hold.

Hardware-independent (deterministic — reviewer-proof):
- **Parameters** per model × variant.
- **FLOPs / MACs** per forward pass (and per train step ≈ 3× fwd for fwd+bwd) —
  the primary compute number, via `torch.utils.flop_counter` or `fvcore`.
  **Static** — computable for the whole roster *before* any training.
- **Peak activation memory** per batch (the binding constraint;
  combo ≈ 114 MB/sample → combo batch ≈ 4).

Hardware-specific (with controls: `cuda.synchronize()`, warmup discarded, fixed
precision, mean ± std over the 3 seeds, data-loading measured separately):
- **Throughput** (samples/s, train + inference) — normalizes the flat-batch-512
  vs combo-batch-4 mismatch so the two are comparable.
- **MFU** (achieved FLOP/s ÷ H20 BF16 peak) — flags compute- vs memory/comm-bound;
  especially diagnostic on the compute-modest H20.
- **Time-to-target**, **inference latency/forecast**, optional **energy**
  (NVML joules/sample — HPEC-friendly).
- **8×-GPU use:** default = **8 concurrent single-GPU runs** (sweep throughput →
  GPU-hours to finish the roster; NVLink idle). Optional: **shard the 28,292
  groups across GPUs over NVLink** for the heaviest 4-D CaFA — if used, report
  strong-scaling efficiency + all-reduce/NVLink cost (a genuine HPEC result).

**Headline figure (answers Objective 2):** sMAPE/MSE vs cost (FLOPs *or* peak
memory) **Pareto**, points = flat / axial / factorized × dims — you can *see*
whether multidimensionality earns its place, and whether factorization moves the
frontier vs dense axial. The **batch-4 ceiling on the axial path is itself a
reported result** ("the multidimensional path cannot exploit large-batch
throughput at scale"), not a footnote.

**Training setup — RESOLVED, see §4c freeze.**

---

## 4b. Executed experiment set (2026-07 cycle, as actually run)

- **Test 1 — Aggregate** (18 runs): 6 flat models × 3 seeds, one national series.
- **Test 2 — One-hot N-D** (72 runs): 6 flat + {gru,lstm,transformer} ×
  {asa,fa_local,fa} × {2,3,4}d, onehot leftovers.
- **Test 3 — Embeddings N-D** (81 runs): same + mamba_nd (native-ND scan,
  grid axes) × 3 dims.
- **Test 4 — Identity-aware axial** (108 runs): the full two-mixer grid ×
  BOTH encoders × `--axis-identity` (learned embeddings on promoted axes).
  Kills the permutation-equivariance confound: Tests 2/3 axial modules are
  provably identity-blind along promoted axes while flat baselines always see
  identity. `mamba_nd` excluded (ordered scan = position-aware).
- **Exp 0 — learning-rate selection** (210 runs, one seed, 20-epoch budget):
  five rates spanning two decades × every (arm, model) cell in the five
  selecting arms (flat / nd / aggregate / rolling / fa-mech), plus six
  transfer-check arms (dim2/dim4, rollend, encflat, encnd, axid) — see
  `catalog.md` for the authoritative breakdown. Selected on validation loss
  only; the rest of the matrix then trains at the selected rate. Supersedes the retired Test 5, which searched four rates over
  one decade for the flat arm alone -- the multidimensional arm it was compared
  against received no search at all. Mamba-3's rotational state diverges
  per-series at the shared 1e-3 *despite* grad-clip 1.0, and underfits at 1e-4
  (worse than persistence); that finding stands on the Test-5 runs already
  collected and is reported from those rather than re-trained.
- **Statistical baselines** (analytic, no training): persistence(T−1)
  norm-MSE 0.0533, seasonal-naïve(T−12) 0.0685 on the 2024-25 test window —
  the floor every learned model must beat (best learned: 0.0348).
- **Budget-uniformity refit** (24 runs): early-phase Test-3 embeds combos ran
  under a 60-epoch cap; the 24 runs whose best checkpoint fell at epoch ≥30
  are re-run under the uniform 30-cap (old artifacts archived).

## 4c. PROTOCOL FREEZE — 2026-07-13 (final; changes require restarting the benchmark)

- **Roster:** gru, lstm, transformer, s4 (genuine S4Block/DPLR), mamba2 (SSD),
  mamba3 (SISO, ICLR 2026), mamba_nd. XGBoost/S4ND/GPT/LightGBM out of headline
  tables (code retained). Naive baselines = analytic persistence/seasonal rows.
- **Architecture:** hidden 128, 4 layers/blocks, dropout 0.1, last-timestep
  readout everywhere; transformer sinusoidal PE.
- **Training:** AdamW (β 0.9/0.999, wd 1e-4), lr selected per (arm, model) by
  Exp 0 on validation loss from the same five-rate grid for every cell in every
  arm), grad-clip 1.0
  (uniform, all runs), ReduceLROnPlateau (patience 5, ×0.5), early stop
  patience 10, **epoch cap 200**, no AMP, best-val checkpointing on combined
  norm-MSE.
- **Batches:** flat 512 · axial/combo 1 (measured training peak 40-61 GB at
  G=28,308; worst cell mamba3_fa_4d 61.4 GB) ·
  aggregate 32. Equal within task type; the combo batch ceiling is a reported
  result.
- **Data:** `census_lattice_9ch.npz` + sidecar json (targets = channels [0,5] =
  agg value/weight), lag_mode=agg (F=35), input 36 mo, splits 144/168/192,
  log1p + train-only floored MinMax. Strictly causal windows `[t−36, t) → t`.
- **Seeds:** 947 / 732 / 619 (config seed set retired).
- **Hardware:** 5× H20 96 GB (job-parallel, one run per GPU); torch 2.13 env
  for mamba*, torch 2.8 for the rest; two-pass eval merge.
- **Known-identical cells:** at 4-D the leftover-encoder set is empty, so
  onehot ≡ embeddings architecturally; their spread is reported as the
  measured nondeterminism floor.

---

## 5. Implementation task list

- [ ] **P-A Rebuild `build_census_lattice.py`** → new `census_lattice.npz`:
  read all mode fields (`value_mo, air_val, air_swt, ves_val, ves_swt, cnt_val,
  cnt_swt`); `weight = air_swt + ves_swt`; derive 9 channels (4 modes + agg);
  `log1p` + floored train-MinMax; store per-mode channels + mask. Re-parse
  192×2 monthly zips.
- [ ] **P-B Update `census_loader.py`** → emit the 9-channel feature panel
  (35 or 119 features), read new npz. (Flat + combo datasets already drafted.)
- [ ] **P-C Encoder × dims wiring** — add leftover-categorical encoding to the
  combo models keyed on `dims`; extend `OneHotEncoding`/`EmbeddingEncoding` +
  `base.py`. ✅ `base.py` combo-encoder remap bug already fixed (staged).
- [ ] **P-D Configs** — `census.yaml` (feature count, splits, dims), the three
  `cafa_*.yaml` (done), reconcile hyperparams.
- [ ] **P-E Efficiency panel (static — do now)** — params + **FLOPs** (per fwd +
  per train step) + theoretical peak-memory counter over the whole roster (one
  script, no training); fills half the cost table immediately. Throughput / MFU /
  time-to-target / energy hooks land in the training loop (P-H). Deliverable: the
  §4.1 cost–accuracy Pareto.
- [ ] **P-F PR** — (1) push the standalone `base.py` CaFA fix now; (2) the census
  bundle (builder + loader + encoders + configs + this PLAN) once coherent.
- [ ] **P-G VM verify** — move new npz; 1-batch torch forward on a flat model +
  `gru fa_local_2d` + a mode-aware run; confirm `lstm` registers.
- [ ] ⏸ **P-H Training sweep** — deferred per "no batches yet": roster × lattice
  × 3 seeds, flat-first then axial/CaFA; effect sizes + HPEC throughput/MFU.

---

## 6. Open decisions — ALL RESOLVED 2026-07-13 (see §4c freeze)

1. **Narrative:** ✅ multimodal foreign-trade through US ports (the 9-channel
   mode decomposition *is* the framing; no barge/IWT/USACE language).
2. **Targets:** ✅ (A) `(agg value, agg weight)`; per-mode channels are lagged
   covariates (ARX structure), `lag_mode="all"` stated as future ablation.
3. **Lags:** ✅ (A) 35 features (aggregate lags only).
4. **Hyperparams:** ✅ 4-layer / 128 / flat-batch 512 / combo-batch 1 /
   seeds 947-732-619 / 30-epoch cap.
5. **Objective 3 scope:** ✅ single-cardinality encoder comparison this cycle;
   the crossover-vs-vocab curve deferred.

---

## 7. Data program (multi-track) — context; all but track 1 ⏸ parked

HS2 = universal commodity spine. Tracks stay separate (no fusion) until per-track
baselines exist. The finer land/mode split the paper envisions (air/barge/
truck/rail) is the **multimodal fusion**: Census (air, vessel) + USACE (barge) +
TransBorder (truck, rail) — deferred.

| # | track | axes | status |
|---|---|---|---|
| 0 | Trade (WCTR, in repo) | 8 st × HS2(93) × Im/Ex × month | ✅ committed |
| 1 | **Census port HS6 (PRIMARY)** | US ports × HS6 × flow × month, 2010–25 | ✅ pulled; **lattice built + being rebuilt (mode/log1p)** |
| 2 | BTS TransBorder | state × HS2 × mode(truck/rail/vessel) × port × month | ✅ pulled (Wayback) ⏸ |
| 3 | USACE WCSC/LPMS (**barge**) | waterway/lock × WCSC→HS2 × dir × month | ⛔ pending (browser+WAF) ⏸ |
| 4 | FAF5 | FAF region × SCTG2→HS2 × mode × year | ✅ pulled ⏸ |
| 5 | STB Waybill | BEA × STCC→HS2 × rail × quarter | ✅ pulled ⏸ |
| 6 | Eurostat COMEXT | EU × HS6 × mode(9) × flow × month | ✅ pulled ⏸ |
| 7 | Brazil Comex Stat | BR state × NCM→HS2 × via(12) × month | ✅ pulled ⏸ |
| 8 | StatCan CIMT | CA prov × HS6→HS2 × partner × month | candidate ⏸ |

All D:/ pulls gitignored (see `D:/wctr_data/MANIFEST.md`). Detail:
`milestones.md` §3–§5.
