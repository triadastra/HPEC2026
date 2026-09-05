# Are Multidimensional Models Worth Their Computational Cost in Demand Forecasting?

Code and benchmark for the IEEE HPEC 2026 paper — the sequel to
[Multidimensional-Demand-Forecasting](https://github.com/triadastra/Multidimensional-Demand-Forecasting)
(WCTR 2026). We benchmark RNNs, Transformers, gradient boosting, and four
generations of state-space models (S4, S4ND, Mamba-2, Mamba-3) on a US
foreign-trade lattice and ask whether explicitly multidimensional
architectures earn their compute.

**Release status:** local publication candidate for `triadastra/HPEC2026`.
The pinned Hugging Face archive contains the completed 477-run main matrix,
195 base LR probes, 25 FA LR probes, and 27 runs each for Exp 7 and Exp 8.
These are archived counts; the current generator's Exp 0 plan is 240 runs.
See [the saved audit](docs/audit/completed-training-audit.md) for the evidence
and remaining qualifications. The archive is not proof of a fresh inference run.

## Start here

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt
python scripts/release.py test
python scripts/release.py results
python scripts/release.py checkpoint --run gru_embeddings_1d_s947
python scripts/release.py plan
```

`results` downloads the paper-facing tables from the pinned Hugging Face revision.
`checkpoint` verifies manifest membership, the completion fingerprint and the
checkpoint SHA-256, then loads the tensor state dictionary on CPU. Neither
command starts training. See [REPRODUCING.md](REPRODUCING.md) for data setup,
CUDA training, evaluation and limitations of replaying archived runs.
The upstream FA code is included at its pinned revision; no submodule setup is needed.


## Dataset

**Input release gap:** the published HF dataset is the older 28,292-series build;
the checkpoints use 30,087 series and 1,343 commodities. `release.py data` detects
and rejects this mismatch. See [REPRODUCING.md](REPRODUCING.md) before evaluation.


US Census foreign-trade port HS6 files (`PORTHS6MM`/`PORTHS6XM`), 2010–2025.
The builder selects dense series, top states, and retained commodities from
2010–2021 only, then freezes that cohort before touching validation or test
targets. The generated sidecar JSON is the authority for lattice dimensions
and series count. The panel has 192 monthly points and 9 transport-mode
channels (5 value + 4 weight; land is not weighed at source).
Targets are next-month **(aggregate value, aggregate weight)** per series —
categorical identifiers are inputs only. Features per timestep: 9 channels +
12 lags of each target + month sin/cos = 35. Normalization: log1p then
per-series MinMax fit on training months only. Chronological splits:
2010–2021 train / 2022–2023 val / 2024–2025 test. Build:
`scripts/build_census_lattice.py` → `data/census_port/processed/census_lattice_9ch.npz`.

## Exp 0, the six tests, and the two diagnostic arms (`scripts/sweep.py --tests N`)

| test | what | runs |
|------|------|------|
| 0 | **Exp 0**: learning-rate selection, seven arms + five transfer checks. Run first, alone; gates the rest | 240 |
| 1 | Aggregate: one national series, all flat models (+ `scripts/xgb_agg.py`) | 18 |
| 1.1 | Annual rolling aggregate: six expanding-origin folds, fold-local transform | 108 |
| 2 | One-hot N-D: flat + ASA / authors-FA grid, one-hot leftovers | 72 |
| 3 | Embeddings N-D: same grid, embeddings encoder (+ Mamba-ND) | 81 |
| 4 | Identity-aware axial: Test 2/3 grid × both encoders × `--axis-identity` | 108 |
| 6 | SSM completion: Mamba-2/3 × both attention mixers + genuine S4ND | 90 |
| 7 | **Exp 7**: the replication. `fa_local_*` = OUR build of the FA operator, the one the draft's CaFA numbers came from, rerun under the current protocol. Run alone; diagnostic, not part of the 477 | 27 |
| 8 | **Exp 8**: FA kernel nonlinearity. `fa_sm_*` = the authors' own `softmax=True` switch. Only meaningful after Exp 7. Run alone; diagnostic, not part of the 477 | 27 |

Uniform protocol (PLAN.md §4c): hidden 128 / 4 layers / dropout 0.1 / AdamW
(wd 1e-4) / per-(arm, model) LR selected by Exp 0 / grad-clip 1.0 / 200-epoch cap / early
stopping patience 10 / best-val checkpoint / seeds 947-732-619. Flat runs use
gradient accumulation to match the multidimensional arm's effective batch and
optimizer-step count; multidimensional batch is 1 and aggregate batch is 32.

## Model taxonomy

|  | per-line sequential | shared factorized kernel |
|--|--------------------|--------------------------|
| **attention** | axial self-attention (`combo_attention.py`) + axial cross-attention (`aca.py`, opt-in) | the authors' FA operator (`fa.py`) |
| **SSM** | Mamba-ND (`mamba_nd.py`, scan) | S4ND (`s4nd.py`, separable DPLR LTI kernels) |

Flat backbones: GRU, LSTM, Transformer, S4 (genuine `S4Block`/DPLR), Mamba-2
(SSD), Mamba-3 (SISO). Test 6 additionally fills the hybrid cells
(`mamba_axial.py`: the exact attention grid-mixing harness of the RNN combos
with the temporal operator swapped to Mamba stacks). SSM kernels are vendored
(`external/mamba`, `external/s4`) and run without compiled CUDA extensions
(Triton / cauchy_naive fallbacks).

The authors' unmodified [BaratiLab/CaFA](https://github.com/BaratiLab/CaFA)
repository is pinned as `external/cafa-authors`. `fa_{2d,3d,4d}` imports its
real `PoolingReducer`, `LowRankKernel`, `MLP`, and `GroupNorm` while local code
supplies categorical sparse-lattice glue. It isolates the **FA operator**
without the weather-only sphere geometry — running the authors' own
no-positional-encoding path (`rope_module=None`, `modulation=None`) at their
released `kernel_multiplier=2` / `qk_norm=True` / non-softmax settings. It is
**not** a reproduction of CaFA, which is the complete weather model.
`authors_cafa_*` is accepted only as a legacy alias for `fa_*`.

A local reimplementation of the same operator (`fa_local_*`,
`src/models/fa_local.py`) exists but is **off the declared roster**. It differs
from `fa_*` in at least four ways at once — softmax vs LeakyReLU gating,
gamma-MLP vs `PoolingReducer`, single-head vs `LowRankKernel`, and a different
sparse renormalization — so "`fa` vs `fa_local`" cannot be attributed to any one
of them. In the paper, CoFA maps to `fa_local_*`; the authors-operator comparison maps
to `fa_*`. Keep these labels distinct when interpreting the results. The code and its regression tests stay (they carry the
sparse-renormalization fix).

It is, however, the arm the **submitted draft's** CaFA numbers came from (the
workbook spells it `cafa_*`), and **Exp 7 (`--tests 7`) is its completed rerun** under the current
cohort and protocol, and it is the one
that separates a change of operator from a change of protocol. `--mechs
asa,fa,fa_local` still reaches it inside the main roster if the full 144-run
version is ever wanted.

### Naming: CaFA, FA, and self- vs cross-attention

Three corrections, applied throughout. Legacy spellings still resolve in code
so old manifests load, but nothing new is generated under them.

* **CaFA is the authors' weather model**, not an operator: *Fore**Ca**sting with
  **F**actorized **A**ttention* (Li et al. 2024). The operator inside it is
  **FA**; their code calls it `FABlockS2`. This benchmark runs the operator, not
  the model, so no variant here is named `cafa`.
* `fa_{2,3,4}d` is the **authors' FA operator** (their `PoolingReducer`,
  `LowRankKernel`, `MLP`, `GroupNorm`, their non-softmax path, geometry-free).
  `fa_local_{2,3,4}d` is a **local reimplementation** of the same operator with
  softmax kernels — formerly `cafa_*`, which borrowed the model's name. It is
  off the declared roster (see above).
* `asa_{2,3,4}d` is **axial self-attention** — formerly `cross_attention_*d`,
  which meant "attention across groups" and collided with the standard term.
  Q, K and V all project the same tensor. Upstream keeps the same distinction
  (`LowRankKernel` defaults `u_y = u_x`; `CABlock` is the cross-attentive one).
* `aca_{2,3,4}d` is **axial cross-attention** — the only arm whose Q and K
  come from different sources. Opt-in; see below.
* `grid_{2,3,4}d` tags the **grid-native SSMs** (`s4nd`, `mamba_nd`), which
  contain no attention at all; the tag only selects how many axes are promoted.

| old | new |
|---|---|
| `cross_attention_{2,3,4}d` | `asa_{2,3,4}d` (attention hosts) |
| `cross_attention_{2,3,4}d` | `grid_{2,3,4}d` (`s4nd`, `mamba_nd`) |
| `cafa_{2,3,4}d` | `fa_local_{2,3,4}d` |
| `fa_{2,3,4}d`, `authors_cafa_*` | unchanged |

### Axial cross-attention (`aca_*`, opt-in)

Every other attention arm here is **self**-attention: `asa` and `fa` both
derive Q and K from the same source. `aca_{2,3,4}d` is the one arm where
they do not — each promoted axis queries the pooled **complement** of that axis:

```
U_A   = masked mean over every axis except A     -> (B, L, |A|, H)   # queries
U_M   = masked mean over A, flattened            -> (B, L, M,  H)   # keys/values
attn  = softmax(W_q U_A · (W_k U_M)ᵀ / sqrt(H))  -> (B, L, |A|, M)
out   = lattice + broadcast(attn · W_v U_M along A)
```

**Provenance, stated honestly.** There is no canonical axial-cross-attention
paper to clone. Ho et al. 2019 (arXiv:1912.12180), the origin of axial
attention, calls it "a simple generalization of **self**-attention";
Axial-DeepLab (arXiv:2003.07853) adds position-sensitive encodings but is still
self-attention (its categorical analogue here is `--axis-identity`, Test 4). The
closest published pattern is axial-centric cross-plane attention for 3D medical
imaging (arXiv:2602.21636) — primary plane supplies queries, complementary
planes supply keys/values, "directional cross-plane fusion". `aca_*` adapts that
to a categorical lattice. It is **our construction**, not a reproduction, and
the paper must describe it that way.

### FA head geometry

The main table is pinned at `fa_heads: 4`, `fa_dim_head: 32`,
`fa_kernel_multiplier: 2`. The claim is exactly:

> FA geometry is scaled for this benchmark's 128-dimensional residual width:
> four 32-dimensional value heads give a 128-dimensional value space, while
> `kernel_multiplier=2` gives 64-dimensional Q/K features per head. This is a
> benchmark-specific scaling choice, not an invariant inherited from the
> authors' substantially larger 768-wide processor.

It is **not** "we matched the other attention models" and **not** "we reproduced
the authors' head geometry" — these are different quantities:

| | model width | heads | value dim/head | value width | value/model | kernel Q/K dim/head |
|---|---:|---:|---:|---:|---:|---:|
| transformer / gpt | 128 | 8 | 16 | 128 | 1.00× | 16 |
| `fa_*` here | 128 | 4 | 32 | 128 | 1.00× | 64 |
| authors' processor | 768 | 16 | 64 | 1024 | 1.33× | 128 |

The kernel dim is `dim_head × kernel_multiplier`, so 32 gives a 128-wide value
space with 64-dim Q/K per head.

The authors' 64 was never used at a 128-wide model. Their processor carries a
**768-wide** residual stream: `era5-EPD-240121-*.yml` sets `model.latent_dim:
768`, which [`weather_transformer.py`](external/cafa-authors/weather_transformer.py)
passes to `FactFormerS2` as `dim` and on to `FABlockS2` as the residual/channel
width. The 384 in that config is `model.processor.latent_dim` — the
`PoolingReducer` bottleneck the axial kernels are computed in, *not* the model
width. So their 16 × 64 = 1024 of value width is ≈**1.33×** the residual width
(1024 / 768). They do not hold `heads × dim_head = width`, but only mildly; in
their smaller 512-wide config (`era5-EPD-6432-*.yml`) the same 1024 is 2.0×.

Either way, keeping 64 while shrinking heads and width to 128 would reproduce
neither their ratio nor their scale — 4 × 32 is a closed geometry chosen for
this 128-wide benchmark, not an invariant inherited from the authors'
substantially larger processor. Raising transformer/gpt to 32 is worse still: at
width 128 that means 8→4 heads (a different attention factorization, not a
bigger head), and keeping 8 heads would force width 256 and invalidate every
baseline.

`--fa-dim-head-ablation` runs {16, 32, 64} at fixed heads and multiplier on one
representative cell (`gru` + `fa_2d` + embeddings, 3 seeds = **9 runs**) to show
32 is not arbitrary. Diagnostic only: the main table stays pinned at 32 and is
never selected from that sweep on test data.


Opt in with `--mechs asa,fa,aca`. It costs **+144 runs** on top of the declared
477, which is why it is not a default. The cost panel measures it
either way (`model_cost.py` is static), so the params/FLOPs are available before
committing GPU time.

## Reproduce

```bash
# Pinned upstream FA source is already included.
pip install -r requirements-cuda.txt
python scripts/check_environment.py --require-mamba

# train one configuration
python scripts/train.py --config config/base.yaml --data-config config/census.yaml \
    --model s4nd --variant grid_4d --combo-encoder embeddings \
    --batch-size 1 --epochs 200 --seed 947 --out-dir outputs/sweep/demo

# a whole test, job-parallel across GPUs
python scripts/sweep.py --tests 6 --gpus 0,1,2,3 --epochs 200 --reset-manifest

# a full official rerun gets a new isolated, resumable session directory
GPUS=0,1,2,3 TESTS=1,1.1,2,3,4,6 RESET_MANIFEST=1 \
    bash scripts/hpec_pipeline.sh

# evaluate every checkpoint -> metrics.json (per seed) + results.csv (seed-averaged)
python scripts/evaluate.py --runs outputs/sweep

# cost panel (params / FLOPs / peak mem; fails if any required cell is unmeasured)
CUDA_VISIBLE_DEVICES="" python scripts/model_cost.py

# statistical significance (Diebold-Mariano + Wilcoxon + paired t)
python scripts/dump_errors.py s4_embeddings_1d_s947 ... # per-series error dump
python scripts/significance_tests.py

# XGBoost baseline on the aggregate test
CUDA_VISIBLE_DEVICES="" python scripts/xgb_agg.py

# mechanical verification of the task spec against the live code
CUDA_VISIBLE_DEVICES="" python tests/verify_task_spec.py
```

Evaluation preserves both levels of detail: `metrics.json` contains one row per
run and seed, while `results.csv` is the separate compact seed-averaged summary.
The manifest validates completeness/provenance only; it does not collapse the
per-seed records.

Mamba models require Linux, an NVIDIA CUDA runtime, and Triton. The committed
CUDA requirements and environment check define one installable roster
environment; `scripts/sweep.py --models <list>` can still split jobs across
machines without changing their dependency contract.

## Repository structure

```
PLAN.md                  living plan; §4b executed test set, §4c protocol freeze
src/models/              model zoo (local FA build + authors' FA, S4ND, Mamba variants)
src/data/census_loader.py  lattice loader: features, splits, flat+combo datasets
scripts/                 build_census_lattice · train · sweep · evaluate ·
                         model_cost · xgb_agg · dump_errors · significance_tests
tests/                   test_fa_local_smoke.py · test_fa_smoke.py · verify_task_spec.py
config/                  base.yaml · census.yaml · models/ · variants/
external/                vendored mamba, s4, and the pinned authors' CaFA submodule
```

## Predecessor

The predecessor's 903-series data, reports, figures, and scripts are preserved
removed from this repository. Its five-test-month results used a
different cohort, split, and seed protocol and are intentionally excluded from
the current benchmark artifacts.

## License

MIT (see `LICENSE`). Vendored code under `external/` retains its upstream
licenses.
