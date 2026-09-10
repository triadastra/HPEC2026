# Are Multidimensional Models Worth Their Computational Cost in Demand Forecasting?

**Cheng-Jui Fan · Nikolay Aristov · Elenna R. Dugundji**

IEEE High Performance Extreme Computing (HPEC) 2026

Code, experiment pipelines, and results for benchmarking the accuracy and
computational cost of multidimensional demand-forecasting models. We compare
statistical baselines, gradient boosting, RNNs, Transformers, and state-space
models on monthly U.S. Census merchandise-trade data.

**[Hugging Face Dataset](https://huggingface.co/datasets/Celsia/HPEC2026)** ·
**[Hugging Face Models & Checkpoints](https://huggingface.co/Celsia/HPEC2026)** ·
**[Result Tables](results/)** ·
**[Reproduction Guide](#training-and-evaluation)**

> This public-facing repository was cleaned using Anthropic Fable 5.1 and OpenAI Astra 6 to make the code easier for future researchers to run. No experiments or configurations have been altered.

## Overview

The benchmark asks whether explicitly modeling the state, commodity, and trade-flow
axes improves forecasting enough to justify its computational cost. It compares
flat encoders with Axial Self-Attention (ASA), Convex Factorized Attention (CoFA),
the factorized-attention operator adapted from CaFA, and native multidimensional
SSMs, including S4ND and Mamba-ND.

The experiments cover fixed aggregate forecasts, rolling-origin aggregate
forecasts, and forecasts for individual state–commodity–flow series. Evaluation
includes MSE, MAE, RMSE, sMAPE, MASE, model size, inference FLOPs, and statistical
comparisons. See the [archived tables](results/) and
[statistical audit data](docs/audit/significance_extended.json) for results and qualifications.

## Dataset

The **[HPEC2026 dataset on Hugging Face](https://huggingface.co/datasets/Celsia/HPEC2026)**
contains the processed lattice, its required JSON sidecar, and the source Census
port-level HS6 archives (`PORTHS6MM` for imports and `PORTHS6XM` for exports).

| Property | Benchmark specification |
|---|---|
| Period | January 2010–December 2025; 192 months |
| Categorical axes | 1,343 commodities × 14 states × 2 flows |
| Possible combinations | 37,604 |
| Retained series | 30,087; 80.01% lattice density |
| Numerical channels | 9: five value channels and four weight channels |
| Forecast targets | Next-month aggregate trade value and aggregate shipping weight |
| Input features | 35 per timestep: 9 channels + 24 target lags + month sine/cosine |
| Input history | 36-month window with 12 months of lag history |
| Training / validation / test | 2010–2021 / 2022–2023 / 2024–2025 |
| Normalization | `log1p`, then per-series MinMax fitted on training months only |

Cohort filtering and state/commodity selection use training months only. After
reserving the required history, the first training target is January 2014.
The retained series contain **11,553,408 scalar value/weight observations** and
**8,665,056 eligible scalar forecasting targets** across all three splits.

The corrected input pair is pinned to dataset revision
[`d3f61546516002a44f79962507422f47c4f80263`](https://huggingface.co/datasets/Celsia/HPEC2026/tree/d3f61546516002a44f79962507422f47c4f80263).
The earlier 28,292-series build is preserved under `processed/legacy_precorrection/`
for historical inspection. Use the corrected pair for this benchmark.

**Manuscript correction:** the supplied paper's dataset paragraph retains the
older 1,263-commodity count. The released data and checkpoints use **1,343
commodities, 37,604 possible combinations, and 80.01% density**. The commodity
embedding width remains fixed at **88**, as used in the experiments; it is not
recomputed from the corrected commodity count.

## Quick start

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt

# Run the portable test suite.
python scripts/release.py test

# Download and verify the exact processed dataset and sidecar.
python scripts/release.py data
python scripts/release.py data-check
python tests/verify_task_spec.py

# Download saved result tables and verify an archived checkpoint.
python scripts/release.py results
python scripts/release.py checkpoint --run gru_embeddings_1d_s947

# Run the saved GRU on a small synthetic input.
python scripts/release.py smoke

# Generate the main experiment roster without starting training.
python scripts/release.py plan
```

`data` verifies the pinned SHA-256 checksums and refuses to overwrite an existing
input pair. If the data is already installed, continue with `data-check`.
`checkpoint` verifies manifest membership, completion provenance, and checkpoint
SHA-256 before loading its tensors on CPU. `smoke` checks model compatibility;
it does not measure forecasting accuracy.

The **[model repository on Hugging Face](https://huggingface.co/Celsia/HPEC2026)**
hosts checkpoints, training curves, manifests, and saved results. This release
pins model revision
[`51e064a2f8e1af1ff867794b5ce8aeeade4692cd`](https://huggingface.co/Celsia/HPEC2026/tree/51e064a2f8e1af1ff867794b5ce8aeeade4692cd).
Model and dataset repositories share the name `Celsia/HPEC2026` but are distinct
Hub repository types. [SOURCE.json](SOURCE.json) records both pins and the input
checksums. Downloaded artifacts stay outside Git under `outputs/` and `data/`.

## Models and naming

| Family or mechanism | Models / code variants |
|---|---|
| Statistical baselines | Persistence, 12-month moving average, seasonal naive |
| Gradient boosting | XGBoost on the aggregate test |
| Flat neural backbones | GRU, LSTM, Transformer, S4, Mamba-2, Mamba-3 |
| Axial Self-Attention (ASA) | `asa_2d`, `asa_3d`, `asa_4d` |
| Convex Factorized Attention (CoFA) | `fa_local_2d`, `fa_local_3d`, `fa_local_4d` |
| Authors' FA operator | `fa_2d`, `fa_3d`, `fa_4d` |
| Authors' FA with softmax | `fa_sm_2d`, `fa_sm_3d`, `fa_sm_4d` |
| Native multidimensional SSMs | S4ND and Mamba-ND with `grid_2d`, `grid_3d`, `grid_4d` |

`fa_*` uses the upstream [CaFA](https://github.com/BaratiLab/CaFA) components with
categorical-lattice adaptations. It isolates the FA operator; it does not reproduce
the full weather model. CoFA (`fa_local_*`) uses a local implementation with
softmax kernels and sparse renormalization. Their differences involve multiple
architectural choices, so their comparison is not a single-factor ablation.
Pinned upstream FA source is included under `external/cafa-authors/`; no submodule
initialization is required.

## Experiments

Select experiments with `scripts/sweep.py --tests N`.

| Test | Description | Runs |
|---|---|---|
| 0 | Learning-rate selection: seven arms and five transfer checks | 240 |
| 1 | Fixed aggregate forecast | 18 |
| 1.1 | Rolling-origin aggregate forecast over six folds | 108 |
| 2 | One-hot encoders with flat, ASA, and FA variants | 72 |
| 3 | Embedding encoders with flat, ASA, FA, and Mamba-ND variants | 81 |
| 4 | Identity-aware axial variants with both encoders | 108 |
| 6 | SSM completion: Mamba attention hybrids and S4ND | 90 |
| 7 | CoFA diagnostic using the local FA implementation | 27 |
| 8 | Authors' FA softmax diagnostic | 27 |

Tests **1, 1.1, 2, 3, 4, and 6 form the 477-run main matrix**. Experiments 7 and
8 run separately. The archived sessions contain 195 base LR probes, 25 FA LR
probes, and 27 runs for each diagnostic; the current Exp 0 generator's 240-run
plan is distinct from those archived LR counts.

The neural protocol uses hidden width 128, four layers, dropout 0.1, AdamW with
weight decay `1e-4`, gradient clipping at 1.0, a 200-epoch cap, early-stopping
patience 10, and the best validation checkpoint. Seeds are **947, 732, and 619**.
Flat runs accumulate gradients to match the multidimensional effective batch
and optimizer-step budget. Multidimensional batch size is 1; aggregate batch
size is 32. Learning rates are selected per arm and model.

Optional axial cross-attention (`aca`) or adding `fa_local` throughout the main
roster costs **+144 runs** per mechanism. These extensions are outside the
default matrix. The optional identity-aware SSM hybrid arm adds **+72 runs**.
Test 5 is retired; its historical emitter from `032c75f` is preserved in
`tests/fixtures/retired_sweep.py.txt` for inspecting older runs.

## Training and evaluation

The full model roster requires **Linux, an NVIDIA CUDA runtime, and Triton**.
After installing and validating the dataset:

```bash
pip install -r requirements-cuda.txt
python scripts/check_environment.py --require-mamba

GPUS=0,1,2,3 RESET_MANIFEST=1 SESSION_ID=public-reproduction \
  python scripts/release.py train
```

This runs learning-rate selection, main-matrix training, evaluation, model-cost
measurement, and the aggregate XGBoost baseline. Resume the same session without
`RESET_MANIFEST=1`. Diagnostic runs, error dumps, and significance analysis are
shown below.

Evaluation retains per-run metrics and separate seed-averaged summaries. Strict
manifest checks protect against incomplete or mismatched experiment outputs.
Archived source/config/data fingerprints must be respected when replaying saved
runs in a different checkout.

### Diagnostics and source-data preparation

Run diagnostics separately with the selected learning rates:

```bash
python scripts/sweep.py --tests 7 --gpus 0,1,2,3 --runs-dir outputs/exp7 \
  --lr-selection outputs/sweep/exp0/lr_selection.json
python scripts/sweep.py --tests 8 --gpus 0,1,2,3 --runs-dir outputs/exp8 \
  --lr-selection outputs/sweep/exp0/lr_selection.json
```

For significance, use `scripts/hpec_pipeline.sh` with `STAGES=errors,sig` and
`ERRDUMP_RUNS` naming the intended comparisons in the selected session.
The aggregate statistical baselines are also evaluated on the fixed split.

For a new experiment, `python scripts/release.py rebuild-data` downloads the
pinned raw archives and rebuilds the lattice. It refuses to overwrite existing
inputs. A rebuild is not automatically byte-identical to the archived training
pair. The older research helper `fetch_census_ports.py` is not the bulk lattice
source; this release uses `scripts/fetch_census_bulk.py` and
`scripts/build_census_lattice.py`.

## Results and verification

- [Result tables](results/) include combined and per-run metrics,
  aggregate and ND results, rolling-origin summaries, inference costs, and
  significance output. [SOURCES.json](results/SOURCES.json) records their
  original Hub paths and SHA-256 checksums.
- The portable suite passed **604 tests, with 1 skipped**. The corrected data
  passed the loader validation and all **15 task-spec checks**.
- The archived GRU reproduced its logged test MSE on the corrected lattice:
  approximately **0.69196 on CPU versus 0.69195 in the training log**, over
  **30,087 × 24 series-month observations**, each with two target channels.
- The [completion audit data](docs/audit/audit.json) documents the
  complete main checkpoint matrix. Remaining qualifications include unstable
  base LR-selection cells, four supplementary provenance gaps, and partial
  training-FLOP totals. Training costs and inference costs must be distinguished.

These checks establish the tested code and artifact consistency; they do not
constitute a fresh full GPU training or inference sweep. See
the instructions and qualifications above for the reproduction scope.

## Repository layout

```text
src/             Models, encoders, data loaders, training, and metrics
config/          Shared protocol, model, and variant configurations
scripts/         Data preparation, training, evaluation, and release commands
tests/           Regression tests and task-spec verification
external/        Upstream model implementations and their licenses
results/         Archived result tables with source and checksum records
artifacts/       Learning-rate selections and diagnostic artifacts
docs/            Machine-readable release validation and audit evidence
SOURCE.json      Source provenance, Hugging Face revisions, and data checksums
```

## Citation and related work

```bibtex
@inproceedings{fan2026multidimensional,
  title     = {Are Multidimensional Models Worth Their Computational Cost in Demand Forecasting?},
  author    = {Fan, Cheng-Jui and Aristov, Nikolay and Dugundji, Elenna R.},
  booktitle = {IEEE High Performance Extreme Computing Conference (HPEC)},
  year      = {2026}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).
This work follows the WCTR 2026
[Multidimensional Demand Forecasting](https://github.com/triadastra/Multidimensional-Demand-Forecasting)
benchmark. Its earlier dataset and results use a different protocol and are not
Census benchmark evidence.

## Declaration on the use of generative AI

As declared in the paper, Claude Code was used for code scaffolding, literature
research, table typesetting, result graphing, and mathematical notation. ChatGPT
Codex was used for proofreading and reference checking. All experiments and
reported results were run and checked by the authors.

This public-facing repository was cleaned using Anthropic Fable 5.1 and OpenAI
Astra 6 to make the code easier for future researchers to run. No experiments or
configurations were altered during this cleanup.

This README was written using GPT 6 Astra.

## License

The benchmark code is released under the [MIT License](LICENSE). Upstream code
under `external/` retains its original licenses. Census source data and dataset
licensing are documented in the
[Hugging Face dataset card](https://huggingface.co/datasets/Celsia/HPEC2026).
