# Reproducing HPEC2026

This is the standalone code release for **Are Multidimensional Models Worth Their
Computational Cost in Demand Forecasting?** The original research checkout is
unchanged. `SOURCE.json` records the export commit, upstream CaFA commit, paper
checksum, and separate pinned model and dataset Hub revisions.

## Install and test

Use Python 3.11. The CPU environment uses PyTorch 2.9.1; the manuscript describes
training on PyTorch 2.12 and local CPU inference on 2.9. These are distinct
runtime records. This release's CPU checks do not certify the original CUDA runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt
python scripts/release.py test
python scripts/release.py results
python scripts/release.py checkpoint --run gru_embeddings_1d_s947
python scripts/release.py smoke
python scripts/release.py plan
```

The test suite includes training-step, data-loader, metric, orchestration,
provenance, ASA, CoFA and upstream FA regression tests using small fixtures.
`smoke` strictly loads the real archived GRU and runs a synthetic `(2,36,35)`
input through it. It checks compatibility and finite output, not paper accuracy.
`plan` writes a 477-run roster without scheduling work; its example step budget
uses 30,087 series. The actual sweep derives its budget from the loaded lattice.

The GitHub CPU workflow runs the same tests. Mamba requires Linux/NVIDIA CUDA
and the additional dependencies in `requirements-cuda.txt`. A successful CPU
suite does not establish that CUDA kernels execute correctly.

## Hugging Face: two repositories with the same name

| Purpose | Repository type | Pinned revision |
|---|---|---|
| Checkpoints and saved results | [model](https://huggingface.co/Celsia/HPEC2026) | `51e064a2f8e1af1ff867794b5ce8aeeade4692cd` |
| Input data and raw archives | [dataset](https://huggingface.co/datasets/Celsia/HPEC2026) | `d3f61546516002a44f79962507422f47c4f80263` |

`checkpoint` accepts a run path relative to the model archive, including a
session prefix where necessary. It checks membership in that session's manifest,
the completion fingerprint, checkpoint SHA-256, and fresh-session record where
required. It loads tensors with `weights_only=True` on CPU. Downloads live under
`outputs/hf/` and are ignored by Git. No upload or public visibility change occurs.

### Input version — resolved 2026-09-06

The dataset revision pinned above carries the corrected processed lattice:
**30,087 series over a 1,343 × 14 × 2 grid**, with the cohort ranked and
filtered on the 144 training months only (sidecar
`cohort_selection.state_and_commodity_ranking = training_months_only`). It is
the exact file instance the archived sweep trained on, copied from the
training machine rather than rebuilt; the archived `gru_embeddings_1d_s947`
checkpoint reproduces its logged test MSE (0.69195) on it, and its commodity
embedding has the matching 1,343 rows. `SOURCE.json` records both checksums.

The earlier 28,292-series / 1,263-commodity build selected its cohort on all
192 months, a leak into the 2022–2025 evaluation period. It is preserved
unchanged under `processed/legacy_precorrection/` (and at revision
`47dd2a5f6be759562fa9c1e257bca50854e0fd6f`) for inspection only; the loader
refuses it because its sidecar has no cohort-selection contract.

```bash
python scripts/release.py data
python scripts/release.py data-check
python tests/verify_task_spec.py
```

`data` downloads and checks the small sidecar first, then the lattice, verifies
both against the pinned SHA-256 values, and installs the pair under
`data/census_port/processed/`. It refuses to overwrite an existing pair.

One manuscript-side inconsistency remains: the paper text combines 30,087
series with 1,263 commodities. The archived runs and this dataset use 1,343.

For a **new experiment**, the archived raw Census files can be downloaded and
processed with the corrected builder:

```bash
python scripts/release.py rebuild-data --npz data/census_port/processed/census_lattice_9ch.npz
python scripts/release.py data-check
```

This fetches all 384 monthly raw archives (several GB), uses the matching archived
Schedule D reference, and refuses to overwrite an existing input pair. It has not
been run end-to-end as part of preparing this release. A rebuilt lattice must
not be claimed byte-identical to the missing training input without evidence.

## Train and evaluate a new experiment

On Linux with an NVIDIA GPU:

```bash
pip install -r requirements-cuda.txt
python scripts/check_environment.py --require-mamba
GPUS=0,1,2,3 RESET_MANIFEST=1 SESSION_ID=public-reproduction \
  python scripts/release.py train
```

This validates the data, checks the full model environment, runs Exp 0 LR search,
trains Tests 1/1.1/2/3/4/6, evaluates the declared complete matrix, measures model
cost, and computes XGBoost aggregate baselines. The full run is substantial;
none of it is started by installation, `test`, `results`, `checkpoint`, or `plan`.
Resume with the same session and omit `RESET_MANIFEST=1`.

The main pipeline retains the existing strict provenance checks. Archived
manifests include source/config/data fingerprints, so moving checkpoints into a
new checkout is not automatically an official replay. Do not rewrite their
fingerprints or use partial evaluation to present an official leaderboard.

Exp 7 (CoFA, `fa_local_*`) and Exp 8 (authors' softmax, `fa_sm_*`) are separate
27-run diagnostics. Use distinct directories and the selected LR files:

```bash
python scripts/sweep.py --tests 7 --gpus 0,1,2,3 --runs-dir outputs/exp7 \
  --lr-selection outputs/sweep/exp0/lr_selection.json
python scripts/sweep.py --tests 8 --gpus 0,1,2,3 --runs-dir outputs/exp8 \
  --lr-selection outputs/sweep/exp0/lr_selection.json
```

For significance after producing matching error dumps, use the existing pipeline's
`errors,sig` stages with `ERRDUMP_RUNS` specifying the intended comparisons.
The saved paper-facing tables are in `results/`; their exact Hub paths and hashes
are recorded in `results/SOURCES.json`.

## Evidence and release qualifications

- The saved audit validates the 477-run main matrix plus base LR (195), FA LR
  (25), Exp 7 (27), and Exp 8 (27) sessions. The current Exp 0 generator plans
  240 runs; that is not the count of archived LR runs.
- The base LR selection was marked unofficial; several tuning cells were unstable.
- Four supplementary error-dump entries have incomplete declaration/completion
  pairing. Their checkpoint hashes match valid main checkpoints, but the full
  supplementary provenance chain is not established.
- Some archived training FLOP totals are partial/lower bounds. They must not be
  confused with separately measured inference FLOPs.
- See `docs/audit/` for the saved completion and statistical sensitivity audits.
  Those audits are snapshot evidence, not fresh training or full inference here.

Public release should carry these qualifications and correct the manuscript's
commodity count to 1,343. No GitHub repository has been created or published by
this preparation step.
