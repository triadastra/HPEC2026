# Release documentation

Start with `../REPRODUCING.md`. `audit/` preserves the saved completion and
statistical audits of the pinned model archive. References there to temporary
local downloads describe the environment at audit time, not files shipped here.
Machine-local audit runners and VM deployment scripts are not needed to consume
this release. General merge/publication helpers are retained under `tools/`;
`coordinate.py` now takes host/ports from environment configuration rather than
author-specific endpoints.

`PLAN.md`, `RETRAIN.md`, `catalog.md`, and `future_work.md` at the repository root
are inherited research protocol/history documents. Their historical status
statements are superseded by the release README and saved completion audit.
Historical Git hashes refer to the source repository recorded in `SOURCE.json`.

## Public release contents

- `src/`, `config/`: forecasting models, features, encoders, training and metrics.
- `scripts/`: Census bulk input preparation, sweep, LR selection, evaluation,
  cost measurement, significance, baselines and release entry points.
- `tests/`: portable regression coverage and a checksummed retired-emitter fixture.
- `external/`: S4, Mamba, Mamba-ND and pinned upstream CaFA; original licenses remain.
- `results/`: unchanged paper-facing Hub tables with source paths and hashes.
- `artifacts/`: inherited LR selections and diagnostic artifacts.

No original research Git history, VM credentials, raw datasets, checkpoint
binaries, or manuscript PDF is committed. The paper is identified by title and
SHA-256 in `SOURCE.json`; the supplied document was treated as reference material.
