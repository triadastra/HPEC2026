# Local release validation

- Python 3.11 / PyTorch 2.9.1 on macOS CPU.
- Full portable suite: **604 passed, 1 skipped**, 32 warnings, 193.47 seconds.
- Pinned Hugging Face GRU: manifest/completion/hash checks passed; all 23 tensor
  entries loaded, with 531,856 parameters/buffer elements. Strict model loading
  and a synthetic forward produced finite output of shape `(2, 2)`.
- Eight staged result tables match the SHA-256 checksums recorded in
  `results/SOURCES.json`, including their original line endings.
- The planning command generated 477 main runs without starting training.
- Corrected input pair rechecked on 2026-09-09: both SHA-256 hashes match
  `SOURCE.json`; the 30,087-series lattice passes loader validation and all
  15 task-spec checks. The earlier incompatible dataset has been superseded.
- Dependency check: all 151 installed packages compatible.
- Original research checkout was left unchanged; no remote publication occurred.

The GPU/Mamba runtime, full training, raw-data rebuild and full real-data
inference sweep were not run. The single archived GRU real-data reproduction
is recorded separately in `REPRODUCING.md`. `docs/verification.json` contains compact machine-readable evidence.
