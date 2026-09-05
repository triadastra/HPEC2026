# Local release validation

- Python 3.11 / PyTorch 2.9.1 on macOS CPU.
- Full portable suite: **604 passed, 1 skipped**, 32 warnings, 193.47 seconds.
- Pinned Hugging Face GRU: manifest/completion/hash checks passed; all 23 tensor
  entries loaded, with 531,856 parameters/buffer elements. Strict model loading
  and a synthetic forward produced finite output of shape `(2, 2)`.
- Eight staged result tables match the SHA-256 checksums recorded in
  `results/SOURCES.json`, including their original line endings.
- The planning command generated 477 main runs without starting training.
- The pinned dataset sidecar was downloaded and rejected as intended: 28,292
  series, rather than the archived main matrix's 30,087.
- Dependency check: all 151 installed packages compatible.
- Original research checkout was left unchanged; no remote publication occurred.

The GPU/Mamba runtime, full training, raw-data rebuild and real-data evaluation
were not run. `docs/verification.json` contains compact machine-readable evidence.
