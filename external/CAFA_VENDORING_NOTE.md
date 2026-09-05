# Authors' CaFA vendoring note

- Upstream: https://github.com/BaratiLab/CaFA
- Pinned submodule path: `external/cafa-authors`
- Pinned commit at integration: `ff88ac033189b4df8bb4be3041a03c2f8d7e5530`
- License: MIT; the upstream `LICENSE` remains inside the submodule.
- Paper: Zijie Li, Anthony Zhou, Saurabh Patil, and Amir Barati Farimani,
  “CaFA: Global Weather Forecasting with Factorized Attention on Sphere,”
  arXiv:2405.07395 (2024).

The submodule is intentionally unchanged. `src/models/fa.py` implements
geometry-free Factorized Attention by importing upstream `PoolingReducer`,
`LowRankKernel`, `MLP`, and `GroupNorm`. It maps those components to the
benchmark's categorical sparse lattice and must not be described as the
authors' complete weather model.

The standalone HPEC2026 release includes these exact upstream files directly.
