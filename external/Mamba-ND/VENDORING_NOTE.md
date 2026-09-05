# Vendored: Mamba-ND

Source: https://github.com/jacklishufan/Mamba-ND (ECCV 2024), shallow-cloned
2026-06-21. Upstream had **no LICENSE file** at clone time — provenance/usage
rights unconfirmed; resolve license with the authors before any redistribution
or publication of this repo.

Only the alternating per-axis scan method (`Block` in
`video_*/src/mamba.py`, `video_pretraining/models/mamband.py`) is reused.
The image/video backbones depend on mmcv/mmengine and are NOT used here.

For the trade grid we run the Mamba-ND scan around the **Mamba-2 (SSD)**
mixer from `external/mamba`, since the box has only the Triton SSD path
(Mamba-1's `selective_scan_cuda` kernel is unavailable/stubbed).
See `src/models/mamba_nd.py`.
