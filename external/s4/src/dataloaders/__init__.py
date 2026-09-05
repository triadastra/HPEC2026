# Upstream default + `tx` (registers the `trade_base` dataset). `ts` and `et`
# carry the trade datasets used here (s4nd_agg4d/5d/6d, agg_flat, agg). The s4
# framework needs torchaudio (pulled in by base.py/audio.py); it's present on
# the training VM. S4 runs on the VM (also needs the CUDA Cauchy/Vandermonde
# kernel), so local import on machines without torchaudio is expected to fail.
# Trade-forecasting subset of the upstream defaults. ts/et/tx register the
# trade datasets (s4nd_agg4d/5d/6d, agg_flat, agg, trade_base). Dropped
# audio/vision/lm/lra/synthetic to avoid optional heavy deps (torchaudio etc.)
# not needed here. basic is kept for base image-resolution dataset classes.
from . import et, ts, tx, trade_unified, trade_grid, trade_features  # noqa: F401
from .base import SequenceDataset
