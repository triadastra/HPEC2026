from .base import (COMBO_GRID_STEMS, COMBO_GRID_VARIANTS, BaseModel,
                   ModelFactory, create_model)

# Importing each model file registers it with ModelFactory. Order doesn't
# matter; we just need the side-effect imports so ``create_model("lstm", ...)``
# can find the class.
from . import lstm  # noqa: F401
from . import gru  # noqa: F401
from . import transformer  # noqa: F401
# Optional models with heavy/extra deps (HF transformers / xgboost / lightgbm) —
# imported defensively so a lean env (the torch-2.8 `t28` venv, which only needs
# gru/lstm/transformer/s4) can still import src.models; base/2.13 has them all.
for _opt in ("gpt", "xgboost", "lightgbm"):
    try:
        __import__(f"{__name__}.{_opt}")
    except Exception as _opt_err:  # pragma: no cover
        import warnings as _warnings
        _warnings.warn(f"{_opt} unavailable ({_opt_err!r}); '{_opt}' not registered.")
# Real Mamba-2 (vendored state-spaces/mamba under external/mamba/). It needs
# triton/GPU, so import it DEFENSIVELY: on CPU-only machines / fresh clones the
# import fails and 'mamba' is simply not registered, while `import src.models`
# still works. The old ssm_models LSTM "mamba" stub is intentionally NOT
# imported, so a missing real Mamba errors loudly rather than silently
# training an LSTM under the 'mamba' name.
try:
    from . import mamba  # noqa: F401  (registers 'mamba2' + 'mamba' alias)
except Exception as _mamba_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"Mamba-2 unavailable ({_mamba_err!r}); 'mamba2' model not registered.")
# Real Mamba-3 (ICLR 2026: exp-trapezoidal discretization + complex/RoPE state +
# optional MIMO), vendored under external/mamba/. Same triton/GPU requirement as
# Mamba-2; import defensively so CPU-only clones still import src.models.
try:
    from . import mamba3  # noqa: F401
except Exception as _mamba3_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"Mamba-3 unavailable ({_mamba3_err!r}); 'mamba3' model not registered.")
# Real S4: wraps the GENUINE S4Block (FFTConv + DPLR SSM kernel) from the vendored
# state-spaces/s4 (external/s4/models/s4/s4.py) -- the SAME implementation as the
# upstream per-combo Exp 2 runs, exposed through the github factory so the real S4
# can also run the aggregate (Exp 1) path the Hydra framework doesn't cover. NOT a
# stub: the old LSTM-placeholder s4 stays removed, so a missing real S4 fails loudly
# rather than silently registering a fake. Imported defensively (needs the s4 deps).
try:
    from . import s4  # noqa: F401
except Exception as _s4_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"Real S4 unavailable ({_s4_err!r}); 's4' model not registered.")
# Genuine S4ND (separable per-axis DPLR kernels = outer-product N-D LTI conv)
# over the masked lattice. Shares the standalone s4 file, so it fails exactly
# where 's4' fails (e.g. base/2.13 torchvision circularity) — defensive too.
try:
    from . import s4nd  # noqa: F401
except Exception as _s4nd_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"S4ND unavailable ({_s4nd_err!r}); 's4nd' model not registered.")
# Mamba-ND grid model (external/Mamba-ND method over the Mamba-2 SSD mixer).
# Needs triton/GPU like 'mamba'; import defensively so CPU-only imports still work.
try:
    from . import mamba_nd  # noqa: F401
except Exception as _mnd_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"Mamba-ND unavailable ({_mnd_err!r}); 'mamba_nd' model not registered.")
from . import baselines  # noqa: F401  # seasonal_naive, random_walk, moving_average

__all__ = ["COMBO_GRID_STEMS", "COMBO_GRID_VARIANTS", "BaseModel",
           "ModelFactory", "create_model"]
