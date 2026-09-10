"""Backward-compatible import for the former ``authors_cafa_*`` name.

The geometry-free operator is now named Factorized Attention (``fa_*``).
"""

from .fa import FactorizedAttention


AuthorsFactorizedAxialCA = FactorizedAttention

__all__ = ["AuthorsFactorizedAxialCA", "FactorizedAttention"]
