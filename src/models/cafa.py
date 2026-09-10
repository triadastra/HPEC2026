"""LEGACY (pre-`5eeb6f9`, 2026-08-24) — superseded by ``fa_local.py``.

Kept as a compatibility shim so that code, notebooks and recorded manifest
commands written before the CaFA->FA rename still import. Do not use it in new
code; import ``LocalFactorizedAttention`` from ``src.models.fa_local`` instead.

Why the name changed: "CaFA" is the authors' WEATHER MODEL -- ForeCasting with
Factorized Attention (Li et al. 2024, arXiv:2405.07395). The operator inside it
is FA. This module never ran their code, so the local reimplementation of that
operator is now ``fa_local_*`` (``src/models/fa_local.py``), and the
unqualified ``fa_*`` name belongs to ``src/models/fa.py``, which runs the
authors' own components from the pinned ``BaratiLab/CaFA`` submodule.

Legacy *variant* spellings ("cafa_2d" and friends) are handled separately, by
``canonical_variant`` in ``src/models/combo_attention.py``; this module covers
only the legacy import path. See ``milestones.md`` §2 for the full rename.
"""

from .fa_local import LocalFactorizedAttention

#: Superseded name for :class:`~src.models.fa_local.LocalFactorizedAttention`.
FactorizedAxialCA = LocalFactorizedAttention

#: Provenance tag: the last revision under which this module's own name was
#: current, and the module that replaced it.
__legacy_until__ = "5eeb6f9"          # Rename: CaFA->FA (2026-08-24)
__superseded_by__ = "src.models.fa_local"

__all__ = ["FactorizedAxialCA", "LocalFactorizedAttention"]
