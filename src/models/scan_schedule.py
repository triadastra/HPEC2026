"""Per-layer scan schedules for the Mamba-ND grid model.

Kept in its own module, free of the ``mamba_ssm`` import, so the schedule
logic can be imported and unit-tested on a CPU-only box where
``src/models/mamba_nd.py`` cannot even be imported.

Two periods interact in an N-D scan and getting them wrong is silent. The
axis ordering advances with one period; the direction flips with another. If
the two share a factor they phase-lock, and every axis is pinned to a single
direction no matter how deep the stack gets.

That is what the original schedule did::

    layer_pos     = scan_positions[i % n]     # period n
    layer_reverse = bool(i % 2)               # period 2

For ``grid_2d`` (n=2) and ``grid_4d`` (n=4) the two
periods lock, so Time is always forward, State always reverse, and so on --
at *any* depth, not just shallow ones. ``grid_3d`` (n=3) escapes
only because 3 and 2 are coprime. That schedule is retained as ``"legacy"``
solely to reproduce runs recorded before this module existed.
"""

import warnings

_AXIS_NAME = {1: "Time", 2: "State", 3: "Commodity", 4: "Flow"}

# Variant tag -> categorical grid axes scanned (Time is always scanned).
# Tensor axis positions in the dense grid (B, T, S, C, Fl, H): T=1, S=2, C=3, Fl=4.
#
# Lives here, not in mamba_nd.py, for the same reason the schedule does: that
# module cannot be imported without Triton, and this is pure data. The tag is
# ``grid_*d`` because these models contain NO attention -- the old
# ``cross_attention_*d`` spelling named a mechanism they do not have. Legacy
# spellings resolve to the same axes.
VARIANT_CAT_AXES = {
    "grid_2d": (2,),                       # State
    "grid_3d": (2, 3),                     # State, Commodity
    "grid_4d": (2, 3, 4),                  # State, Commodity, Flow
}
VARIANT_CAT_AXES.update({
    f"cross_attention_{d}d": VARIANT_CAT_AXES[f"grid_{d}d"] for d in (2, 3, 4)
})



def build_scan_schedule(scan_positions, n_layers, schedule="cyclic",
                        bidirectional=True, warn=True):
    """Assign a (scan axis, direction) to each layer.

    Schedules:

    ``"paired"``
        The official Mamba-ND scheme (``z = i // 2; d = z % len(orders)`` with
        ``reverse = i % 2``): each axis takes two consecutive layers, once
        forward and once backward. Correct for any axis count, but it needs
        ``2 * len(scan_positions)`` layers before the schedule repeats, so at
        shallow depth it never reaches the later axes at all. Upstream can
        afford this because it runs ViT-depth stacks.

    ``"cyclic"`` (default)
        One axis per layer, direction flipped after each *full cycle*. Reaches
        every axis within ``len(scan_positions)`` layers and degrades
        gracefully when there is no budget for bidirectionality -- it simply
        stays forward rather than pinning axes to arbitrary directions.

    ``"legacy"``
        The original, phase-locking schedule. See the module docstring. Not
        recommended for new runs.

    Returns ``(layer_pos, layer_reverse)``, both lists of length ``n_layers``.
    """
    n = len(scan_positions)
    if schedule == "cyclic":
        pos = [scan_positions[i % n] for i in range(n_layers)]
        rev = [bool(bidirectional and (i // n) % 2 == 1) for i in range(n_layers)]
    elif schedule == "paired":
        directions = (False, True) if bidirectional else (False,)
        template = [(p, r) for p in scan_positions for r in directions]
        chosen = [template[i % len(template)] for i in range(n_layers)]
        pos = [p for p, _ in chosen]
        rev = [r for _, r in chosen]
    elif schedule == "legacy":
        pos = [scan_positions[i % n] for i in range(n_layers)]
        rev = [bool(i % 2) for i in range(n_layers)]
    else:
        raise ValueError(
            f"unknown schedule {schedule!r}; expected 'cyclic', 'paired', or 'legacy'"
        )

    if warn:
        seen = {}
        for p, r in zip(pos, rev):
            seen.setdefault(p, set()).add(r)
        missing = [_AXIS_NAME.get(p, p) for p in scan_positions if p not in seen]
        if missing:
            warnings.warn(
                f"schedule {schedule!r} never scans {missing} in {n_layers} layers; "
                f"raise n_layers to at least "
                f"{n * (2 if schedule == 'paired' and bidirectional else 1)}",
                RuntimeWarning, stacklevel=2,
            )
        pinned = [_AXIS_NAME.get(p, p) for p in scan_positions
                  if p in seen and len(seen[p]) < 2]
        if bidirectional and pinned:
            warnings.warn(
                f"schedule {schedule!r} scans {pinned} in one direction only; "
                f"bidirectional coverage of {n} axes needs about {2 * n} layers",
                RuntimeWarning, stacklevel=2,
            )
    return pos, rev
