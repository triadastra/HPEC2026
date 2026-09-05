"""Tests for the Mamba-ND per-layer scan schedule. No torch, no GPU:

    python tests/test_scan_schedule.py

Covers the phase-locking bug the ``"legacy"`` schedule has, and the coverage
trade-off between ``"cyclic"`` and ``"paired"`` at a given depth.
"""
import sys
sys.path.insert(0, ".")

import warnings

from src.models.scan_schedule import build_scan_schedule

# Scan axes per variant, as MambaND builds them: Time (1) + categorical axes.
SCAN_2D = [1, 2]           # Time, State
SCAN_3D = [1, 2, 3]        # Time, State, Commodity
SCAN_4D = [1, 2, 3, 4]     # Time, State, Commodity, Flow


def directions(scan_positions, n_layers, schedule, bidirectional=True):
    """axis -> set of directions it is scanned in."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos, rev = build_scan_schedule(
            scan_positions, n_layers, schedule=schedule, bidirectional=bidirectional
        )
    out = {}
    for p, r in zip(pos, rev):
        out.setdefault(p, set()).add(r)
    return out


def check_legacy_phase_locks_on_even_axis_counts():
    """The original schedule advances the axis with period n and flips
    direction with period 2. When n is even the two lock, and NO depth
    rescues it -- this is why 2d and 4d were effectively unidirectional."""
    for scan in (SCAN_2D, SCAN_4D):
        for n_layers in (4, 8, 16, 64):
            d = directions(scan, n_layers, "legacy")
            assert all(len(v) == 1 for v in d.values()), (
                f"legacy unexpectedly bidirectional for n={len(scan)}, "
                f"n_layers={n_layers}"
            )
    # n=3 is coprime with 2, so it escapes -- which is exactly why the bug
    # hid: 3d is the variant one would eyeball.
    d = directions(SCAN_3D, 8, "legacy")
    assert all(len(v) == 2 for v in d.values())
    print("legacy phase-locking reproduced (2d/4d pinned at every depth)")


def check_cyclic_covers_every_axis():
    """cyclic reaches all axes within n layers, at any depth."""
    for scan in (SCAN_2D, SCAN_3D, SCAN_4D):
        d = directions(scan, len(scan), "cyclic")
        assert set(d) == set(scan), f"cyclic missed an axis: {d}"
    print("cyclic covers every axis within n layers")


def check_cyclic_is_bidirectional_given_enough_layers():
    for scan in (SCAN_2D, SCAN_3D, SCAN_4D):
        d = directions(scan, 2 * len(scan), "cyclic")
        assert all(len(v) == 2 for v in d.values()), f"not bidirectional: {d}"
    print("cyclic is fully bidirectional at 2n layers")


def check_cyclic_degrades_to_forward_only():
    """Below the budget cyclic stays forward rather than pinning axes to
    arbitrary directions."""
    d = directions(SCAN_4D, 4, "cyclic")
    assert all(v == {False} for v in d.values()), d
    print("cyclic degrades to forward-only rather than arbitrary directions")


def check_paired_matches_upstream():
    """Upstream Mamba-ND: z = i // 2, d = z % len(orders), reverse = i % 2."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos, rev = build_scan_schedule(SCAN_3D, 6, schedule="paired")
    assert pos == [1, 1, 2, 2, 3, 3], pos
    assert rev == [False, True, False, True, False, True], rev
    print("paired reproduces the upstream Mamba-ND schedule")


def check_paired_trades_coverage_for_depth():
    """At shallow depth paired never reaches the later axes. This is the
    reason cyclic is the default here: n_layers is 4 in every config."""
    d = directions(SCAN_4D, 4, "paired")
    assert set(d) == {1, 2}, d
    assert 3 not in d and 4 not in d
    print("paired at depth 4 reaches only 2 of 4 axes (as expected)")


def check_warnings_fire():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_scan_schedule(SCAN_4D, 4, schedule="paired")
        assert any("never scans" in str(x.message) for x in w), [str(x.message) for x in w]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_scan_schedule(SCAN_4D, 4, schedule="cyclic")
        assert any("one direction only" in str(x.message) for x in w), [
            str(x.message) for x in w
        ]
    print("warnings fire when the layer budget is insufficient")


def check_unknown_schedule_rejected():
    try:
        build_scan_schedule(SCAN_2D, 4, schedule="nope")
    except ValueError as e:
        assert "nope" in str(e)
        print("unknown schedule rejected")
    else:
        raise AssertionError("expected ValueError")


def check_lengths():
    for sched in ("legacy", "cyclic", "paired"):
        for n_layers in (1, 3, 4, 7, 12):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pos, rev = build_scan_schedule(SCAN_3D, n_layers, schedule=sched)
            assert len(pos) == len(rev) == n_layers
    print("schedules have exactly n_layers entries")


def main():
    check_legacy_phase_locks_on_even_axis_counts()
    check_cyclic_covers_every_axis()
    check_cyclic_is_bidirectional_given_enough_layers()
    check_cyclic_degrades_to_forward_only()
    check_paired_matches_upstream()
    check_paired_trades_coverage_for_depth()
    check_warnings_fire()
    check_unknown_schedule_rejected()
    check_lengths()
    print("\nScan schedule tests passed.")


# pytest entry points
test_legacy_phase_locks = check_legacy_phase_locks_on_even_axis_counts
test_cyclic_covers = check_cyclic_covers_every_axis
test_cyclic_bidirectional = check_cyclic_is_bidirectional_given_enough_layers
test_cyclic_degrades = check_cyclic_degrades_to_forward_only
test_paired_upstream = check_paired_matches_upstream
test_paired_coverage = check_paired_trades_coverage_for_depth
test_warnings = check_warnings_fire
test_unknown_schedule = check_unknown_schedule_rejected
test_lengths = check_lengths


if __name__ == "__main__":
    main()
