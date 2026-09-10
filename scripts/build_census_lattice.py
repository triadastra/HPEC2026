#!/usr/bin/env python3
"""Build the model-ready lattice from the Census port HS6 raw files.

Config (frozen from the 2026-07 density study, see D:/wctr_data/LATTICE_CONFIG.md):
  axes      : commodity(HS6) x state(port-state) x flow(import/export)
  states    : top-N by combined dense-commodity fill
  commodity : greedy-pruned to hold grid density >= TARGET_DENSITY
  filter    : keep a (comm,state,flow) series only if value_mo is non-zero in
              >=95% of TRAIN months; validation/test availability is ignored
  channels  : 9 = transport-mode decomposition (see below)
  norm      : per-series, per-channel  log1p -> MinMax(fit on TRAIN only, range
              floored at 1.0). log1p tames post-2021 structural breaks; the floor
              stops divide-by-~0 on near-constant (dead-in-train) series.

Why 9 channels (PLAN.md S2.2): value_mo is ALL-modes; ves_swt is vessel-only and
air_swt was previously dropped, so ~35% of weight was NA->0. We now read every
mode field and keep 4 mutually-exclusive modes (air / containerized / breakbulk /
land) separate, plus the two aggregates. Land is value-only (truck/rail/pipeline
are not weighed in this file). Channel order:

  0 agg_value     value_mo (all modes)
  1 air_value     air_val
  2 cnt_value     cnt_val  (containerized vessel)
  3 bulk_value    ves_val - cnt_val  (breakbulk/bulk vessel)
  4 land_value    value_mo - air_val - ves_val  (truck/rail/pipeline residual)
  5 agg_weight    air_swt + ves_swt  (RECOVERED total shipping weight)
  6 air_weight    air_swt
  7 cnt_weight    cnt_swt
  8 bulk_weight   ves_swt - cnt_swt
  (no land_weight -- land modes are not weighed)

Output (does NOT materialize the dense C*S*F*T grid): a packed series panel +
each series' grid coordinate + a validity mask.
  <out>/census_lattice.npz  : panel_norm(N,T,9), panel_raw(N,T,9), series_idx,
                              norm_min/max/range(N,9) [in LOG space], mask,
                              aggregate_raw(T,2) summed before cohort selection
  <out>/census_lattice.json : vocabs, grid shape, splits, channels, summary
"""
import argparse, csv, json, os, zipfile
from collections import defaultdict
import numpy as np

# ---- field layout (1-based cols -> py slices), 15-digit each, monthly card ----
F_VALUE = slice(20, 35)   # value_mo   total value (all modes)
F_AIRV  = slice(35, 50)   # air_val_mo
F_AIRW  = slice(50, 65)   # air_swt_mo
F_VESV  = slice(65, 80)   # ves_val_mo (incl. containerized)
F_VESW  = slice(80, 95)   # ves_swt_mo (incl. containerized)
F_CNTV  = slice(95, 110)  # cnt_val_mo containerized vessel
F_CNTW  = slice(110, 125) # cnt_swt_mo containerized vessel
NRAW = 7                  # accumulate order: value,airv,airw,vesv,vesw,cntv,cntw
_SL = (F_VALUE, F_AIRV, F_AIRW, F_VESV, F_VESW, F_CNTV, F_CNTW)

FLOWS = {"export": ("ex_hs6_m", "PORTHS6XM"), "import": ("im_hs6_m", "PORTHS6MM")}
CHANNELS = ["agg_value", "air_value", "cnt_value", "bulk_value", "land_value",
            "agg_weight", "air_weight", "cnt_weight", "bulk_weight"]
VALUE_CH  = [0, 1, 2, 3, 4]
WEIGHT_CH = [5, 6, 7, 8]
TARGET_CH = [0, 5]        # (agg_value, agg_weight)

def months_list():
    return [f"{y}-{m:02d}" for y in range(2010, 2026) for m in range(1, 13)]

def load_state_map(ref):
    ds = {}
    with open(os.path.join(ref, "scheduleD_dist3.txt"), encoding="latin-1") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and len(row[0]) == 2 and row[0].isdigit():
                st = row[2].rsplit(",", 1)[-1].strip()
                if len(st) == 2 and st.isalpha():
                    ds.setdefault(row[0].encode(), st)
    return ds

def _num(b):
    b = b.strip()
    return int(b) if b.isdigit() else 0

def accumulate(base, ds, level, allow_missing_months=False):
    """Sum the 7 raw mode fields per (comm, state, flow, month) over port+country.

    Presence gate is value_mo>0 (UNCHANGED), so the dense-series set is identical
    to the 2-channel build; the extra fields only add mode detail per cell.
    """
    months = months_list(); T = len(months)
    expected = []
    for flow, (folder, prefix) in FLOWS.items():
        for mstr in months:
            y, m = mstr.split("-")
            expected.append((flow, mstr, os.path.join(
                base, folder, f"{prefix}{y[2:]}{m}.ZIP"
            )))
    missing = [(flow, month, path) for flow, month, path in expected
               if not os.path.exists(path)]
    if missing and not allow_missing_months:
        preview = ", ".join(f"{flow}:{month}" for flow, month, _ in missing[:8])
        if len(missing) > 8:
            preview += f", ... (+{len(missing) - 8} more)"
        raise FileNotFoundError(
            f"missing {len(missing)} expected Census monthly archive(s): {preview}. "
            "A missing source file is unknown data, not observed zero trade. "
            "Re-download it, or pass --allow-missing-months for a diagnostic build."
        )
    raw = defaultdict(lambda: np.zeros((T, NRAW)))
    for flow, (folder, prefix) in FLOWS.items():
        for j, mstr in enumerate(months):
            y, m = mstr.split("-")
            p = os.path.join(base, folder, f"{prefix}{y[2:]}{m}.ZIP")
            if not os.path.exists(p):
                continue
            with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as fh:
                for s in fh:
                    if len(s) < 125:
                        continue
                    v = _num(s[F_VALUE])
                    if v <= 0:                      # same presence gate as before
                        continue
                    st = ds.get(s[10:12])
                    if not st:
                        continue
                    a = raw[(s[0:level].decode(), st, flow)][j]
                    for c, sl in enumerate(_SL):
                        a[c] += _num(s[sl])
    return months, raw, [path for _, _, path in missing]

def _derive_channels(a):
    """(T,7) raw sums -> (T,9) mode channels (order = CHANNELS)."""
    value, airv, airw, vesv, vesw, cntv, cntw = (a[:, k] for k in range(NRAW))
    out = np.stack([
        value,                                   # 0 agg_value
        airv,                                    # 1 air_value
        cntv,                                    # 2 cnt_value
        np.maximum(vesv - cntv, 0.0),            # 3 bulk_value (breakbulk = ves-cnt)
        np.maximum(value - airv - vesv, 0.0),    # 4 land_value (residual)
        airw + vesw,                             # 5 agg_weight (recovered)
        airw,                                    # 6 air_weight
        cntw,                                    # 7 cnt_weight
        np.maximum(vesw - cntw, 0.0),            # 8 bulk_weight
    ], axis=1)
    return out


def _national_aggregate(raw, n_months):
    """Return national value/weight before any benchmark-cohort selection.

    Rolling aggregate tests must not inherit the state/commodity/flow cohort
    selected for the multidimensional benchmark. That cohort is frozen using
    the canonical 2010-2021 training period and is therefore future-selected
    for early rolling folds. Summing the complete raw universe here makes the
    aggregate target independent of every later density/ranking decision.
    """
    aggregate = np.zeros((n_months, 2), dtype=np.float64)
    for values in raw.values():
        channels = _derive_channels(values)
        aggregate[:, 0] += channels[:, TARGET_CH[0]]
        aggregate[:, 1] += channels[:, TARGET_CH[1]]
    return aggregate.astype(np.float32)

def _select_cohort(raw, train_months, n_states=14, target_density=0.80,
                   filter_frac=0.95):
    """Select and freeze the benchmark population from training data only.

    ``raw`` may contain validation/test months, but none of those observations
    participate in density filtering, state ranking, or commodity pruning.
    This prevents future target availability from changing who is evaluated.
    """
    threshold = filter_frac * train_months

    # 1) dense series = training value_mo non-zero in >=95% of TRAIN months.
    dense = {
        k: a for k, a in raw.items()
        if np.count_nonzero(a[:train_months, 0]) >= threshold
    }

    # 2) top-N states by number of dense series carried (both flows)
    sfill = defaultdict(int)
    for (c, st, fl) in dense:
        sfill[st] += 1
    states = [st for st, _ in sorted(sfill.items(), key=lambda x: -x[1])[:n_states]]
    Sset = set(states)
    dense = {k: a for k, a in dense.items() if k[1] in Sset}

    # 3) greedy-prune commodities to hold density >= target
    csupport = defaultdict(int)
    for (c, st, fl) in dense:
        csupport[c] += 1
    ranked = sorted(csupport, key=lambda c: -csupport[c])
    cells_per_comm = len(states) * len(FLOWS)
    keep, cum = [], 0
    for i, c in enumerate(ranked, 1):
        cum += csupport[c]
        if cum >= target_density * i * cells_per_comm:
            keep.append(c)
        else:
            break
    Cset = set(keep)
    series = sorted(k for k in dense if k[0] in Cset)  # (comm, state, flow)
    return dense, states, series


def build(base, ref, out, name="census_lattice", level=6, n_states=14,
          target_density=0.80, filter_frac=0.95,
          train_end="2021-12", val_end="2023-12",
          allow_missing_months=False):
    ds = load_state_map(ref)
    if not ds:
        # Every record is keyed by district -> state through this map, so an
        # unparsed reference silently drops the entire panel and the build goes
        # on to write a zero-series artifact.
        raise ValueError(
            f"no district->state entries parsed from {ref}/scheduleD_dist3.txt. "
            "Expected CSV rows whose first field is a 2-digit district code and "
            "whose third field ends in a 2-letter state, with the city/state "
            "field quoted so the comma inside it survives: "
            "01,000,\"Boston, MA\". Check that this is the Schedule D district "
            "reference and that the quoting survived export."
        )
    months, raw, missing_archives = accumulate(
        base, ds, level, allow_missing_months=allow_missing_months
    )
    T = len(months)
    tr_end = months.index(train_end) + 1
    va_end = months.index(val_end) + 1

    # Build the national target BEFORE density filtering, state ranking, or
    # commodity pruning. Test 1 / 1.1 use this array exclusively.
    aggregate_raw = _national_aggregate(raw, T)

    # Freeze the complete state/commodity/flow cohort before looking at any
    # validation or test observation. Full-history arrays are used only after
    # membership has been decided.
    dense, states, series = _select_cohort(
        raw, tr_end, n_states=n_states, target_density=target_density,
        filter_frac=filter_frac,
    )
    if not series:
        # Without this the density divide below is 0/0, the build prints
        # "0 series | density nan%", exits 0, and writes an empty .npz that only
        # fails later, in training, with an unrelated-looking error.
        raise ValueError(
            f"cohort selection retained 0 series from {len(raw):,} raw "
            f"(commodity, state, flow) keys. Nothing was written. Likely causes: "
            f"the monthly archives under {base} did not parse; no series is "
            f"non-zero in >={filter_frac:.0%} of the {tr_end} training months; "
            f"or --density {target_density} pruned every commodity."
        )

    # 4) vocabs + grid coords
    comm_v = sorted({k[0] for k in series}); state_v = sorted({k[1] for k in series})
    flow_v = ["export", "import"]
    ci = {c: i for i, c in enumerate(comm_v)}; si = {s: i for i, s in enumerate(state_v)}
    fi = {f: i for i, f in enumerate(flow_v)}
    C, S, F = len(comm_v), len(state_v), len(flow_v)

    n = len(series); K = len(CHANNELS)
    panel = np.zeros((n, T, K), dtype=np.float32)
    idx = np.zeros((n, 3), dtype=np.int32)
    mask = np.zeros((C, S, F), dtype=bool)
    for r, k in enumerate(series):
        panel[r] = _derive_channels(dense[k])
        idx[r] = (ci[k[0]], si[k[1]], fi[k[2]])
        mask[ci[k[0]], si[k[1]], fi[k[2]]] = True

    # 5) chronological splits + per-series/per-channel  log1p -> train MinMax(floor 1)
    logp = np.log1p(np.clip(panel, 0.0, None))            # (n,T,K)
    tr = logp[:, :tr_end, :]
    mn = tr.min(axis=1); mx = tr.max(axis=1)             # (n,K) in LOG space
    rng = np.maximum(mx - mn, 1.0)                        # floor 1.0
    panel_norm = (logp - mn[:, None, :]) / rng[:, None, :]

    density = mask.sum() / (C * S * F)
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(out, f"{name}.npz")
    np.savez_compressed(npz, panel_norm=panel_norm, panel_raw=panel,
                        aggregate_raw=aggregate_raw,
                        series_idx=idx, norm_min=mn, norm_max=mx,
                        norm_range=rng, mask=mask)
    meta = dict(level=f"HS{level}", n_series=n, grid_shape=[C, S, F],
                density=round(float(density), 4), channels=CHANNELS,
                value_channels=VALUE_CH, weight_channels=WEIGHT_CH,
                target_channels=TARGET_CH,
                aggregate_source=dict(
                    scope="all_raw_series_before_cohort_selection",
                    channels=["agg_value", "agg_weight"],
                    n_source_series=len(raw),
                ),
                norm="log1p->trainMinMax(floor=1.0); norm_min/max/norm_range "
                     "in LOG space; invert: raw=expm1(norm*norm_range+norm_min)",
                cohort_selection=dict(
                    months=[months[0], train_end], n_months=tr_end,
                    filter_nonzero_fraction=filter_frac,
                    state_and_commodity_ranking="training_months_only",
                ),
                months=[months[0], months[-1]], n_months=T,
                source_missing_archives=missing_archives,
                splits=dict(train=[0, tr_end], val=[tr_end, va_end], test=[va_end, T]),
                states=state_v, commodities=comm_v, flows=flow_v,
                n_commodities=C,
                per_flow={fl: sum(1 for k in series if k[2] == fl) for fl in flow_v})
    json.dump(meta, open(os.path.join(out, f"{name}.json"), "w"), indent=2)
    print(f"[built] {n:,} series | grid {C}x{S}x{F} | density {100*density:.1f}% | "
          f"panel {panel_norm.shape} | {K} channels")
    print(f"        channels: {CHANNELS}")
    print(f"        per-flow: {meta['per_flow']}  ->  {npz}")
    return meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/census_port/raw")
    ap.add_argument("--ref",  default="data/census_port/reference")
    ap.add_argument("--out",  default="data/census_port/processed")
    # Must match what config/census.yaml and sweep.DEFAULT_NPZ read. The old
    # default wrote census_lattice.npz, which nothing in the pipeline opens --
    # a mismatch that only surfaces after the multi-GB download and the build
    # have both already been paid for.
    ap.add_argument("--name", default="census_lattice_9ch",
                    help="output basename (default: %(default)s, which is what "
                         "config/census.yaml and sweep.DEFAULT_NPZ open)")
    ap.add_argument("--level", type=int, default=6, help="commodity HS digits (2/4/6)")
    ap.add_argument("--n-states", type=int, default=14)
    ap.add_argument("--density", type=float, default=0.80)
    ap.add_argument("--allow-missing-months", action="store_true",
                    help="diagnostic only: zero-fill months whose entire source archive is absent")
    a = ap.parse_args()
    build(a.base, a.ref, a.out, name=a.name, level=a.level, n_states=a.n_states,
          target_density=a.density, allow_missing_months=a.allow_missing_months)
