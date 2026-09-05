"""Loader for the pre-built Census port HS6 lattice (`census_lattice*.npz`).

Drop-in replacement for ``TradeDataPipeline`` when training on the HS6 census
lattice built by ``scripts/build_census_lattice.py``. Emits batches in the EXACT
format the existing models expect and **reuses** ``collate_fn`` /
``combo_collate_fn`` from ``dataloader.py``, so every flat and combo/axial model
runs unchanged.

The panel now carries the **9-channel transport-mode decomposition** (PLAN.md
§2.2) — agg + air/containerized/breakbulk/land value, and agg + air/cont/breakbulk
weight (land is value-only). All are already `log1p` + train-MinMax normalized in
the .npz. Channel semantics come from the sibling `<name>.json`.

Feature vector per timestep (default = **35 dims**):
    [9 mode channels] + [agg_value lag_1..12] + [agg_weight lag_1..12]
    + [sin_month, cos_month]
- ``lag_mode="agg"`` (default) lags only the two aggregate channels (the 36-step
  window already carries the per-mode history) → 9 + 24 + 2 = 35.
- ``lag_mode="all"`` lags all 9 channels → 9 + 108 + 2 = 119 (heavier; combo
  per-sample ≈ 484 MB vs 143 MB).

Targets: ``(agg_value, agg_weight)`` = channels ``meta.target_channels`` — the two
aggregates, matching the models' 2-D output head (per-mode targets would need a
wider head; out of scope here).

Aggregate Tests 1/1.1 use ``aggregate_raw(T,2)``, which the builder sums across
the complete raw universe before multidimensional cohort selection. Rolling
folds refit their transform on training months and truncate the in-memory panel
at the fold's exclusive test boundary.

Two differences from ``TradeDataPipeline``, both forced by scale:
  1. Source is the already-normalized numpy panel in the .npz — no CSV/pandas.
  2. Windows are sliced **on the fly** in ``__getitem__`` (materializing every
     series×window up front is ~tens of GB).

The axial (combo) models need ``combo_coords (G,3)`` in **(State, Commodity,
Flow)** order and ``lattice_dims = (S, C, Flow)`` — provided as attributes,
mirroring ``TradeDataPipeline.build_combo_maps``.

NOTE: written against the model/loader interfaces; verify with a 1-batch forward
on the GPU box (no torch on the dev machine).
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .dataloader import collate_fn, combo_collate_fn  # identical batch format


@dataclass
class CensusConfig:
    npz: str = "data/census_port/processed/census_lattice_9ch.npz"
    input_len: int = 36
    lag_count: int = 12
    lag_mode: str = "agg"          # "agg" (lag target channels) | "all" (lag all)
    # Split by TARGET-month index (inclusive-exclusive), matching the builder's
    # meta.splits default (train 0-143 / val 144-167 / test 168-191).
    train_end: int = 144
    val_end: int = 168
    test_end: Optional[int] = None  # exclusive; None means the end of the panel
    aggregate: bool = False        # Test 1: collapse ALL series -> one national series
    # Test 1.1 may move the split earlier than the lattice's canonical split.
    # The source artifact is still validated against its own frozen contract,
    # then the national aggregate is re-normalized on this fold's train months.
    refit_normalization: bool = False
    # Historical artifacts may have selected their cohort using validation/test
    # availability. Official benchmark paths therefore fail closed by default.
    allow_legacy_artifact: bool = False


def census_config_from_config(cfg: Dict[str, Any], **overrides: Any) -> CensusConfig:
    """Build a ``CensusConfig`` from a composed config's ``data:`` block.

    Training, evaluation, the cost panel and the tabular aggregate runner must
    read the SAME window length, lag mode and split boundaries. Constructing
    ``CensusConfig`` from its dataclass defaults instead binds those scripts to
    values that merely happen to match ``config/census.yaml`` today.
    """
    data = dict(cfg.get("data", {}))
    data.update(overrides)
    fields = CensusConfig.__dataclass_fields__
    return CensusConfig(**{key: value for key, value in data.items() if key in fields})


def _artifact_contract_errors(
    cfg: CensusConfig,
    files: List[str],
    metadata: Dict[str, Any],
    n_months: int,
) -> List[str]:
    """Return reasons a lattice cannot prove the leakage-free contract."""
    errors: List[str] = []
    if "norm_range" not in files:
        errors.append("NPZ has no stored train-only norm_range")
    if cfg.aggregate:
        aggregate_source = metadata.get("aggregate_source")
        if "aggregate_raw" not in files:
            errors.append("NPZ has no unfiltered pre-cohort aggregate_raw")
        if not isinstance(aggregate_source, dict) or aggregate_source.get("scope") != (
            "all_raw_series_before_cohort_selection"
        ):
            errors.append("sidecar does not prove aggregate was built before cohort selection")
    missing_archives = metadata.get("source_missing_archives")
    if missing_archives:
        errors.append(
            f"sidecar reports {len(missing_archives)} missing monthly source archive(s)"
        )

    cohort = metadata.get("cohort_selection")
    if not isinstance(cohort, dict):
        errors.append("sidecar has no cohort_selection contract")
    else:
        if cohort.get("state_and_commodity_ranking") != "training_months_only":
            errors.append("cohort ranking is not marked training_months_only")
        if cohort.get("n_months") != cfg.train_end:
            errors.append(
                "cohort training length does not match configured train_end "
                f"({cohort.get('n_months')!r} != {cfg.train_end})"
            )

    expected_splits = {
        "train": [0, cfg.train_end],
        "val": [cfg.train_end, cfg.val_end],
        "test": [cfg.val_end, n_months],
    }
    splits = metadata.get("splits")
    if not isinstance(splits, dict):
        errors.append("sidecar has no chronological split contract")
    else:
        for name, expected in expected_splits.items():
            actual = splits.get(name)
            if actual != expected:
                errors.append(
                    f"sidecar {name} split does not match configured boundaries "
                    f"({actual!r} != {expected!r})"
                )
    return errors


def _normalization_contract_errors(
    cfg: CensusConfig,
    panel_raw: np.ndarray,
    panel_norm: np.ndarray,
    norm_min: np.ndarray,
    norm_range: np.ndarray,
) -> List[str]:
    """Verify that stored normalization was fitted on training months only."""
    errors: List[str] = []
    if panel_raw.shape != panel_norm.shape:
        return [
            "panel_raw and panel_norm shapes differ "
            f"({panel_raw.shape} != {panel_norm.shape})"
        ]
    if not 0 < cfg.train_end <= panel_raw.shape[1]:
        return [
            f"configured train_end {cfg.train_end} is outside panel length "
            f"{panel_raw.shape[1]}"
        ]

    expected_shape = (panel_raw.shape[0], panel_raw.shape[2])
    if norm_min.shape != expected_shape or norm_range.shape != expected_shape:
        return [
            "normalization arrays have the wrong shape "
            f"(norm_min={norm_min.shape}, norm_range={norm_range.shape}, "
            f"expected={expected_shape})"
        ]

    train_log = np.log1p(np.clip(panel_raw[:, :cfg.train_end, :], 0.0, None))
    expected_min = train_log.min(axis=1)
    expected_max = train_log.max(axis=1)
    expected_range = np.maximum(expected_max - expected_min, 1.0)
    if not np.allclose(norm_min, expected_min, rtol=1e-5, atol=1e-6):
        errors.append("norm_min does not match a training-only log1p fit")
    if not np.allclose(norm_range, expected_range, rtol=1e-5, atol=1e-6):
        errors.append("norm_range does not match a training-only log1p fit")

    # Verify the complete panel in bounded chunks. Correct training bounds are
    # not enough if panel_norm itself was generated with a different transform.
    for start in range(0, panel_raw.shape[0], 1024):
        stop = min(start + 1024, panel_raw.shape[0])
        log_chunk = np.log1p(np.clip(panel_raw[start:stop], 0.0, None))
        expected = (
            (log_chunk - norm_min[start:stop, None, :])
            / norm_range[start:stop, None, :]
        )
        if not np.allclose(
            panel_norm[start:stop], expected, rtol=2e-5, atol=2e-6,
            equal_nan=False,
        ):
            errors.append("panel_norm is inconsistent with the stored training-only transform")
            break
    return errors


class CensusLattice:
    """Loads `census_lattice*.npz` + its `.json`, builds the feature panel, exposes
    the id / combo metadata the trainer needs, and hands back PyTorch dataloaders."""

    def __init__(self, config: Union[CensusConfig, Dict[str, Any]]):
        self.config = CensusConfig(**config) if isinstance(config, dict) else config
        cfg = self.config

        d = np.load(cfg.npz)
        meta_path = Path(cfg.npz).with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
        else:
            meta = {}

        panel_norm = d["panel_norm"].astype(np.float32)
        panel_raw = d["panel_raw"].astype(np.float32)
        aggregate_raw = (
            d["aggregate_raw"].astype(np.float32)
            if "aggregate_raw" in d.files else None
        )
        norm_min = d["norm_min"].astype(np.float64)
        norm_range = (
            d["norm_range"].astype(np.float64)
            if "norm_range" in d.files else None
        )
        panel_shape = panel_norm.shape
        if len(panel_shape) != 3:
            raise ValueError(f"panel_norm must be (N,T,K), got {panel_shape}")
        if aggregate_raw is not None and aggregate_raw.shape != (panel_shape[1], 2):
            raise ValueError(
                f"aggregate_raw must be (T,2), got {aggregate_raw.shape} for T={panel_shape[1]}"
            )
        test_end = int(panel_shape[1]) if cfg.test_end is None else int(cfg.test_end)
        burn = cfg.input_len + cfg.lag_count
        if not burn < cfg.train_end < cfg.val_end < test_end <= int(panel_shape[1]):
            raise ValueError(
                "Census split must satisfy input_len + lag_count < train_end < "
                f"val_end < test_end <= {panel_shape[1]}; got burn={burn}, "
                f"train_end={cfg.train_end}, val_end={cfg.val_end}, test_end={test_end}"
            )
        if cfg.refit_normalization and not cfg.aggregate:
            raise ValueError("refit_normalization is restricted to aggregate Test 1.1 runs")
        if not cfg.allow_legacy_artifact:
            validation_cfg = cfg
            if cfg.refit_normalization:
                # A rolling fold intentionally differs from the artifact's
                # canonical split. Validate the artifact against the split it
                # declares before using panel_raw to fit the fold transform.
                splits = meta.get("splits", {})
                try:
                    canonical_train_end = int(splits["train"][1])
                    canonical_val_end = int(splits["val"][1])
                except (KeyError, TypeError, ValueError, IndexError):
                    canonical_train_end = cfg.train_end
                    canonical_val_end = cfg.val_end
                validation_cfg = replace(
                    cfg, train_end=canonical_train_end, val_end=canonical_val_end,
                    test_end=None, refit_normalization=False,
                )
            contract_errors = _artifact_contract_errors(
                validation_cfg, list(d.files), meta, int(panel_shape[1])
            )
            if norm_range is not None:
                contract_errors.extend(_normalization_contract_errors(
                    validation_cfg, panel_raw, panel_norm, norm_min, norm_range,
                ))
            if contract_errors:
                details = "; ".join(contract_errors)
                raise ValueError(
                    f"unsafe Census lattice {cfg.npz}: {details}. Rebuild it with "
                    "scripts/build_census_lattice.py. Historical inspection only: "
                    "set CensusConfig(allow_legacy_artifact=True)."
                )

        self.panel = panel_norm                              # (N, T, K) normalized
        self.panel_raw = panel_raw                           # (N, T, K) real units
        self.series_idx = d["series_idx"].astype(np.int64)   # (N, 3) comm,state,flow ids
        self.mask_grid = d["mask"]                           # (C, S, Flow) bool
        self.norm_min = norm_min
        # Legacy inspection can reproduce the builder's floored divisor, but
        # official paths above require the explicitly stored train-only range.
        self.norm_range = (
            norm_range
            if norm_range is not None
            else np.maximum(d["norm_max"].astype(np.float64) - self.norm_min, 1.0)
        )
        self.N, self.T, self.K = self.panel.shape
        self.test_end = test_end

        # Channel fallback is reachable only under explicit legacy opt-in.
        if meta:
            self.channels: List[str] = meta["channels"]
            default_weight = self.channels.index("agg_weight") if "agg_weight" in self.channels else 1
            self.target_ch: List[int] = meta.get("target_channels", [0, default_weight])
        else:
            meta = {}
            self.channels = [f"ch{i}" for i in range(self.K)]
            self.target_ch = [0, 1]
        self.metadata = meta
        assert len(self.target_ch) == 2, "expects (value, weight) aggregate targets"

        self.C, self.S, self.Flow = self.mask_grid.shape
        self.num_commodities, self.num_states, self.num_flows = self.C, self.S, self.Flow
        self.num_combos = self.N

        # combo_coords in (State, Commodity, Flow) order; series_idx is (comm,state,flow).
        self.combo_coords = np.stack(
            [self.series_idx[:, 1], self.series_idx[:, 0], self.series_idx[:, 2]], axis=1
        ).astype(np.int64)
        self.lattice_dims: Tuple[int, int, int] = (self.S, self.C, self.Flow)

        if cfg.aggregate:                            # Test 1: one national series
            if aggregate_raw is None:
                # Explicit legacy inspection only. Official aggregate runs
                # fail the contract above instead of silently reusing the
                # future-selected multidimensional cohort.
                raw = self.panel_raw.astype(np.float64)
                agg = raw[:, :, self.target_ch].sum(axis=0)
            else:
                # Test 1.1 receives no values beyond its declared fold. This is
                # stronger than merely bounding the Dataset indices: future
                # months never enter its in-memory feature/target panel.
                aggregate_limit = self.test_end if cfg.refit_normalization else self.T
                agg = aggregate_raw[:aggregate_limit].astype(np.float64)
            logp = np.log1p(np.clip(agg, 0.0, None))
            mn, mx = logp[:cfg.train_end].min(0), logp[:cfg.train_end].max(0)
            rng = np.maximum(mx - mn, 1.0)
            self.panel = ((logp - mn) / rng)[None].astype(np.float32)  # (1,T,2)
            self.panel_raw = agg[None].astype(np.float32)
            self.norm_min = mn[None]
            self.norm_range = rng[None]
            self.N, self.T, self.K = 1, agg.shape[0], 2
            self.channels = ["agg_value", "agg_weight"]; self.target_ch = [0, 1]
            self.series_idx = np.zeros((1, 3), np.int64)
            self.C = self.S = self.Flow = 1
            self.num_commodities = self.num_states = self.num_flows = self.num_combos = 1
            self.combo_coords = np.zeros((1, 3), np.int64)
            self.lattice_dims = (1, 1, 1)

        # which channels get explicit lags
        self.lag_channels: List[int] = (
            list(self.target_ch) if cfg.lag_mode == "agg" else list(range(self.K))
        )
        self.feat = self._build_features()                   # (N, T, Fdim)
        self.features_per_group = self.feat.shape[2]
        self.feat_cols = (
            list(self.channels)
            + [f"{self.channels[ch]}_lag_{k}" for ch in self.lag_channels
               for k in range(1, cfg.lag_count + 1)]
            + ["sin_month", "cos_month"]
        )
        combo_mb = self.N * cfg.input_len * self.features_per_group * 4 / 1e6
        print(f"[census] {self.N:,} series | grid {self.C}x{self.S}x{self.Flow} | "
              f"{self.T} months | {self.K} channels | F={self.features_per_group} "
              f"(lag_mode={cfg.lag_mode}) | combo/sample ~ {combo_mb:.0f} MB")

    def inverse_targets(self, values: np.ndarray, series_ids: np.ndarray) -> np.ndarray:
        """Invert normalized (aggregate value, aggregate weight) predictions.

        The transform is ``log1p`` followed by a per-series MinMax transform
        whose range is floored at one. ``norm_range`` is therefore required;
        using ``norm_max - norm_min`` is wrong for low-variation channels.
        """
        values = np.asarray(values, dtype=np.float64)
        series_ids = np.asarray(series_ids, dtype=np.int64).reshape(-1)
        if values.ndim != 2 or values.shape[1] != 2 or len(series_ids) != len(values):
            raise ValueError("values must be (n,2) and series_ids must have length n")
        channels = np.asarray(self.target_ch, dtype=np.int64)
        mins = self.norm_min[series_ids][:, channels]
        ranges = self.norm_range[series_ids][:, channels]
        return np.expm1(values * ranges + mins)

    def _build_features(self) -> np.ndarray:
        N, T, K, L = self.N, self.T, self.K, self.config.lag_count
        Fdim = K + len(self.lag_channels) * L + 2
        feat = np.zeros((N, T, Fdim), dtype=np.float32)
        feat[:, :, :K] = self.panel                          # the K mode channels
        col = K
        for ch in self.lag_channels:                         # explicit lags
            base = self.panel[:, :, ch]                      # (N, T)
            for k in range(1, L + 1):
                feat[:, k:, col] = base[:, :-k]
                col += 1
        months = (np.arange(T) % 12) + 1                     # calendar month 1..12
        feat[:, :, col] = np.sin(2 * np.pi * months / 12)[None, :]; col += 1
        feat[:, :, col] = np.cos(2 * np.pi * months / 12)[None, :]; col += 1
        return feat

    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        generator: Optional[torch.Generator] = None,
        worker_init_fn: Optional[Any] = None,
        combo: bool = False,
        effective_batch_size: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        cfg = self.config
        burn = cfg.input_len + cfg.lag_count                 # first target w/ full history+lags
        bounds = [
            (burn, cfg.train_end),
            (cfg.train_end, cfg.val_end),
            (cfg.val_end, self.test_end),
        ]
        tv, tw = self.target_ch                              # value, weight target channels

        if combo:
            sets = [CensusComboDataset(self.feat, self.panel, cfg.input_len, lo, hi, tv, tw)
                    for lo, hi in bounds]
            coll = combo_collate_fn
        else:
            sets = [CensusDataset(self.feat, self.series_idx, self.panel,
                                  cfg.input_len, lo, hi, tv, tw)
                    for lo, hi in bounds]
            coll = collate_fn

        # NOTE: do NOT add persistent_workers=True here. It looks like a free
        # win -- the combo loader emits only ~96 batches per epoch, so
        # respawning workers each epoch is real overhead -- but it is not
        # numerically neutral. With num_workers > 0 the DataLoader draws a base
        # seed from `generator` every time it builds a worker iterator, i.e.
        # once per epoch; persisting the workers draws it once for the whole
        # run. Since that same generator also drives the sampler's shuffle, the
        # epoch-1-onward batch order changes (verified: epoch 0 matched, every
        # later epoch differed, and end-to-end training loss moved). Worker
        # COUNT is safe -- 0/1/2/4/8 all give the identical order -- it is
        # worker LIFETIME that is not. Pinned by
        # tests/test_combo_loader_equivalence.py.
        def mk(ds: Dataset, shuffle: bool, drop: bool) -> DataLoader:
            return DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop,
                num_workers=num_workers, pin_memory=True, collate_fn=coll,
                generator=generator if shuffle else None, worker_init_fn=worker_init_fn,
            )

        if effective_batch_size is not None:
            if combo:
                raise ValueError("effective_batch_size is only valid for flat Census data")
            batch_sampler = ExactGroupBatchSampler(
                len(sets[0]), batch_size, effective_batch_size,
                shuffle=shuffle_train, generator=generator,
            )
            train = DataLoader(
                sets[0], batch_sampler=batch_sampler, num_workers=num_workers,
                pin_memory=True, collate_fn=coll, generator=generator,
                worker_init_fn=worker_init_fn,
            )
        else:
            # Retain the final partial batch. Dropping it makes data exposure and
            # optimizer-step counts depend on divisibility by the memory batch.
            train = mk(sets[0], shuffle_train, False)
        val, test = mk(sets[1], False, False), mk(sets[2], False, False)
        print(f"[census] dataloaders ({'combo' if combo else 'per-series'}): "
              f"{len(train)} train, {len(val)} val, {len(test)} test batches")
        return train, val, test


class ExactGroupBatchSampler(Sampler[List[int]]):
    """Shuffle samples, then emit fixed-size optimizer groups as micro-batches.

    A flat Census sample is one series-month observation, whereas one combo
    sample carries every series for a month. For exact step matching, each flat
    optimizer group must therefore contain exactly ``G`` observations. This
    sampler partitions the epoch into groups of that size and then slices each
    group into memory-safe micro-batches without allowing a micro-batch to cross
    an optimizer-group boundary.
    """

    def __init__(self, dataset_size: int, micro_batch_size: int,
                 effective_batch_size: int, shuffle: bool = True,
                 generator: Optional[torch.Generator] = None):
        if dataset_size <= 0 or micro_batch_size <= 0 or effective_batch_size <= 0:
            raise ValueError("dataset and batch sizes must be positive")
        if dataset_size % effective_batch_size:
            raise ValueError(
                f"dataset size {dataset_size} is not divisible by exact effective "
                f"batch {effective_batch_size}"
            )
        self.dataset_size = int(dataset_size)
        self.micro_batch_size = int(micro_batch_size)
        self.effective_batch_size = int(effective_batch_size)
        self.shuffle = bool(shuffle)
        self.generator = generator
        self.micro_batches_per_group = (
            self.effective_batch_size + self.micro_batch_size - 1
        ) // self.micro_batch_size
        self.num_groups = self.dataset_size // self.effective_batch_size

    def __iter__(self):
        indices = (torch.randperm(self.dataset_size, generator=self.generator)
                   if self.shuffle else None)
        for group_start in range(0, self.dataset_size, self.effective_batch_size):
            group_end = group_start + self.effective_batch_size
            for micro_start in range(group_start, group_end, self.micro_batch_size):
                micro_end = min(micro_start + self.micro_batch_size, group_end)
                if indices is None:
                    yield list(range(micro_start, micro_end))
                else:
                    yield indices[micro_start:micro_end].tolist()

    def __len__(self) -> int:
        return self.num_groups * self.micro_batches_per_group


class CensusDataset(Dataset):
    """Per-series windows, sliced on the fly. Emits the same dict as ``TradeDataset``."""

    def __init__(self, feat: np.ndarray, series_idx: np.ndarray, panel: np.ndarray,
                 input_len: int, target_lo: int, target_hi: int, tv: int, tw: int):
        self.feat = feat                     # (N, T, F)
        self.series_idx = series_idx         # (N, 3) comm,state,flow
        self.panel = panel                   # (N, T, K) for targets
        self.input_len = input_len
        self.tv, self.tw = tv, tw            # target value/weight channel indices
        N = feat.shape[0]
        ts = np.arange(target_lo, target_hi, dtype=np.int64)
        s = np.repeat(np.arange(N, dtype=np.int64), len(ts))
        t = np.tile(ts, N)
        self.samples = np.stack([s, t], axis=1)                     # (M, 2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s, t = int(self.samples[idx, 0]), int(self.samples[idx, 1])
        x = self.feat[s, t - self.input_len:t, :]                  # (L, F)
        comm, state, flow = self.series_idx[s]
        return {
            "x_numeric": torch.from_numpy(x.copy()),
            "state_id": torch.tensor(int(state), dtype=torch.long),
            "comm_id": torch.tensor(int(comm), dtype=torch.long),
            "flow_id": torch.tensor(int(flow), dtype=torch.long),
            "target_value": torch.tensor(self.panel[s, t, self.tv], dtype=torch.float32),
            "target_weight": torch.tensor(self.panel[s, t, self.tw], dtype=torch.float32),
            "target_time": int(t),
        }


class CensusComboDataset(Dataset):
    """All-groups-per-window, sliced + transposed on the fly. Same dict as
    ``TradeComboDataset``.

    ``group_mask`` is all-ones: every packed series exists at every month (the
    dense panel 0-fills no-trade months). Structural grid sparsity (which
    C×S×Flow cells are empty) is handled inside ``AxialComboSA`` via
    ``combo_coords`` + ``lattice_dims``, not here.

    A window is ``(L, G, F)`` but the panel is stored ``(G, T, F)``, so every
    sample needs the time axis in front. Slicing ``feat[:, t-L:t, :]`` and
    transposing produces a STRIDED view whose ``ascontiguousarray`` is a full
    gather of the window -- 143 MB at the benchmark's G, measured at ~51 ms per
    sample, several times the GPU cost of the step it feeds. The panel is
    static, so the transpose is hoisted into ``__init__``: one ``(T, G, F)``
    copy (0.76 GB at benchmark scale) makes every window a contiguous slice,
    0.133 ms per sample. Bytes and dtype are identical either way; only the
    cost moves. Pinned by tests/test_combo_loader_equivalence.py.
    """

    def __init__(self, feat: np.ndarray, panel: np.ndarray,
                 input_len: int, target_lo: int, target_hi: int, tv: int, tw: int):
        self.panel = panel                   # (N=G, T, K)
        self.input_len = input_len
        self.tv, self.tw = tv, tw
        self.G = feat.shape[0]
        self.samples = np.arange(target_lo, target_hi, dtype=np.int64)
        # (G, T, F) -> (T, G, F), once. A window is then feat_t[t-L:t], already
        # in the layout the model expects and already contiguous.
        self.feat_t = np.ascontiguousarray(feat.transpose(1, 0, 2))
        # All-ones and never varies; rebuilding it per sample allocates G floats
        # for a constant. Shared read-only -- collate stacks (copies) it.
        self._group_mask = torch.ones(self.G, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = int(self.samples[idx])
        x = self.feat_t[t - self.input_len:t]                          # (L, G, F)
        return {
            "x_numeric": torch.from_numpy(x),
            "group_mask": self._group_mask,
            "target_value": torch.from_numpy(self.panel[:, t, self.tv].copy()),   # (G,)
            "target_weight": torch.from_numpy(self.panel[:, t, self.tw].copy()),  # (G,)
            "target_time": int(t),
        }
