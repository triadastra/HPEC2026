"""
Unified data pipeline for time series forecasting.

Implements TradeDataPipeline class that loads, preprocesses, normalizes,
and creates train/val/test splits for agricultural trade data.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Column name constants
VALUE_CANON = "Value"
WEIGHT_CANON = "Weight"
KEYS = ["State", "Commodity", "Import/Export"]

# Raw column name mappings
EXPORT_VALUE_RAW = "Containerized Vessel Total Exports Value ($US)"
EXPORT_WEIGHT_RAW = "Containerized Vessel Total Exports SWT (kg)"
IMPORT_VALUE_PRI = "Vessel Value ($US)"
IMPORT_VALUE_ALT = "Customs Containerized Vessel Value (Gen) ($US)"
IMPORT_WEIGHT_PRI = "Vessel SWT (kg)"
IMPORT_WEIGHT_ALT = "Containerized Vessel SWT (Gen) (kg)"

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}


@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    imports_dir: str = "data/imports"
    exports_dir: str = "data/exports"
    start: str = "2009-01-01"
    end: str = "2025-05-01"
    input_len: int = 36
    lag_count: int = 12
    normalize: bool = True
    normalization_method: str = "minmax"

    # Sparse-combo filtering. The dense monthly grid fills no-trade months
    # with 0, so a large fraction of (State, Commodity, Flow) combos are
    # structurally sparse (mostly zero). Those zeros make relative metrics
    # (MAPE) explode and add no signal. When ``filter_sparse`` is True we
    # keep only combos whose RAW Value is nonzero in at least
    # ``min_nonzero_frac`` of TRAINING months (<= train_end), matching the
    # census path's train-only cohort policy. This DIVERGES from paper
    # included/ (which used every combo): the Phase 7 equivalence test must
    # set filter_sparse=False; real runs use the default.
    filter_sparse: bool = True
    min_nonzero_frac: float = 0.95
    aggregate: bool = False  # Test 1: collapse all combos into one summed series

    # Data splits
    train_end: str = "2023-12-01"
    val_start: str = "2024-01-01"
    val_end: str = "2024-12-01"
    test_start: str = "2025-01-01"
    test_end: str = "2025-05-01"


class TradeDataPipeline:
    """
    Unified data pipeline for agricultural trade forecasting.

    This class handles:
    - Loading CSV data from imports/exports directories
    - Creating complete monthly timeline grid
    - Adding lag features and seasonal encodings
    - Computing hierarchical normalization stats
    - Creating train/val/test splits
    - Building PyTorch dataloaders
    """

    def __init__(self, config: Union[DataConfig, Dict[str, Any]]):
        """
        Initialize data pipeline.

        Args:
            config: DataConfig instance or config dictionary
        """
        if isinstance(config, dict):
            self.config = DataConfig(**config)
        else:
            self.config = config

        # Internal state
        self.df_full = None
        self.stats = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.feat_cols = None
        self.kept_combos = None

        # Mappings
        self.state2id = {}
        self.comm2id = {}
        self.flow2id = {}

    def parse_month(self, s) -> pd.Timestamp:
        """Parse month string to Timestamp."""
        if pd.isna(s):
            return pd.NaT
        s = str(s).strip()

        # Try YYYY-MM
        try:
            dt = pd.to_datetime(s, format="%Y-%m", errors="raise")
            return pd.Timestamp(dt.year, dt.month, 1)
        except Exception:
            pass

        # Try "YY-Mon" (e.g., "08-Jan")
        try:
            yy, mon = s.split("-")
            return pd.Timestamp(int(yy) + 2000, MONTH_MAP[mon.capitalize()], 1)
        except Exception:
            pass

        # Fallback
        try:
            dt = pd.to_datetime(s)
            return pd.Timestamp(dt.year, dt.month, 1)
        except Exception:
            return pd.NaT

    def _numeric_clean(self, series: pd.Series) -> pd.Series:
        """Clean numeric series by removing commas and converting to float."""
        return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)

    def read_folder(self, folder: str, label: str) -> pd.DataFrame:
        """
        Read all CSV files from a folder.

        Args:
            folder: Path to folder containing CSV files
            label: "imports" or "exports"

        Returns:
            Combined DataFrame
        """
        files = sorted([str(p) for p in Path(folder).glob("*.csv")])
        if not files:
            raise FileNotFoundError(f"No CSV files in {folder}")

        dfs = []
        for fp in files:
            df = pd.read_csv(fp, on_bad_lines="skip", dtype=str)

            # Ensure required columns
            for k in ["State", "Commodity", "Country", "Time"]:
                if k not in df.columns:
                    raise ValueError(f"{fp} missing column: {k}")

            df["Time"] = df["Time"].apply(self.parse_month)

            if label.lower().startswith("export"):
                for c in [EXPORT_VALUE_RAW, EXPORT_WEIGHT_RAW]:
                    if c not in df.columns:
                        raise ValueError(f"{fp} missing export column: {c}")
                df[EXPORT_VALUE_RAW] = self._numeric_clean(df[EXPORT_VALUE_RAW])
                df[EXPORT_WEIGHT_RAW] = self._numeric_clean(df[EXPORT_WEIGHT_RAW])
                df = df.rename(columns={
                    EXPORT_VALUE_RAW: VALUE_CANON,
                    EXPORT_WEIGHT_RAW: WEIGHT_CANON
                })
            else:
                vcol = IMPORT_VALUE_PRI if IMPORT_VALUE_PRI in df.columns else IMPORT_VALUE_ALT
                wcol = IMPORT_WEIGHT_PRI if IMPORT_WEIGHT_PRI in df.columns else IMPORT_WEIGHT_ALT
                for c in [vcol, wcol]:
                    if c not in df.columns:
                        raise ValueError(f"{fp} missing import column: {c}")
                df[vcol] = self._numeric_clean(df[vcol])
                df[wcol] = self._numeric_clean(df[wcol])
                df = df.rename(columns={vcol: VALUE_CANON, wcol: WEIGHT_CANON})

            df["Import/Export"] = "Import" if label.lower().startswith("import") else "Export"
            keep = ["State", "Commodity", "Country", "Time", "Import/Export", VALUE_CANON, WEIGHT_CANON]
            dfs.append(df[keep])

        return pd.concat(dfs, ignore_index=True)

    def load_data(self) -> pd.DataFrame:
        """
        Load data from imports and exports directories.

        Returns:
            Combined DataFrame with all trade data
        """
        # Load imports and exports
        df_imports = self.read_folder(self.config.imports_dir, "imports")
        df_exports = self.read_folder(self.config.exports_dir, "exports")

        # Combine
        df_full = pd.concat([df_imports, df_exports], ignore_index=True)

        print(f"Loaded {len(df_imports)} import rows and {len(df_exports)} export rows")
        print(f"Total combined rows: {len(df_full)}")

        self.df_full = df_full
        return df_full

    def build_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build complete monthly timeline grid.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with complete monthly timeline
        """
        # Aggregate to monthly
        num_cols = [c for c in df.columns if c not in KEYS + ["Country", "Time"]]
        grouped = df.groupby(KEYS + ["Time"], dropna=False)[num_cols].sum().reset_index()

        # Full monthly timeline
        full_time = pd.date_range(start=self.config.start, end=self.config.end, freq="MS")
        time_df = pd.DataFrame({"Time": full_time, "key": 1})
        unique_keys = grouped[KEYS].drop_duplicates().assign(key=1)

        grid = unique_keys.merge(time_df, on="key", how="outer").drop(columns="key")
        grid = grid.merge(grouped, on=KEYS + ["Time"], how="left")

        for c in num_cols:
            grid[c] = grid[c].fillna(0.0)

        # Add month encodings
        grid["month"] = grid["Time"].dt.month
        grid["sin_month"] = np.sin(2 * np.pi * grid["month"] / 12)
        grid["cos_month"] = np.cos(2 * np.pi * grid["month"] / 12)

        return grid

    def filter_sparse_combos(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Drop structurally-sparse (State, Commodity, Flow) combos.

        Keeps only combos whose RAW ``Value`` is nonzero in at least
        ``config.min_nonzero_frac`` of TRAINING months. Membership is
        decided before any validation/test observation is consulted — the
        same fail-closed policy the census path enforces (see
        ``build_census_lattice._select_cohort`` and the loader's
        cohort_selection contract) — so future target availability cannot
        change who is evaluated.

        Must be called on the raw grid BEFORE normalization, so the
        nonzero test is against real dollar values and not normalized ones.

        NOTE: diverges from paper included/, which used every combo. The
        kept set is deterministic given (train slice, threshold), so
        inference reconstructs the identical population. Disable via
        ``filter_sparse=False`` for the equivalence test.
        """
        thr = self.config.min_nonzero_frac
        train_slice = grid[grid["Time"] <= pd.Timestamp(self.config.train_end)]
        nz_frac = train_slice.groupby(KEYS)[VALUE_CANON].apply(lambda s: float((s != 0).mean()))
        keep_idx = nz_frac[nz_frac >= thr].index
        before = nz_frac.shape[0]
        after = keep_idx.shape[0]
        if after == 0:
            raise ValueError(
                f"Sparse-combo filter dropped ALL {before} combos at "
                f"min_nonzero_frac={thr}. Lower the threshold."
            )
        keep_df = keep_idx.to_frame(index=False)
        filtered = grid.merge(keep_df, on=KEYS, how="inner")
        self.kept_combos = sorted(map(tuple, keep_df.itertuples(index=False, name=None)))
        print(
            f"Sparse-combo filter (>= {thr:.0%} nonzero raw Value over train months): "
            f"kept {after}/{before} combos, dropped {before - after}"
        )
        return filtered

    def add_lags(self, df: pd.DataFrame, n_lags: Optional[int] = None) -> pd.DataFrame:
        """
        Add lag features.

        Args:
            df: Input DataFrame
            n_lags: Number of lags (default from config)

        Returns:
            DataFrame with lag features
        """
        if n_lags is None:
            n_lags = self.config.lag_count

        df = df.sort_values(KEYS + ["Time"]).copy()
        for k in range(1, n_lags + 1):
            df[f"value_lag_{k}"] = df.groupby(KEYS)[VALUE_CANON].shift(k).fillna(0.0)
            df[f"weight_lag_{k}"] = df.groupby(KEYS)[WEIGHT_CANON].shift(k).fillna(0.0)

        # Burn-in: drop only the months whose LAG columns would be fabricated
        # by the .fillna(0.0) above. That is `n_lags` months -- NOT
        # `input_len + n_lags`.
        #
        # The receptive field of one prediction really is input_len + n_lags,
        # but TradeDataset._build_sequences already reserves the input_len
        # part: it emits `target_idx = i + input_len`, so the first target
        # already sits input_len months past the first surviving row. Deleting
        # input_len months here as well applies that requirement twice, pushes
        # the first target input_len months late, and loses that many target
        # months on every series. Measured on the base.yaml config: the first
        # target moved 2016-01 -> 2013-01 and train targets 96 -> 132 per
        # series, with val/test counts unchanged (this only affects the start
        # of the panel). The input_len months are history the first window
        # CONSUMES -- they must be reserved, not deleted.
        #
        # Contract: first valid target == start + n_lags + input_len.
        # For start=2008-01, n_lags=12, input_len=36 that is 2012-01, so the
        # paper's §5.6 statement ("Years from 2008 to 2011 were excluded")
        # still holds under this trim: those years yield no targets. The
        # sentence describes which months can be TARGETS, not how many rows to
        # delete. The original WCTR pipeline dropped year 2008 only
        # (= n_lags months), i.e. it already satisfied the contract; the
        # four-year drop was the change that broke it.
        #
        # Month granularity matters independently: the old `// 12` integer
        # division dropped 0 years for any n_lags < 12 -- letting fabricated
        # lags into the first windows -- and was wrong for any start date that
        # is not January. A window/lag sensitivity sweep hits both cases.
        # Pinned by tests/test_burn_in_contract.py.
        first_valid = pd.Timestamp(self.config.start) + pd.DateOffset(months=n_lags)
        df = df[df["Time"] >= first_valid]

        return df

    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns.

        Returns:
            List of feature column names
        """
        n_lags = self.config.lag_count
        core_feats = (
            [VALUE_CANON, WEIGHT_CANON]
            + [f"value_lag_{k}" for k in range(1, n_lags + 1)]
            + [f"weight_lag_{k}" for k in range(1, n_lags + 1)]
            + ["sin_month", "cos_month"]
        )
        return core_feats

    def compute_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute per-combo min/max normalization statistics from training data.

        One (min, max) pair per exact (State, Commodity, Flow) combo — no
        borrowing from broader groups. The 95% nonzero filter ensures every
        kept combo has enough real observations to compute meaningful stats.
        """
        s_exact = (
            df.groupby(KEYS)[[VALUE_CANON, WEIGHT_CANON]]
            .agg(["min", "max"])
            .reset_index()
        )
        s_exact.columns = ["State", "Commodity", "Import/Export",
                           "val_min", "val_max", "wt_min", "wt_max"]
        # If min == max (constant series) the range is 0 — normalize to 0.
        # Guard by setting range to 1 so division is safe; denorm recovers
        # the constant by multiplying 0 * 1 + min = min.
        for col in ("val_min", "val_max", "wt_min", "wt_max"):
            s_exact[col] = s_exact[col].fillna(0.0)
        return {"exact": s_exact}

    def normalize_minmax(self, df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
        """Apply per-combo min-max normalization: x_norm = (x - min) / (max - min).

        Maps training range to [0, 1]. Val/test values outside training range
        can exceed [0, 1] — that is intentional and correct (no clipping).
        Constant series (max == min) are mapped to 0.
        """
        out = df.sort_values(KEYS + ["Time"]).copy()
        out = out.merge(stats["exact"], on=KEYS, how="left")

        val_range = (out["val_max"] - out["val_min"]).replace(0, 1.0)
        wt_range  = (out["wt_max"]  - out["wt_min"]).replace(0, 1.0)

        out[VALUE_CANON]  = (out[VALUE_CANON]  - out["val_min"]) / val_range
        out[WEIGHT_CANON] = (out[WEIGHT_CANON] - out["wt_min"])  / wt_range

        out = out.drop(columns=["val_min", "val_max", "wt_min", "wt_max"])
        return out

    def build_id_maps(self, df: pd.DataFrame):
        """
        Build ID mappings for categorical features.

        Args:
            df: DataFrame with categorical features
        """
        states = sorted(df["State"].dropna().unique().tolist())
        comms = sorted(df["Commodity"].dropna().unique().tolist())
        flows = sorted(df["Import/Export"].dropna().unique().tolist())

        self.state2id = {s: i for i, s in enumerate(states)}
        self.comm2id = {c: i for i, c in enumerate(comms)}
        self.flow2id = {f: i for i, f in enumerate(flows)}

        print(f"Built mappings: {len(states)} states, {len(comms)} commodities, {len(flows)} flows")

    def build_combo_maps(self, df: pd.DataFrame):
        """
        Build mapping from (State, Commodity, Import/Export) combinations
        to integer slot indices. Required by combo-window models
        (CrossAttention / FiLM variants) which need a fixed G dimension.

        Mirrors ``paper included/shared_dataloader.py::build_combo_maps``.
        Should be called on the TRAIN slice (or full grid) so val/test
        do not introduce previously-unseen combos.
        """
        unique = (
            df[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
        )
        self.combo2id = {
            (row["State"], row["Commodity"], row["Import/Export"]): i
            for i, row in unique.iterrows()
        }
        self.id2combo = {i: k for k, i in self.combo2id.items()}
        self.num_combos = len(self.combo2id)
        print(f"Built combo map: {self.num_combos} unique (state, commodity, flow) combos")

        # Lattice coordinates for the axial (S4ND-style 2d/3d/4d) combo models:
        # map each combo's g_idx -> (state_idx, commodity_idx, flow_idx) over the
        # categories actually present in the kept combos. lattice_dims = (S,C,Flow).
        states = sorted({k[0] for k in self.combo2id})
        comms = sorted({k[1] for k in self.combo2id})
        flows = sorted({k[2] for k in self.combo2id})
        s2i = {s: i for i, s in enumerate(states)}
        c2i = {c: i for i, c in enumerate(comms)}
        f2i = {f: i for i, f in enumerate(flows)}
        coords = np.zeros((self.num_combos, 3), dtype=np.int64)
        for combo, g in self.combo2id.items():
            coords[g] = (s2i[combo[0]], c2i[combo[1]], f2i[combo[2]])
        self.combo_coords = coords
        self.lattice_dims = (len(states), len(comms), len(flows))

    def _aggregate_grid(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Collapse the whole grid into one summed monthly series (Test 1)."""
        agg = grid.groupby("Time", as_index=False)[[VALUE_CANON, WEIGHT_CANON]].sum()
        agg["State"] = "ALL"
        agg["Commodity"] = "ALL"
        agg["Import/Export"] = "ALL"
        agg["month"] = agg["Time"].dt.month
        agg["sin_month"] = np.sin(2 * np.pi * agg["month"] / 12)
        agg["cos_month"] = np.cos(2 * np.pi * agg["month"] / 12)
        self.kept_combos = [("ALL", "ALL", "ALL")]
        print(f"[aggregate] collapsed to 1 summed series over {len(agg)} months")
        return agg

    def create_splits(self):
        """
        Create train/val/test splits.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df_full is None:
            raise RuntimeError("Must call load_data() first")

        # Build grid (raw monthly panel, no lags yet)
        grid = self.build_grid(self.df_full)

        # Drop structurally-sparse combos on the RAW grid, before stats /
        # normalization, so the nonzero test is on real dollar values.
        if getattr(self.config, "aggregate", False):
            grid = self._aggregate_grid(grid)
        elif self.config.filter_sparse:
            grid = self.filter_sparse_combos(grid)

        # Create splits based on time
        t_train_end = pd.Timestamp(self.config.train_end)
        t_val_start = pd.Timestamp(self.config.val_start)
        t_val_end = pd.Timestamp(self.config.val_end)
        t_test_start = pd.Timestamp(self.config.test_start)
        t_test_end = pd.Timestamp(self.config.test_end)

        # IMPORTANT: normalize BEFORE adding lags. If we add lags first,
        # the lag columns hold raw dollar values (up to ~3e8 for high-volume
        # commodities) while Value/Weight get MinMax-normalized — feeding two
        # scales of input into the model. RNNs tolerate this via their Linear
        # input projection learning a tiny scale, but the Transformer's
        # attention softmax silently overflows to NaN. Paper-included
        # normalizes first, then computes lags, so lag columns are lags of
        # normalized values (matches `paper included/shared_dataloader.py`
        # ordering).

        # Compute normalization stats on the TRAIN slice only (no leakage)
        train_slice = grid[grid["Time"] <= t_train_end]
        self.stats = self.compute_stats(train_slice)

        # Apply normalization to the FULL grid using train stats.
        if self.config.normalize:
            grid = self.normalize_minmax(grid, self.stats)

            tr = grid[grid["Time"] <= t_train_end]
            print(f"[Train] minmax ~ Value [{tr[VALUE_CANON].min():.3f}, {tr[VALUE_CANON].max():.3f}]"
                  f" | Weight [{tr[WEIGHT_CANON].min():.3f}, {tr[WEIGHT_CANON].max():.3f}]")

        # Add lags AFTER normalization, so lag columns hold normalized values.
        grid = self.add_lags(grid)

        # Get feature columns (now that lag columns exist)
        self.feat_cols = self.get_feature_columns()

        # The "split DataFrames" returned here are reporting/inspection
        # views only — they are NOT what get_dataloaders feeds to
        # TradeDataset. See `self.df_full_grid` for the master.
        self.df_full_grid = grid
        df_train = grid[grid["Time"] <= t_train_end].copy()
        df_val = grid[(grid["Time"] >= t_val_start) & (grid["Time"] <= t_val_end)].copy()
        df_test = grid[(grid["Time"] >= t_test_start) & (grid["Time"] <= t_test_end)].copy()

        print(f"Train: {len(df_train)} rows until {t_train_end.date()}")
        print(f"Val: {len(df_val)} rows from {t_val_start.date()} to {t_val_end.date()}")
        print(f"Test: {len(df_test)} rows from {t_test_start.date()} to {t_test_end.date()}")

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        # Store time boundaries so get_dataloaders can build target-time
        # filtered datasets without re-parsing the config.
        self._t_train_end = t_train_end
        self._t_val_start = t_val_start
        self._t_val_end = t_val_end
        self._t_test_start = t_test_start
        self._t_test_end = t_test_end

        return df_train, df_val, df_test

    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        generator: Optional[torch.Generator] = None,
        worker_init_fn: Optional[Any] = None,
        combo: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch dataloaders.

        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            shuffle_train: Whether to shuffle training data
            generator: torch.Generator for reproducible shuffling. Pass the
                generator returned by ``src.utils.seeding.set_seed(seed)``.
            worker_init_fn: Callable seeded per worker. Pass
                ``src.utils.seeding.seed_worker`` to make NumPy/random
                deterministic inside workers.
            combo: If True, return dataloaders over a ``TradeComboDataset``
                with batches shaped ``(B, L, G, F)`` plus a ``group_mask``.
                Required by ``cross_attention`` / ``film_attention`` model
                variants (which set ``requires_combo_loader = True``).
                If False (default), returns per-series ``TradeDataset``
                loaders shaped ``(B, L, F)``.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.df_train is None:
            raise RuntimeError("Must call create_splits() first")

        # Build ID maps from the train slice (so val/test do not introduce
        # new categorical values).
        self.build_id_maps(self.df_train)
        if combo:
            self.build_combo_maps(self.df_train)

        # All three datasets read the FULL grid; the target_time filters
        # decide which samples each split actually emits. This is what
        # lets a 5-month test set still produce real samples whose 36-month
        # windows reach into the train period.
        #
        # Normally `df_full_grid` is set by `create_splits()`. Tests and
        # other callers that bypass `create_splits()` (e.g. by setting
        # `df_train/val/test` directly on the pipeline) fall back to
        # rebuilding the panel from the union of the three splits.
        if getattr(self, "df_full_grid", None) is not None:
            full = self.df_full_grid
            t_train_end = self._t_train_end
            t_val_start = self._t_val_start
            t_val_end = self._t_val_end
            t_test_start = self._t_test_start
            t_test_end = self._t_test_end
        else:
            full = pd.concat(
                [self.df_train, self.df_val, self.df_test],
                ignore_index=True,
            ).drop_duplicates(subset=KEYS + ["Time"]).sort_values(KEYS + ["Time"])
            # Derive boundaries from the actual split dfs rather than config
            # defaults, since callers in this path set the splits directly
            # and may not match config dates (e.g. synthetic test data).
            t_train_end = self.df_train["Time"].max()
            t_val_start = self.df_val["Time"].min() if len(self.df_val) else None
            t_val_end = self.df_val["Time"].max() if len(self.df_val) else None
            t_test_start = self.df_test["Time"].min() if len(self.df_test) else None
            t_test_end = self.df_test["Time"].max() if len(self.df_test) else None
        if combo:
            # Combo path: all-groups-per-window dataset, used by the
            # cross_attention / film_attention model variants.
            ds_kwargs = dict(
                feat_cols=self.feat_cols,
                input_len=self.config.input_len,
                combo2id=self.combo2id,
            )
            train_ds = TradeComboDataset(full, **ds_kwargs,
                                          target_time_start=None,
                                          target_time_end=t_train_end)
            val_ds = TradeComboDataset(full, **ds_kwargs,
                                        target_time_start=t_val_start,
                                        target_time_end=t_val_end)
            test_ds = TradeComboDataset(full, **ds_kwargs,
                                         target_time_start=t_test_start,
                                         target_time_end=t_test_end)
            active_collate_fn = combo_collate_fn
        else:
            train_ds = TradeDataset(
                full, self.feat_cols, self.config.input_len,
                self.state2id, self.comm2id, self.flow2id,
                target_time_start=None, target_time_end=t_train_end,
            )
            val_ds = TradeDataset(
                full, self.feat_cols, self.config.input_len,
                self.state2id, self.comm2id, self.flow2id,
                target_time_start=t_val_start, target_time_end=t_val_end,
            )
            test_ds = TradeDataset(
                full, self.feat_cols, self.config.input_len,
                self.state2id, self.comm2id, self.flow2id,
                target_time_start=t_test_start, target_time_end=t_test_end,
            )
            active_collate_fn = collate_fn

        # Create dataloaders. The collate fn (chosen above) stacks per-sample
        # dicts and renames keys to the plural form the trainer/model expects.
        # Without it the default collate produces singular keys and downstream
        # code silently breaks.
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle_train,
            drop_last=True, num_workers=num_workers, pin_memory=True,
            collate_fn=active_collate_fn, generator=generator,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=active_collate_fn, worker_init_fn=worker_init_fn,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=active_collate_fn, worker_init_fn=worker_init_fn,
        )

        mode = "combo" if combo else "per-series"
        print(f"Created dataloaders ({mode}): {len(train_loader)} train, "
              f"{len(val_loader)} val, {len(test_loader)} test batches")

        return train_loader, val_loader, test_loader

    def save_artifacts(self, out_dir: str):
        """
        Save data artifacts for later use.

        Args:
            out_dir: Output directory
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        artifacts = out_path / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        # Save stats
        if self.stats:
            stats_json = {"exact": self.stats["exact"].to_dict(orient="list")}
            (artifacts / "series_stats.json").write_text(json.dumps(stats_json, indent=2))

        # Save ID maps
        maps = {
            "state2id": self.state2id,
            "comm2id": self.comm2id,
            "flow2id": self.flow2id,
            "num_states": len(self.state2id),
            "num_commodities": len(self.comm2id),
            "num_flows": len(self.flow2id),
        }
        (artifacts / "id_maps.json").write_text(json.dumps(maps, ensure_ascii=False, indent=2))

        # Save features
        if self.feat_cols:
            (artifacts / "features.json").write_text(json.dumps({"feat_cols": self.feat_cols}, indent=2))

        # Save the kept-combo population (sparse filter) for transparency.
        if self.kept_combos is not None:
            (artifacts / "kept_combos.json").write_text(json.dumps({
                "min_nonzero_frac": self.config.min_nonzero_frac,
                "num_combos": len(self.kept_combos),
                "combos": [list(c) for c in self.kept_combos],
            }, indent=2))

        # Save normalized data
        if self.df_train is not None:
            self.df_train.to_csv(artifacts / "train_normalized.csv", index=False)
        if self.df_val is not None:
            self.df_val.to_csv(artifacts / "val_normalized.csv", index=False)
        if self.df_test is not None:
            self.df_test.to_csv(artifacts / "test_normalized.csv", index=False)

        print(f"Saved artifacts to {artifacts}")


class TradeDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.

    Each sample contains a sequence of length ``input_len`` with features
    for a specific (State, Commodity, Flow) combination, plus the target
    row at ``window_end + 1``.

    The dataset is always fed the FULL normalized monthly grid. Which
    samples are emitted is controlled by ``target_time_start`` /
    ``target_time_end``: a sample is emitted iff the target row's ``Time``
    falls within ``[target_time_start, target_time_end]``. This is the
    only correct way to construct val/test sets — a 12-month val slice
    cannot fit a 36-month window inside itself, so val/test windows MUST
    reach back into earlier splits for historical context.

    Train: pass ``target_time_end=train_end`` to keep targets within train.
    Val:   pass ``[val_start, val_end]``; windows reach back into train.
    Test:  pass ``[test_start, test_end]``; windows reach back into train+val.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        input_len: int,
        state2id: Dict[str, int],
        comm2id: Dict[str, int],
        flow2id: Dict[str, int],
        target_time_start: Optional[pd.Timestamp] = None,
        target_time_end: Optional[pd.Timestamp] = None,
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with time series data (typically the full
                normalized grid, not a temporally pre-sliced subset).
            feat_cols: List of feature column names
            input_len: Input sequence length
            state2id: State to ID mapping
            comm2id: Commodity to ID mapping
            flow2id: Flow to ID mapping
            target_time_start: Optional inclusive lower bound on target
                time. If None, no lower bound.
            target_time_end: Optional inclusive upper bound on target
                time. If None, no upper bound.
        """
        self.df = df.sort_values(KEYS + ["Time"]).copy()
        self.feat_cols = feat_cols
        self.input_len = input_len
        self.state2id = state2id
        self.comm2id = comm2id
        self.flow2id = flow2id
        self.target_time_start = target_time_start
        self.target_time_end = target_time_end

        # Build sequences
        self.samples = self._build_sequences()

    def _build_sequences(self) -> List[Dict[str, Any]]:
        """Build sequences from DataFrame, filtered by target-time range."""
        samples = []

        grouped = self.df.groupby(KEYS)
        for (state, comm, flow), group in grouped:
            if len(group) < self.input_len + 1:
                continue  # Skip groups that are too short

            group = group.sort_values("Time").reset_index(drop=True)
            times = group["Time"].to_numpy()

            # Create sliding windows
            for i in range(len(group) - self.input_len):
                target_idx = i + self.input_len
                target_time = times[target_idx]

                # Target-time filter — this is what makes val/test work:
                # a 5-month test set still emits samples whose 36-month
                # windows reach into train.
                if self.target_time_start is not None and target_time < self.target_time_start:
                    continue
                if self.target_time_end is not None and target_time > self.target_time_end:
                    continue

                # Input sequence
                x = group.iloc[i : i + self.input_len]

                # Target (next time step)
                y = group.iloc[target_idx]

                samples.append({
                    "x_numeric": torch.tensor(x[self.feat_cols].values, dtype=torch.float32),
                    "state_id": self.state2id[state],
                    "comm_id": self.comm2id[comm],
                    "flow_id": self.flow2id[flow],
                    "target_value": torch.tensor(y[VALUE_CANON], dtype=torch.float32),
                    "target_weight": torch.tensor(y[WEIGHT_CANON], dtype=torch.float32),
                    "target_time": pd.Timestamp(target_time),
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "x_numeric": sample["x_numeric"],
            "state_id": torch.tensor(sample["state_id"], dtype=torch.long),
            "comm_id": torch.tensor(sample["comm_id"], dtype=torch.long),
            "flow_id": torch.tensor(sample["flow_id"], dtype=torch.long),
            "target_value": sample["target_value"],
            "target_weight": sample["target_weight"],
            "target_time": sample["target_time"],
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch: List of samples

    Returns:
        Batched tensors
    """
    x_numeric = torch.stack([item["x_numeric"] for item in batch])
    state_ids = torch.stack([item["state_id"] for item in batch])
    comm_ids = torch.stack([item["comm_id"] for item in batch])
    flow_ids = torch.stack([item["flow_id"] for item in batch])
    target_value = torch.stack([item["target_value"] for item in batch])
    target_weight = torch.stack([item["target_weight"] for item in batch])

    return {
        "x_numeric": x_numeric,
        "state_ids": state_ids,
        "comm_ids": comm_ids,
        "flow_ids": flow_ids,
        "target_value": target_value,
        "target_weight": target_weight,
        # Timestamps can't be stacked into a tensor; pass through as a list.
        "target_time": [item["target_time"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Combo-window dataset for CA-only / CA+FiLM model variants.
#
# Mirrors `paper included/shared_dataloader.py::ComboWindowDataset`. Where
# `TradeDataset` produces per-series windows of shape (L, F), this dataset
# produces all-groups-at-once windows of shape (L, G, F) plus a per-group
# validity mask. Combo models need this shape so they can cross-attend
# across groups at each timestep.
#
# Same target_time_start / target_time_end filtering convention as
# TradeDataset, so val/test samples reach back into earlier splits.
# ---------------------------------------------------------------------------

class TradeComboDataset(Dataset):
    """All-groups-per-window dataset for combo-window models.

    Each sample is one window ending at ``target_time``, containing
    features for EVERY (state, commodity, flow) combination. Combinations
    that have no data at a given timestep are zero-filled and flagged
    in the mask.

    Sample dict:
        x_numeric:    (L, G, F) -- window of features per group per timestep
        group_mask:   (G,)      -- 1 if combo is valid at the target time
        target_value: (G,)      -- target Value per combo
        target_weight:(G,)      -- target Weight per combo
        target_time:  scalar    -- the target Timestamp (kept for inference)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        input_len: int,
        combo2id: Dict[tuple, int],
        target_time_start: Optional[pd.Timestamp] = None,
        target_time_end: Optional[pd.Timestamp] = None,
    ):
        self.feat_cols = feat_cols
        self.input_len = input_len
        self.combo2id = combo2id
        self.G = len(combo2id)
        self.F = len(feat_cols)

        df = df.sort_values(KEYS + ["Time"]).copy()
        self.times: List[pd.Timestamp] = sorted(df["Time"].unique().tolist())
        self.T = len(self.times)
        self.time2idx = {t: i for i, t in enumerate(self.times)}

        self.value_idx = feat_cols.index(VALUE_CANON)
        self.weight_idx = feat_cols.index(WEIGHT_CANON)

        # Pre-build (T, G, F) data tensor and (T, G) mask.
        self.data = np.zeros((self.T, self.G, self.F), dtype=np.float32)
        self.mask = np.zeros((self.T, self.G), dtype=np.float32)
        for combo_key, gdf in df.groupby(KEYS, sort=False):
            if combo_key not in combo2id:
                continue
            g_idx = combo2id[combo_key]
            gdf = gdf.sort_values("Time")
            mapped = gdf["Time"].map(self.time2idx).to_numpy(dtype=np.float64)
            valid = ~np.isnan(mapped)
            t_indices = mapped[valid].astype(np.int64)
            feats = gdf[feat_cols].to_numpy(dtype=np.float32)[valid]
            keep = np.isfinite(feats).all(axis=1)
            t_indices = t_indices[keep]
            feats = feats[keep]
            self.data[t_indices, g_idx, :] = feats
            self.mask[t_indices, g_idx] = 1.0

        # Sample indices: target time index t such that a full window
        # [t-L, t) fits AND target_time lies in the requested range.
        self.target_time_start = target_time_start
        self.target_time_end = target_time_end
        self.samples: List[int] = []
        for t in range(self.T):
            if t < self.input_len:
                continue
            tt = self.times[t]
            if target_time_start is not None and tt < target_time_start:
                continue
            if target_time_end is not None and tt > target_time_end:
                continue
            self.samples.append(t)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.samples[idx]
        x = self.data[t - self.input_len : t, :, :]   # (L, G, F)
        target_row = self.data[t, :, :]               # (G, F)
        group_mask = self.mask[t, :]                  # (G,)
        return {
            "x_numeric": torch.from_numpy(x.copy()),
            "group_mask": torch.from_numpy(group_mask.copy()),
            "target_value": torch.from_numpy(target_row[:, self.value_idx].copy()),
            "target_weight": torch.from_numpy(target_row[:, self.weight_idx].copy()),
            "target_time": self.times[t],
        }


def combo_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate combo-window samples.

    Stacks (L, G, F) -> (B, L, G, F); (G,) -> (B, G); etc. Keeps
    ``target_time`` as a list of Timestamps (not a tensor) for inference.
    """
    return {
        "x_numeric":    torch.stack([b["x_numeric"]    for b in batch]),
        "group_mask":   torch.stack([b["group_mask"]   for b in batch]),
        "target_value": torch.stack([b["target_value"] for b in batch]),
        "target_weight":torch.stack([b["target_weight"] for b in batch]),
        "target_time":  [b["target_time"] for b in batch],
    }
