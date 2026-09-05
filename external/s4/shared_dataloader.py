# shared_dataloader.py — Cleaned data utilities for Transformer models
# Shapes you get:
#   ComboWindowDataset  → x: (B, L, G, F), y: (B, G, Ft), group_mask: (B, G), x_mask_t: (B, L, G)
#   SeriesWindowDataset → x: (B, L, F),     y: (B, M, Ft), ids: dict of scalars
#
# Notes:
# - Ft = 1 if target in {"value","weight"}, Ft = 2 if "both"
# - ComboWindowDataset enforces horizon=1 (matches your current Transformer heads)
# - All masks use 1.0 = valid, 0.0 = padded (your model code already expects this)

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Constants / canonical columns
# -----------------------------
VALUE_CANON  = 'Value'
WEIGHT_CANON = 'Weight'
KEYS = ['State', 'Commodity', 'Import/Export']

EXPORT_VALUE_RAW  = 'Containerized Vessel Total Exports Value ($US)'
EXPORT_WEIGHT_RAW = 'Containerized Vessel Total Exports SWT (kg)'
IMPORT_VALUE_PRI   = 'Vessel Value ($US)'
IMPORT_VALUE_ALT   = 'Customs Containerized Vessel Value (Gen) ($US)'
IMPORT_WEIGHT_PRI  = 'Vessel SWT (kg)'
IMPORT_WEIGHT_ALT  = 'Containerized Vessel SWT (Gen) (kg)'

MONTH_MAP = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

# -----------------------------
# Parsing & cleaning helpers
# -----------------------------
def parse_month(s) -> pd.Timestamp:
    """Best-effort month parser; returns first day-of-month or NaT."""
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
        yy, mon = s.split('-')
        return pd.Timestamp(int(yy) + 2000, MONTH_MAP[mon.capitalize()], 1)
    except Exception:
        pass
    # Fallback generic
    try:
        dt = pd.to_datetime(s)
        return pd.Timestamp(dt.year, dt.month, 1)
    except Exception:
        return pd.NaT

def _numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)

# -----------------------------
# IO & panel construction
# -----------------------------
def read_folder(folder: str, label: str) -> pd.DataFrame:
    files = sorted([str(p) for p in Path(folder).glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp, on_bad_lines='skip', dtype=str)

        # Required key columns
        for k in ['State','Commodity','Country','Time']:
            if k not in df.columns:
                raise ValueError(f"{fp} missing column: {k}")
        df['Time'] = df['Time'].apply(parse_month)

        if label.lower().startswith('export'):
            for c in [EXPORT_VALUE_RAW, EXPORT_WEIGHT_RAW]:
                if c not in df.columns:
                    raise ValueError(f"{fp} missing export column: {c}")
            df[EXPORT_VALUE_RAW]  = _numeric_clean(df[EXPORT_VALUE_RAW])
            df[EXPORT_WEIGHT_RAW] = _numeric_clean(df[EXPORT_WEIGHT_RAW])
            df = df.rename(columns={EXPORT_VALUE_RAW: VALUE_CANON, EXPORT_WEIGHT_RAW: WEIGHT_CANON})
        else:
            vcol = IMPORT_VALUE_PRI  if IMPORT_VALUE_PRI  in df.columns else IMPORT_VALUE_ALT
            wcol = IMPORT_WEIGHT_PRI if IMPORT_WEIGHT_PRI in df.columns else IMPORT_WEIGHT_ALT
            for c in [vcol, wcol]:
                if c not in df.columns:
                    raise ValueError(f"{fp} missing import column: {c}")
            df[vcol] = _numeric_clean(df[vcol])
            df[wcol] = _numeric_clean(df[wcol])
            df = df.rename(columns={vcol: VALUE_CANON, wcol: WEIGHT_CANON})

        df['Import/Export'] = 'Import' if label.lower().startswith('import') else 'Export'
        keep = ['State','Commodity','Country','Time','Import/Export', VALUE_CANON, WEIGHT_CANON]
        dfs.append(df[keep])

    return pd.concat(dfs, ignore_index=True)

def build_grid(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Aggregate to monthly, create full grid on KEYS × Time with zeros for missing."""
    num_cols = [c for c in df.columns if c not in KEYS + ['Country','Time']]
    grouped = (
        df.groupby(KEYS + ['Time'], dropna=False)[num_cols]
          .sum()
          .reset_index()
    )

    # full monthly timeline
    full_time = pd.date_range(start=start, end=end, freq="MS")
    time_df = pd.DataFrame({"Time": full_time, "key": 1})
    unique_keys = grouped[KEYS].drop_duplicates().assign(key=1)

    grid = unique_keys.merge(time_df, on="key", how="outer").drop(columns="key")
    grid = grid.merge(grouped, on=KEYS+['Time'], how="left")

    for c in num_cols:
        grid[c] = grid[c].fillna(0.0)

    # Sparse-combo filter — keep only (State, Commodity, Flow) combos whose
    # RAW Value is nonzero in >= MIN_NONZERO_FRAC of months. Mirrors the github
    # TradeDataPipeline.filter_sparse_combos so S4/S4ND train on the SAME
    # kept-combo population as every other model. Set S4_MIN_NONZERO_FRAC=0 to
    # disable (e.g. for a paper-style all-combos run).
    import os as _os
    thr = float(_os.environ.get("S4_MIN_NONZERO_FRAC", "0.95"))
    if thr > 0:
        nz = grid.groupby(KEYS)[VALUE_CANON].apply(lambda s: float((s != 0).mean()))
        keep = nz[nz >= thr].index.to_frame(index=False)
        before = nz.shape[0]
        grid = grid.merge(keep, on=KEYS, how="inner")
        print(f"[shared_dataloader] sparse filter >= {thr:.0%} nonzero raw Value: "
              f"kept {len(keep)}/{before} combos, dropped {before - len(keep)}")

    # Month encodings
    grid['month'] = grid['Time'].dt.month
    grid['sin_month'] = np.sin(2*np.pi*grid['month']/12)
    grid['cos_month'] = np.cos(2*np.pi*grid['month']/12)
    return grid

def temporal_split_calendar(grid: pd.DataFrame, cfg):
    """Return (train_df, val_df, test_df) using calendar boundaries from cfg."""
    t_train_end  = pd.Timestamp(cfg.split_train_end)
    t_val_start  = pd.Timestamp(cfg.split_val_start)
    t_val_end    = pd.Timestamp(cfg.split_val_end)
    t_test_start = pd.Timestamp(cfg.split_test_start)
    t_test_end   = pd.Timestamp(cfg.split_test_end)

    tr = grid[grid['Time'] <= t_train_end].copy()
    va = grid[(grid['Time'] >= t_val_start) & (grid['Time'] <= t_val_end)].copy()
    te = grid[(grid['Time'] >= t_test_start) & (grid['Time'] <= t_test_end)].copy()

    print(f"Train ≤ {t_train_end.date()} | Val {t_val_start.date()}→{t_val_end.date()} | Test {t_test_start.date()}→{t_test_end.date()}")
    return tr, va, te

# -----------------------------
# Normalization (per-combo min-max)
# -----------------------------
def compute_series_stats(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Per-combo min/max from training data only. No hierarchical backoff."""
    s_exact = (
        train_df.groupby(['State','Commodity','Import/Export'])[[VALUE_CANON, WEIGHT_CANON]]
                .agg(['min','max']).reset_index()
    )
    s_exact.columns = ['State','Commodity','Import/Export','val_min','val_max','wt_min','wt_max']
    for col in ('val_min','val_max','wt_min','wt_max'):
        s_exact[col] = s_exact[col].fillna(0.0)
    return {'exact': s_exact}

def apply_stats_with_backoff(df: pd.DataFrame, stats: Dict[str, Any], clip: float = None) -> pd.DataFrame:
    """Apply per-combo min-max normalization: x_norm = (x - min) / (max - min).
    clip parameter kept for API compatibility but ignored (MinMax needs no clipping).
    """
    out = df.sort_values(KEYS + ['Time']).copy()
    out = out.merge(stats['exact'], on=KEYS, how='left')

    val_range = (out['val_max'] - out['val_min']).replace(0, 1.0)
    wt_range  = (out['wt_max']  - out['wt_min']).replace(0, 1.0)

    out[VALUE_CANON]  = (out[VALUE_CANON]  - out['val_min']) / val_range
    out[WEIGHT_CANON] = (out[WEIGHT_CANON] - out['wt_min'])  / wt_range

    out = out.drop(columns=['val_min','val_max','wt_min','wt_max'])
    return out

def _sanity_check_normalization(df: pd.DataFrame, tag: str):
    print(f"[{tag}] minmax ~ Value [{df[VALUE_CANON].min():.3f}, {df[VALUE_CANON].max():.3f}]"
          f" | Weight [{df[WEIGHT_CANON].min():.3f}, {df[WEIGHT_CANON].max():.3f}]")

# -----------------------------
# Feature engineering
# -----------------------------
def add_lags(df: pd.DataFrame, n_lags: int = 12) -> pd.DataFrame:
    df = df.sort_values(KEYS + ['Time']).copy()
    for k in range(1, n_lags + 1):
        df[f'value_lag_{k}']  = df.groupby(KEYS)[VALUE_CANON].shift(k).fillna(0.0)
        df[f'weight_lag_{k}'] = df.groupby(KEYS)[WEIGHT_CANON].shift(k).fillna(0.0)
    # Burn-in year (optional): drop 2008 if desired
    df = df[df['Time'].dt.year != 2008]
    return df

def feature_view(df: pd.DataFrame):
    core_feats = [VALUE_CANON, WEIGHT_CANON] \
               + [f'value_lag_{k}' for k in range(1,13)] \
               + [f'weight_lag_{k}' for k in range(1,13)] \
               + ['sin_month','cos_month']
    feat_cols = core_feats
    return df[KEYS + ['Time'] + feat_cols].copy(), feat_cols, core_feats

# -----------------------------
# Mappings
# -----------------------------
def build_combo_maps(panel: pd.DataFrame):
    """(State, Commodity, Import/Export) → id mappings."""
    unique_combos = panel[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    combo2id, id2combo = {}, {}
    for i, row in unique_combos.iterrows():
        combo = (row['State'], row['Commodity'], row['Import/Export'])
        combo2id[combo] = i
        id2combo[i] = combo
    return combo2id, id2combo, len(unique_combos)

def build_id_maps(panel: pd.DataFrame):
    states = sorted(panel['State'].dropna().unique().tolist())
    comms  = sorted(panel['Commodity'].dropna().unique().tolist())
    flows  = ['Export', 'Import']  # stable order
    state2id = {s:i for i,s in enumerate(states)}
    comm2id  = {c:i for i,c in enumerate(comms)}
    flow2id  = {f:i for i,f in enumerate(flows)}
    return state2id, comm2id, flow2id

# -----------------------------
# Config & loaders
# -----------------------------
@dataclass
class SharedConfig:
    """Shared configuration for Transformer runs."""
    imports_dir: str = "data/imports"
    exports_dir: str = "data/exports"
    start: str = "2008-01-01"
    end:   str = "2025-05-01"
    input_len: int = 36
    horizon: int = 1
    batch_size: int = 4
    epochs: int = 200
    lr: float = 1e-3
    hidden_size: int = 128
    layers: int = 4
    dropout: float = 0.0
    target: str = "both"  # "value" | "weight" | "both"
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Split dates
    split_train_end: str = "2023-12-01"
    split_val_start: str = "2024-01-01"
    split_val_end:   str = "2024-12-01"
    split_test_start: str = "2025-01-01"
    split_test_end:   str = "2025-05-01"

    # Early stopping
    patience: int = 10

def create_data_loaders(cfg, df_train, df_val, df_test, feat_cols, dataset_class, **dataset_kwargs):
    """Create train/val/test DataLoaders with consistent settings."""
    ds_tr = dataset_class(df_train, feat_cols, cfg.input_len, cfg.horizon, cfg.target, **dataset_kwargs)
    ds_va = dataset_class(df_val,   feat_cols, cfg.input_len, cfg.horizon, cfg.target, **dataset_kwargs)
    ds_te = dataset_class(df_test,  feat_cols, cfg.input_len, cfg.horizon, cfg.target, **dataset_kwargs)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  drop_last=True,  num_workers=cfg.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True)
    return dl_tr, dl_va, dl_te

# -----------------------------
# Artifact save/load
# -----------------------------
def save_artifacts(out_dir: str,
                   stats: Dict[str, Any],
                   combo2id: Dict[tuple, int],
                   id2combo: Dict[int, tuple],
                   model: nn.Module,
                   feat_cols: List[str],
                   tr_z: pd.DataFrame = None,
                   va_z: pd.DataFrame = None):
    art = Path(out_dir) / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    stats_json = {
        "exact":  stats["exact"].to_dict(orient="list"),
        "cf":     stats["cf"].to_dict(orient="list"),
        "c":      stats["c"].to_dict(orient="list"),
        "global": stats["global"],
    }
    (art / "series_stats.json").write_text(json.dumps(stats_json, indent=2))

    combo2id_serializable = {str(k): v for k, v in combo2id.items()}
    id2combo_serializable = {str(k): v for k, v in id2combo.items()}
    maps = {"combo2id": combo2id_serializable, "id2combo": id2combo_serializable, "num_combos": len(combo2id)}
    (art / "combo_maps.json").write_text(json.dumps(maps, ensure_ascii=False, indent=2))

    (art / "features.json").write_text(json.dumps({"feat_cols": feat_cols}, indent=2))

    if tr_z is not None:
        tr_z.to_csv(art / "zscores_train.csv", index=False)
    if va_z is not None:
        va_z.to_csv(art / "zscores_val.csv", index=False)

    print(f"📦 Saved artifacts to {art}")

def load_artifacts_for_eval(cfg, out_dir: str, features_per_group_hint: int = None):
    """Rebuild TransformerWithCrossAttention with exact combo maps; load stats."""
    art = Path(out_dir) / "artifacts"

    maps = json.loads((art / "combo_maps.json").read_text())
    combo2id = {eval(k): v for k, v in maps["combo2id"].items()}
    id2combo = {int(k): v for k, v in maps["id2combo"].items()}
    num_combos = maps["num_combos"]

    feat_cols = json.loads((art / "features.json").read_text())["feat_cols"]
    features_per_group = len(feat_cols) if features_per_group_hint is None else features_per_group_hint

    out_dim = 2 if cfg.target == "both" else 1
    from Transformer_model import TransformerWithCrossAttention
    model = TransformerWithCrossAttention(
        num_groups=num_combos,
        features_per_group=features_per_group,
        hidden_size=cfg.hidden_size,
        output_dim=out_dim,
        layers=cfg.layers,
        dropout=cfg.dropout,
        attn_dropout=getattr(cfg, 'attn_dropout', 0.0),
        num_latents=getattr(cfg, 'num_latents', 0),
        film_hidden=getattr(cfg, 'film_hidden', 128),
        gate_temperature=getattr(cfg, 'gate_temperature', 1.0),
        min_gate=getattr(cfg, 'gate_min', 0.05),
        normalize_group_weights=getattr(cfg, 'normalize_group_weights', False),
        nhead=getattr(cfg, 'nhead', 8),
    )

    stats_path_json = art / "series_stats.json"
    if stats_path_json.exists():
        raw = json.loads(stats_path_json.read_text())
        stats = {
            "exact":  pd.DataFrame(raw["exact"]),
            "cf":     pd.DataFrame(raw["cf"]),
            "c":      pd.DataFrame(raw["c"]),
            "global": raw["global"],
        }
    else:
        # Backward-compat: exact CSV only
        exact_csv = art / "series_stats.csv"
        if not exact_csv.exists():
            raise FileNotFoundError("No stats found (series_stats.json or series_stats.csv).")
        exact = pd.read_csv(exact_csv)
        g_val_mean = float(exact["val_mean"].mean())
        g_val_std  = float(exact["val_std"].replace(0, np.nan).mean() or 1.0)
        g_wt_mean  = float(exact["wt_mean"].mean())
        g_wt_std   = float(exact["wt_std"].replace(0, np.nan).mean() or 1.0)
        stats = {
            "exact": exact,
            "cf":    pd.DataFrame(columns=["Commodity","Import/Export","val_mean","val_std","wt_mean","wt_std"]),
            "c":     pd.DataFrame(columns=["Commodity","val_mean","val_std","wt_mean","wt_std"]),
            "global": {"val_mean": g_val_mean, "val_std": g_val_std, "wt_mean": g_wt_mean, "wt_std": g_wt_std},
        }

    return model, (combo2id, id2combo, feat_cols), stats

# -----------------------------
# Datasets
# -----------------------------
class ComboWindowDataset(Dataset):
    """
    Windowed ALL-combo dataset (time-major windows).
    Returns one sample per target month containing ALL groups.

    x: (L, G, F)  -> batches as (B, L, G, F)
    y: (G, Ft)    with Ft∈{1,2} depending on 'target'
    group_mask: (G,) with 1.0 if combo present at target t, else 0.0
    x_mask_t: (L, G) with 1.0 where combo is present in that month, else 0.0

    Note: horizon is enforced to be 1 (consistent with current Transformer heads).
    """
    def __init__(self, df_feat: pd.DataFrame, feat_cols: List[str],
                 input_len: int, horizon: int, target: str,
                 combo2id: Dict[tuple, int], num_combos: int,
                 target_time_start: pd.Timestamp = None,
                 target_time_end: pd.Timestamp = None):
        assert target in ("value","weight","both")
        self.df = df_feat
        self.feat_cols = feat_cols
        self.L = int(input_len)
        self.M = int(horizon)
        self.G = int(num_combos)
        self.target = target

        if self.M != 1:
            raise ValueError("ComboWindowDataset currently supports horizon=1 to match model output_dim.")

        self.combo2id = combo2id
        self.target_time_start = target_time_start
        self.target_time_end   = target_time_end

        # Build global time index (exclude NaT)
        unique_times = [t for t in self.df['Time'].unique() if pd.notna(t)]
        self.times = sorted(unique_times)
        if not self.times:
            raise ValueError("No valid Time values found after parsing.")

        # Indices of samples (target months)
        self.samples: List[int] = []

        self.value_idx  = self.feat_cols.index(VALUE_CANON)
        self.weight_idx = self.feat_cols.index(WEIGHT_CANON)

        for i, target_time in enumerate(self.times):
            if i < self.L:
                continue  # not enough history
            if self.target_time_start is not None and target_time < self.target_time_start:
                continue
            if self.target_time_end is not None and target_time > self.target_time_end:
                continue
            self.samples.append(i)

        # Pre-build dense panel tensors
        self.data_tensor = self._build_data_tensor()

    def _build_data_tensor(self):
        """Build dense panel: data -> (T, G, F), mask -> (T, G)"""
        T = len(self.times)
        F = len(self.feat_cols)
        data = np.zeros((T, self.G, F), dtype=np.float32)
        mask = np.zeros((T, self.G), dtype=np.float32)

        time_to_idx = {t: i for i, t in enumerate(self.times)}

        for combo_key, combo_df in self.df.groupby(KEYS, sort=False):
            g_idx = self.combo2id.get(combo_key, None)
            if g_idx is None:
                continue
            combo_df = combo_df.sort_values('Time')
            for _, row in combo_df.iterrows():
                t = row['Time']
                if pd.isna(t):
                    continue
                t_idx = time_to_idx.get(t, None)
                if t_idx is None:
                    continue
                feats = row[self.feat_cols].values.astype(np.float32)
                if np.isfinite(feats).all():
                    data[t_idx, g_idx, :] = feats
                    mask[t_idx, g_idx] = 1.0

        return data, mask

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, i):
        target_time_idx = self.samples[i]
        start_idx = target_time_idx - self.L

        data, mask = self.data_tensor  # (T,G,F), (T,G)

        # Inputs: [t-L, ..., t-1]
        x_data = data[start_idx:target_time_idx, :, :]   # (L, G, F)
        x_mask = mask[start_idx:target_time_idx, :]      # (L, G)

        # Target: at time t
        y_data = data[target_time_idx, :, :]             # (G, F)
        y_mask = mask[target_time_idx, :]                # (G,)

        # Build y according to target type
        if self.target == "value":
            y = y_data[:, [self.value_idx]].astype(np.float32)   # (G,1)
        elif self.target == "weight":
            y = y_data[:, [self.weight_idx]].astype(np.float32)  # (G,1)
        else:
            y = y_data[:, [self.value_idx, self.weight_idx]].astype(np.float32)  # (G,2)

        # Asserts for sanity (catch shape drift early)
        L, G, F = x_data.shape
        assert L == self.L and G == self.G and F == len(self.feat_cols)
        assert y.shape[0] == self.G
        assert x_mask.shape == (self.L, self.G)

        # Return time-major; DataLoader stacks to (B, L, G, F)
        x = x_data
        x_mask_t = x_mask
        group_mask = y_mask.astype(np.float32)

        return (torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(group_mask),
                torch.from_numpy(x_mask_t))

class SeriesWindowDataset(Dataset):
    """
    Per-series sliding windows (for flat/Plan A models).
    Returns:
      x: (L, F)
      y: (M, Ft) with Ft in {1,2}
      ids: {'state_id','comm_id','flow_id'} (scalars)
    """
    def __init__(self, df_feat: pd.DataFrame, feat_cols: List[str],
                 input_len: int, horizon: int, target: str,
                 state2id: Dict[str,int], comm2id: Dict[str,int], flow2id: Dict[str,int],
                 target_time_start: pd.Timestamp = None,
                 target_time_end: pd.Timestamp = None):
        assert target in ("value","weight","both")
        self.df = df_feat
        self.feat_cols = feat_cols
        self.L = int(input_len)
        self.M = int(horizon)
        self.target = target

        self.state2id = state2id
        self.comm2id  = comm2id
        self.flow2id  = flow2id

        self.target_time_start = target_time_start
        self.target_time_end   = target_time_end

        # Group by series and build samples
        self.groups = dict(tuple(self.df.groupby(KEYS, sort=False)))
        self.samples: List[Tuple[tuple,int]] = []
        self.cache_X: Dict[tuple, np.ndarray] = {}
        self.cache_Y: Dict[tuple, np.ndarray] = {}

        for key, gdf in self.groups.items():
            gdf = gdf.sort_values('Time')
            X = gdf[self.feat_cols].to_numpy(dtype=np.float32)  # (T, F)
            if not np.isfinite(X).all():
                continue  # skip broken series
            self.cache_X[key] = X
            # Cache canonical targets separately from features
            Y = gdf[[VALUE_CANON, WEIGHT_CANON]].to_numpy(dtype=np.float32)  # (T, 2)
            self.cache_Y[key] = Y
            times = gdf['Time'].to_numpy()
            T_len = X.shape[0]
            max_start = T_len - (self.L + self.M)
            for t0 in range(max_start + 1):
                target_idx = t0 + self.L  # first target timestep
                if 0 <= target_idx < T_len and (self.target_time_start is not None or self.target_time_end is not None):
                    t_target = times[target_idx]
                    if self.target_time_start is not None and t_target < self.target_time_start:
                        continue
                    if self.target_time_end is not None and t_target > self.target_time_end:
                        continue
                self.samples.append((key, t0))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, i):
        key, t0 = self.samples[i]
        X = self.cache_X[key]
        Y = self.cache_Y[key]
        x = X[t0 : t0 + self.L, :]                        # (L, F)
        # Targets drawn from canonical columns, independent of feature set
        y_slice = Y[t0 + self.L : t0 + self.L + self.M, :]# (M, 2)

        if self.target == "value":
            y = y_slice[:, [0]]                           # (M, 1)
        elif self.target == "weight":
            y = y_slice[:, [1]]                           # (M, 1)
        else:
            y = y_slice[:, [0, 1]]                      # (M, 2)

        state_id = self.state2id.get(key[0], 0)
        comm_id  = self.comm2id.get(key[1], 0)
        flow_id  = self.flow2id.get(key[2], 0)
        ids = {
            "state_id": torch.tensor(state_id, dtype=torch.long),
            "comm_id":  torch.tensor(comm_id,  dtype=torch.long),
            "flow_id":  torch.tensor(flow_id,  dtype=torch.long),
        }
        return torch.from_numpy(x), torch.from_numpy(y), ids

# -----------------------------
# Convenience helpers
# -----------------------------
def choose_top_k_series(panel: pd.DataFrame,
                        k: Optional[int] = 5,
                        commodity: Optional[str] = None,
                        state: Optional[str] = None,
                        flow: Optional[str] = None,
                        start: Optional[pd.Timestamp] = None,
                        end: Optional[pd.Timestamp] = None) -> List[Tuple[str,str,str]]:
    df = panel.copy()
    if start is not None:
        df = df[df["Time"] >= start]
    if end is not None:
        df = df[df["Time"] <= end]
    if commodity:
        df = df[df["Commodity"] == commodity]
    if state:
        df = df[df["State"] == state]
    if flow:
        df = df[df["Import/Export"] == flow]
    if df.empty:
        raise ValueError("No rows after applying filters; check commodity/state/flow/start/end.")

    grp = (df.groupby(["Commodity","State","Import/Export"])["Time"]
             .count().reset_index().rename(columns={"Time":"n"}))
    grp = grp.sort_values("n", ascending=False)
    if k is not None and k > 0:
        grp = grp.head(k)
    return [(str(r["Commodity"]), str(r["State"]), str(r["Import/Export"])) for _, r in grp.iterrows()]

def get_series_stats_for_exact(stats: Dict[str, Any], commodity: str, state: str, flow: str) -> Dict[str, float]:
    exact = stats["exact"]
    row = exact[(exact["State"]==state) & (exact["Commodity"]==commodity) & (exact["Import/Export"]==flow)]
    if len(row) == 1:
        row = row.iloc[0]
        return dict(val_mean=float(row["val_mean"]), val_std=float(row["val_std"]) or 1.0,
                    wt_mean=float(row["wt_mean"]),   wt_std=float(row["wt_std"])   or 1.0)

    cf = stats["cf"]
    row = cf[(cf["Commodity"]==commodity) & (cf["Import/Export"]==flow)]
    if len(row) == 1:
        row = row.iloc[0]
        return dict(val_mean=float(row["val_mean"]), val_std=float(row["val_std"]) or 1.0,
                    wt_mean=float(row["wt_mean"]),   wt_std=float(row["wt_std"])   or 1.0)

    c = stats["c"]
    row = c[(c["Commodity"]==commodity)]
    if len(row) == 1:
        row = row.iloc[0]
        return dict(val_mean=float(row["val_mean"]), val_std=float(row["val_std"]) or 1.0,
                    wt_mean=float(row["wt_mean"]),   wt_std=float(row["wt_std"])   or 1.0)

    g = stats["global"]
    return dict(val_mean=float(g["val_mean"]), val_std=float(g["val_std"]) or 1.0,
                wt_mean=float(g["wt_mean"]),   wt_std=float(g["wt_std"])   or 1.0)

def build_flat_datasets(cfg) -> Tuple[SeriesWindowDataset, SeriesWindowDataset, SeriesWindowDataset, List[str], Dict[str,int], Dict[str,int], Dict[str,int]]:
    """Create train/val/test SeriesWindowDataset + metadata using cfg fields (mirrors DSS/GPT flat).
    Expected cfg attributes: imports_dir, exports_dir, start, end, input_len, horizon, target,
    split_* dates. Returns (ds_train, ds_val, ds_test, feat_cols, state2id, comm2id, flow2id)."""
    imports = read_folder(cfg.imports_dir, "Import")
    exports = read_folder(cfg.exports_dir, "Export")
    panel = pd.concat([imports, exports], ignore_index=True)
    state2id, comm2id, flow2id = build_id_maps(panel)

    grid = build_grid(panel, cfg.start, cfg.end)
    tr_raw, va_raw, te_raw = temporal_split_calendar(grid, cfg)

    stats = compute_series_stats(tr_raw)
    tr = apply_stats_with_backoff(tr_raw, stats)
    va = apply_stats_with_backoff(va_raw, stats)
    te = apply_stats_with_backoff(te_raw, stats)

    tr = add_lags(tr, 12)
    va = add_lags(va, 12)
    te = add_lags(te, 12)

    trf, feat_cols, _ = feature_view(tr)
    vaf, _, _ = feature_view(va)
    tef, _, _ = feature_view(te)

    t_val_start = pd.Timestamp(cfg.split_val_start)
    t_val_end   = pd.Timestamp(cfg.split_val_end)
    t_test_start= pd.Timestamp(cfg.split_test_start)
    t_test_end  = pd.Timestamp(cfg.split_test_end)

    tr_va_concat = pd.concat([trf, vaf], ignore_index=True)

    ds_train = SeriesWindowDataset(
        trf, feat_cols, cfg.input_len, cfg.horizon, cfg.target,
        state2id, comm2id, flow2id,
        target_time_start=None,
        target_time_end=pd.Timestamp(cfg.split_train_end)
    )
    ds_val = SeriesWindowDataset(
        tr_va_concat, feat_cols, cfg.input_len, cfg.horizon, cfg.target,
        state2id, comm2id, flow2id,
        target_time_start=t_val_start,
        target_time_end=t_val_end
    )
    ds_test = SeriesWindowDataset(
        pd.concat([trf, vaf, tef], ignore_index=True),
        feat_cols, cfg.input_len, cfg.horizon, cfg.target,
        state2id, comm2id, flow2id,
        target_time_start=t_test_start,
        target_time_end=t_test_end
    )
    return ds_train, ds_val, ds_test, feat_cols, state2id, comm2id, flow2id


def build_flat_loaders(cfg, dataset_class=SeriesWindowDataset):
    """Return train/val/test DataLoaders using build_flat_datasets; kept for API parity."""
    ds_tr, ds_va, ds_te, feat_cols, state2id, comm2id, flow2id = build_flat_datasets(cfg)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  drop_last=True,  num_workers=cfg.num_workers)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    meta = {
        "feat_cols": feat_cols,
        "state2id": state2id,
        "comm2id": comm2id,
        "flow2id": flow2id,
    }
    return dl_tr, dl_va, dl_te, meta

def build_flat_dataloaders(cfg):
    ds_tr, ds_va, ds_te, feat_cols, state2id, comm2id, flow2id = build_flat_datasets(cfg)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    meta = dict(feat_cols=feat_cols, state2id=state2id, comm2id=comm2id, flow2id=flow2id)
    return dl_tr, dl_va, dl_te, meta

def train_flat_model(cfg, model_builder, optimizer_builder, loss_fn, device=None, max_epochs=None, early_stopping_patience=10):
    device = torch.device(device or cfg.device)
    dl_tr, dl_va, dl_te, meta = build_flat_dataloaders(cfg)
    model = model_builder(meta).to(device)
    optimizer = optimizer_builder(model)
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    for epoch in range(1, (max_epochs or cfg.epochs) + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, ids in dl_tr:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            preds = model(xb, ids)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(dl_tr))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, ids in dl_va:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                preds = model(xb, ids)
                val_loss += loss_fn(preds, yb).item()
        val_loss /= max(1, len(dl_va))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for xb, yb, ids in dl_te:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            preds = model(xb, ids)
            test_loss += loss_fn(preds, yb).item()
    test_loss /= max(1, len(dl_te))

    return {
        "model": model,
        "metadata": meta,
        "best_val_loss": best_val,
        "test_loss": test_loss,
    }