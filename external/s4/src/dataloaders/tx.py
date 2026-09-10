import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataloaders.base import SequenceDataset

# Constants from original project
VALUE_CANON = "z_value"
WEIGHT_CANON = "z_weight"


# Data loading and preprocessing functions adapted from shared_dataloader.py

def read_folder(folder_path, flow_type):
    dfs = []
    for f in Path(folder_path).glob("*.csv"):
        df = pd.read_csv(f)

        # Normalize flow column name and values
        df["Import/Export"] = str(flow_type)

        # Normalize Time column from formats like '8-Jan' -> 2008-01-01, else try generic parsing
        if "Time" in df.columns:
            def _parse_time(x):
                try:
                    s = str(x).strip()
                    # Match forms like '8-Jan', '09-Feb', etc. Assume YY-Mon -> 20YY-Mon-01
                    if pd.isna(x):
                        return pd.NaT
                    if len(s) >= 5 and "-" in s and s.split("-")[0].isdigit():
                        yy, mon = s.split("-")[0], s.split("-")[1][:3]
                        yy_i = int(yy)
                        year = 2000 + yy_i if yy_i < 100 else yy_i
                        return pd.Timestamp(f"{year}-{mon}-01")
                    # Fallback to pandas parser
                    return pd.to_datetime(s, errors="coerce")
                except Exception:
                    return pd.NaT
            df["Time"] = df["Time"].apply(_parse_time)
        else:
            # If no Time column, skip this file
            continue

        # Create canonical Value/Weight if missing (from vessel columns)
        if "Value" not in df.columns:
            for cand in ["Vessel Value ($US)", "Customs Containerized Vessel Value (Gen) ($US)"]:
                if cand in df.columns:
                    df["Value"] = pd.to_numeric(df[cand].astype(str).str.replace(",", ""), errors="coerce")
                    break
        if "Weight" not in df.columns:
            for cand in ["Vessel SWT (kg)", "Containerized Vessel SWT (Gen) (kg)"]:
                if cand in df.columns:
                    df["Weight"] = pd.to_numeric(df[cand].astype(str).str.replace(",", ""), errors="coerce")
                    break

        # Keep only necessary columns
        cols_needed = ["State", "Commodity", "Import/Export", "Time", "Value", "Weight"]
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            # Skip if required columns unavailable
            continue

        df = df[cols_needed].dropna(subset=["Time"]).reset_index(drop=True)
        # Ensure categorical/string types
        df["Import/Export"] = df["Import/Export"].astype(str)
        df["State"] = df["State"].astype(str)
        df["Commodity"] = df["Commodity"].astype(str)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No usable CSV files found in {folder_path}")
    return pd.concat(dfs, ignore_index=True)

def build_grid(panel_df, start_date, end_date):
    date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq="MS"))
    commodities = panel_df["Commodity"].unique()
    states = panel_df["State"].unique()
    flows = panel_df["Import/Export"].unique()
    
    grid = pd.MultiIndex.from_product([commodities, states, flows, date_range], 
                                      names=["Commodity", "State", "Import/Export", "Time"])
    grid_df = pd.DataFrame(index=grid).reset_index()

    # Aggregate duplicate (combo, month) sub-records by SUM before merging.
    # The raw panel carries multiple distinct sub-records per (combo, month)
    # (different values, not copies); summing reconstructs the true monthly
    # total, matching the github pipeline. Without this the grid would carry
    # many interleaved rows per timestamp and corrupt every downstream series.
    panel_df = (
        panel_df.groupby(["Commodity", "State", "Import/Export", "Time"], as_index=False)[["Value", "Weight"]]
        .sum()
    )

    merged = pd.merge(grid_df, panel_df, on=["Commodity", "State", "Import/Export", "Time"], how="left")
    merged[["Value", "Weight"]] = merged[["Value", "Weight"]].fillna(0)
    return merged.sort_values(by=["Commodity", "State", "Import/Export", "Time"]).reset_index(drop=True)

KEYS = ["State", "Commodity", "Import/Export"]

def compute_series_stats(df):
    """Compute per-combo min/max from training data only (vectorized, no backoff)."""
    agg = df.groupby(KEYS)[["Value", "Weight"]].agg(["min", "max"]).reset_index()
    agg.columns = KEYS + ["val_min", "val_max", "wt_min", "wt_max"]
    for col in ("val_min", "val_max", "wt_min", "wt_max"):
        agg[col] = agg[col].fillna(0.0)
    return agg

def apply_stats_with_backoff(df, stats):
    """Apply per-combo minmax normalization via merge (vectorized, consistent with dataloader.py)."""
    out = df.merge(stats, on=KEYS, how="left")
    val_range = (out["val_max"] - out["val_min"]).replace(0, 1.0)
    wt_range  = (out["wt_max"]  - out["wt_min"]).replace(0, 1.0)
    out["z_value"]  = (out["Value"]  - out["val_min"]) / val_range
    out["z_weight"] = (out["Weight"] - out["wt_min"])  / wt_range
    out = out.drop(columns=["val_min", "val_max", "wt_min", "wt_max"])
    return out

def add_time_features(df):
    df["month"] = df["Time"].dt.month.astype(float) / 12.0
    df["year"] = (df["Time"].dt.year - 2008.0) / (2025.0 - 2008.0)
    return df

def add_lags(df, n_lags):
    df_out = df.copy()
    for lag in range(1, n_lags + 1):
        df_out[f"z_value_lag{lag}"] = df_out.groupby(["Commodity", "State", "Import/Export"])["z_value"].shift(lag)
        df_out[f"z_weight_lag{lag}"] = df_out.groupby(["Commodity", "State", "Import/Export"])["z_weight"].shift(lag)
    return df_out.fillna(0)


def feature_view(df):
    feat_cols = (
        ["month", "year"] +
        [c for c in df.columns if "lag" in c]
    )
    return df, feat_cols

# Custom Dataset for this task
class TradeSeriesWindowDataset(Dataset):
    def __init__(self, data, feat_cols, input_len, horizon, target_cols, encoding='flat'):
        self.data = data
        self.feat_cols = feat_cols
        self.input_len = input_len
        self.horizon = horizon
        # For S4 forecasting task expectations
        self.forecast_horizon = horizon
        self.target_cols = target_cols
        self.encoding = encoding
        
        self.windows = []
        # Creating windows is slow, so we do it once at init
        for _, g in self.data.groupby(["Commodity", "State", "Import/Export"]):
            ts_data = g[self.feat_cols].to_numpy(dtype=np.float32)
            ts_target = g[self.target_cols].to_numpy(dtype=np.float32)
            state_id = g['state_id'].iloc[0]
            comm_id = g['comm_id'].iloc[0]
            flow_id = g['flow_id'].iloc[0]
            
            if len(g) >= input_len + horizon:
                for i in range(len(g) - input_len - horizon + 1):
                    window_features = ts_data[i : i + input_len]
                    window_target = ts_target[i + input_len : i + input_len + horizon]
                    self.windows.append((window_features, window_target, state_id, comm_id, flow_id))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


class TradeDatasetBase(SequenceDataset):
    _name_ = "trade_base"
    
    @property
    def init_defaults(self):
        return {
            "input_len": 36,
            "horizon": 1,
            "target": "both",  # 'both' | 'value' | 'weight'
            "data_dir": "data",
            "encoding": "flat", # 'flat' or 'onehot'
        }

    def setup(self):
        if getattr(self, '_is_setup', False):
            return
        self._is_setup = True
        self.l_output = 0  # tell SequenceDecoder to take last step only
        # 1. Load and preprocess data
        imports_path = Path(self.data_dir) / "imports"
        exports_path = Path(self.data_dir) / "exports"
        if not imports_path.exists() or not exports_path.exists():
            print(f"[tx] Warning: expected data dirs not found: imports={imports_path}, exports={exports_path}")
            # Fallback to /root/data if available
            fallback = Path('/root/data')
            if fallback.exists():
                imports_path = fallback / 'imports'
                exports_path = fallback / 'exports'
                print(f"[tx] Falling back to {imports_path} and {exports_path}")

        imports = read_folder(imports_path, "Import")
        exports = read_folder(exports_path, "Export")
        panel = pd.concat([imports, exports], ignore_index=True)
        
        # 2. Build grid
        grid = build_grid(panel, "2008-01-01", "2025-05-01")
        
        # 3. Create categorical mappings
        self.states = sorted(grid["State"].unique())
        self.commodities = sorted(grid["Commodity"].unique())
        self.flows = sorted(grid["Import/Export"].unique())

        self.state_to_id = {s: i for i, s in enumerate(self.states)}
        self.comm_to_id = {c: i for i, c in enumerate(self.commodities)}
        self.flow_to_id = {f: i for i, f in enumerate(self.flows)}

        # Expose counts for encoders
        self.n_states = len(self.states)
        self.n_comms = len(self.commodities)
        self.n_flows = len(self.flows)
        
        grid["state_id"] = grid["State"].map(self.state_to_id)
        grid["comm_id"] = grid["Commodity"].map(self.comm_to_id)
        grid["flow_id"] = grid["Import/Export"].map(self.flow_to_id)

        # 4. Split data
        train_df = grid[grid["Time"] <= "2023-12-01"]
        val_df = grid[(grid["Time"] > "2023-12-01") & (grid["Time"] <= "2024-12-01")]
        test_df = grid[grid["Time"] > "2024-12-01"]

        # 5. Normalization (fit on train only)
        stats = compute_series_stats(train_df)
        train_z = apply_stats_with_backoff(train_df, stats)
        val_z = apply_stats_with_backoff(val_df, stats)
        test_z = apply_stats_with_backoff(test_df, stats)
        
        # 6. Features
        full_z = pd.concat([train_z, val_z, test_z])
        full_z = add_time_features(full_z)
        full_z = add_lags(full_z, 12)
        full_z, self.feat_cols = feature_view(full_z)
        
        # Target selection
        if self.target.lower() == 'value':
            self.d_output = 1
            self.target_cols = [VALUE_CANON]
        elif self.target.lower() == 'weight':
            self.d_output = 1
            self.target_cols = [WEIGHT_CANON]
        else:
            self.d_output = 2
            self.target_cols = [VALUE_CANON, WEIGHT_CANON]

        # Re-split after feature engineering
        train_data = full_z[full_z["Time"] <= "2023-12-01"]
        val_data = full_z[(full_z["Time"] > "2023-12-01") & (full_z["Time"] <= "2024-12-01")]
        test_data = full_z[full_z["Time"] > "2024-12-01"]
        
        # 7. Create Datasets
        self.dataset_train = TradeSeriesWindowDataset(train_data, self.feat_cols, self.input_len, self.horizon, self.target_cols, self.encoding)
        self.dataset_val = TradeSeriesWindowDataset(val_data, self.feat_cols, self.input_len, self.horizon, self.target_cols, self.encoding)
        self.dataset_test = TradeSeriesWindowDataset(test_data, self.feat_cols, self.input_len, self.horizon, self.target_cols, self.encoding)
    
    def _collate_fn(self, batch):
        xs, ys, state_ids, comm_ids, flow_ids = zip(*batch)
        
        x = torch.from_numpy(np.array(xs))
        y = torch.from_numpy(np.array(ys)).squeeze(1)

        if self.encoding == 'flat':
            state_ids = torch.tensor(state_ids, dtype=torch.long)
            comm_ids = torch.tensor(comm_ids, dtype=torch.long)
            flow_ids = torch.tensor(flow_ids, dtype=torch.long)
            return x, y, {"state_ids": state_ids, "comm_ids": comm_ids, "commodity_ids": comm_ids, "flow_ids": flow_ids}
        elif self.encoding == 'onehot':
            state_oh = torch.nn.functional.one_hot(torch.tensor(state_ids), len(self.states)).float()
            comm_oh = torch.nn.functional.one_hot(torch.tensor(comm_ids), len(self.commodities)).float()
            flow_oh = torch.nn.functional.one_hot(torch.tensor(flow_ids), len(self.flows)).float()
            
            cat_feats = torch.cat([state_oh, comm_oh, flow_oh], dim=-1).unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, cat_feats], dim=-1)
            return x, y, {}
        else:
            raise ValueError(f"Unknown encoding {self.encoding}")

    def _dataloader(self, dataset, **kwargs):
        return DataLoader(dataset, collate_fn=self._collate_fn, **kwargs)


class TradeFlat(TradeDatasetBase):
    _name_ = "s4_flat"
    
    def setup(self):
        super().setup()
        self.d_input = len(self.feat_cols) + 8 + 32 + 2  # feat + tradeid(state=8,comm=32,flow=2)
    
    @property
    def init_defaults(self):
        defaults = super().init_defaults
        defaults['encoding'] = 'flat'
        return defaults


class TradeOneHot(TradeDatasetBase):
    _name_ = "s4_onehot"

    def setup(self):
        super().setup()
        self.d_input = len(self.feat_cols) + len(self.states) + len(self.commodities) + len(self.flows)

    @property
    def init_defaults(self):
        defaults = super().init_defaults
        defaults['encoding'] = 'onehot'
        return defaults


class TradeEmbd3d(TradeFlat):
    """Same flat data as s4_flat; encoder (scid+flowid3d) handles the 3d embedding dims."""
    _name_ = "embd_3d"
