"""
embd_ts.py

Lightweight windowed dataset that mirrors GRU/Transformer SeriesWindowDataset behavior,
but returns embedding IDs (state_id, comm_id, flow_id) per sample so models can
append learnable embeddings at runtime. Intended to live next to ts.py and avoid
crowding it further.

Expected input DataFrame columns (z-score space features constructed upstream):
  ["Time", "Commodity", "State", "Import/Export", <feat_cols...>]

Produces samples:
  x: (L, F_numeric)
  y: (1, Ft) where Ft = 2 if target=="both" else 1
  ids: dict with keys {"state_id","comm_id","flow_id"}

This matches the semantics used in Transformer/GRU flat pipelines so S4 models can
reuse the exact same training/evaluation loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.dataloaders.shared_dataloader import (
    read_folder, build_grid, temporal_split_calendar,
    compute_series_stats, apply_stats_with_backoff, add_lags, feature_view,
    build_id_maps,
)
from src.dataloaders.base import SequenceDataset


VALUE_CANON = "Value"
WEIGHT_CANON = "Weight"


@dataclass
class WindowCfg:
    input_len: int = 36
    horizon: int = 1
    target: str = "both"  # "value" | "weight" | "both"


class SeriesWindowDatasetEmbd(Dataset):
    def __init__(
        self,
        df_feat: pd.DataFrame,
        feat_cols: List[str],
        input_len: int,
        horizon: int,
        target: str,
        state2id: Dict[str, int],
        comm2id: Dict[str, int],
        flow2id: Dict[str, int],
        target_time_start: Optional[pd.Timestamp] = None,
        target_time_end: Optional[pd.Timestamp] = None,
    ) -> None:
        super().__init__()
        self.feat_cols = list(feat_cols)
        self.input_len = int(input_len)
        assert int(horizon) == 1, "Only horizon=1 supported in this windowed dataset"
        self.horizon = 1
        assert target in ("value", "weight", "both")
        self.target = target
        self.state2id = state2id
        self.comm2id = comm2id
        self.flow2id = flow2id
        self.target_time_start = target_time_start
        self.target_time_end = target_time_end

        # Group by series triplet
        groups = df_feat.groupby(["State", "Commodity", "Import/Export"], sort=False)
        samples: List[Tuple[np.ndarray, np.ndarray, Dict[str, int]]] = []

        for (state, comm, flow), g in groups:
            g = g.sort_values("Time").reset_index(drop=True)
            X = g[self.feat_cols].to_numpy(dtype=np.float32)
            times = g["Time"].to_numpy()
            Tn = X.shape[0]
            if Tn < self.input_len + self.horizon:
                continue

            sid = self.state2id.get(state, 0)
            cid = self.comm2id.get(comm, 0)
            fid = self.flow2id.get(flow, 0)

            # Targets in z space
            cols = ([VALUE_CANON] if self.target=="value" else
                    [WEIGHT_CANON] if self.target=="weight" else
                    [VALUE_CANON, WEIGHT_CANON])
            Y = g[cols].to_numpy(dtype=np.float32)

            for t in range(self.input_len, Tn - self.horizon + 1):
                t_pred_time = times[t]  # next-step target time
                if self.target_time_start is not None and t_pred_time < self.target_time_start:
                    continue
                if self.target_time_end is not None and t_pred_time > self.target_time_end:
                    continue

                x = X[t - self.input_len : t, :]
                y = Y[t : t + 1, :]  # shape (1, Ft)
                samples.append((x, y, {"state_id": sid, "comm_id": cid, "flow_id": fid}))

        if not samples:
            self.X = np.zeros((0, self.input_len, len(self.feat_cols)), dtype=np.float32)
            self.Y = np.zeros((0, 1, 2 if self.target=="both" else 1), dtype=np.float32)
            self.IDS = []
        else:
            self.X = np.stack([s[0] for s in samples], axis=0)
            self.Y = np.stack([s[1] for s in samples], axis=0)
            self.IDS = [s[2] for s in samples]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        xb = torch.from_numpy(self.X[i])
        yb = torch.from_numpy(self.Y[i])
        ids = self.IDS[i]
        # Convert ids to tensors lazily to avoid large memory upfront
        ids = {k: torch.tensor(v, dtype=torch.long) for k, v in ids.items()}
        return xb, yb, ids


class Embd3DFlat(SequenceDataset):
    """
    Native S4 dataset that builds flat (B, L, F) windows and returns IDs for
    appending embeddings via encoders (SCID + FlowID3D) before a Linear encoder.

    Collate returns extra dict with keys ['state_ids','commodity_ids','flow_ids'].
    """

    _name_ = "embd_3d"
    _collate_arg_names = ['state_ids', 'commodity_ids', 'flow_ids']

    # Default dims aligned with Transformer defaults
    @property
    def init_defaults(self):
        return {
            "imports_dir": "/root/data/imports",
            "exports_dir": "/root/data/exports",
            "start": "2008-01-01",
            "end":   "2025-05-01",
            "input_len": 36,
            "horizon": 1,
            "target": "both",  # value | weight | both
            # splits
            "split_train_end": "2023-12-01",
            "split_val_start": "2024-01-01",
            "split_val_end":   "2024-12-01",
            "split_test_start": "2025-01-01",
            "split_test_end":   "2025-05-01",
            # embedding dims (for computing d_input once encoders append)
            "state_emb_dim": 8,
            "commodity_emb_dim": 32,
            "flow_emb_dim": 2,
        }

    def setup(self):
        # 1) Read & build ID maps from full panel
        imports = read_folder(self.imports_dir, "Import")
        exports = read_folder(self.exports_dir, "Export")
        panel  = pd.concat([imports, exports], ignore_index=True)
        state2id, comm2id, flow2id = build_id_maps(panel)
        self.n_states, self.n_comms = len(state2id), len(comm2id)

        # 2) Grid + encodings
        grid = build_grid(panel, self.start, self.end)

        # 3) Calendar split
        tr_raw, va_raw, te_raw = temporal_split_calendar(grid, self)

        # 4) Train-only stats → normalize all splits
        stats = compute_series_stats(tr_raw)
        tr = apply_stats_with_backoff(tr_raw, stats)
        va = apply_stats_with_backoff(va_raw, stats)
        te = apply_stats_with_backoff(te_raw, stats)

        # 5) Add lags (12)
        tr = add_lags(tr, 12)
        va = add_lags(va, 12)
        te = add_lags(te, 12)

        # 6) Feature assembly
        trf, feat_cols, _ = feature_view(tr)
        vaf, _, _ = feature_view(va)
        tef, _, _ = feature_view(te)
        self._feat_cols = feat_cols

        # 7) Build flat windows with IDs
        def build_ds(df_feat, t_start, t_end):
            ds = SeriesWindowDatasetEmbd(
                df_feat=df_feat,
                feat_cols=feat_cols,
                input_len=int(self.input_len),
                horizon=1,
                target=str(self.target),
                state2id=state2id,
                comm2id=comm2id,
                flow2id=flow2id,
                target_time_start=t_start,
                target_time_end=t_end,
            )
            class _DS(Dataset):
                def __init__(self, parent):
                    self.parent = parent
                def __len__(self): return len(self.parent)
                def __getitem__(self, i):
                    x, y, ids = self.parent[i]
                    return x, y, ids["state_id"], ids["comm_id"], ids["flow_id"]
            return _DS(ds)

        t_train_end = pd.Timestamp(self.split_train_end)
        t_val_start = pd.Timestamp(self.split_val_start)
        t_val_end   = pd.Timestamp(self.split_val_end)
        t_test_start= pd.Timestamp(self.split_test_start)
        t_test_end  = pd.Timestamp(self.split_test_end)

        tr_ds = build_ds(trf, None, t_train_end)
        trva  = pd.concat([trf, vaf], ignore_index=True)
        va_ds = build_ds(trva, t_val_start, t_val_end)
        trvate= pd.concat([trf, vaf, tef], ignore_index=True)
        te_ds = build_ds(trvate, t_test_start, t_test_end)

        self.dataset_train = tr_ds
        self.dataset_val   = va_ds
        self.dataset_test  = te_ds

        # 8) d_input = numeric features + embedding dims appended by encoders
        F_numeric = len(feat_cols)
        self.d_input = F_numeric + int(self.state_emb_dim) + int(self.commodity_emb_dim) + int(self.flow_emb_dim)

    def __str__(self):
        return "embd_3d"


