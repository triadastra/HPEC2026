"""Clean S4ND grid datasets, backed by the canonical github pipeline.

Subclasses the existing tensor-building logic in ts.py (Agg5DAll / Agg6DAll)
but replaces their data source: instead of /root/shared_dataloader.py (which
does not dedup-sum the 74% duplicate sub-records), we feed the canonical
TradeDataPipeline's clean splits (dedup-summed, 95%-filtered -> 903 combos,
per-combo min-max on train, reach-back-ready). The inherited
_build_sc_tensor_per_flow / _build_scflow_tensor take one aggregated row per
(combo, month) -- so the corruption is gone at the source.

Axis naming = (number of categorical grid axes) + time, i.e. the "+time"
convention (time is always the sequence axis):
    s4nd_2d : State grid, Commodity x Flow folded into a group axis G; commodity
              & flow re-injected as embeddings via the cfid encoder
              (== old 4d / Agg4DState; the paper's "2-D by State" = 1 categorical
              axis + time). The genuine 2-D S4ND.
    s4nd_3d : State x Commodity grid, flow handled per-flow   (== old 5dall;
              2 categorical axes + time = 3-D)
    s4nd_4d : Flow x State x Commodity grid                   (== old 6dall;
              3 categorical axes + time = 4-D)

A kept_mask marks the 903 real combos so eval scores only those (the dense grid
pads absent combos with zeros), keeping the 4515-target comparison fair against
the github models. The mask is (S, C, 2) for the 3d/4d grids and (S, G) for the
2d (state) grid (G = Commodity x Flow groups).
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataloaders.ts import Agg4DState, Agg5DAll, Agg6DAll
from src.dataloaders.trade_unified import _build_denorm

import pathlib
_REPO = pathlib.Path(__file__).resolve().parents[4]  # repo root (was hardcoded /root/full)
_GH_PATH = str(_REPO / "src" / "data" / "dataloader.py")
_spec = importlib.util.spec_from_file_location("github_dataloader_grid", _GH_PATH)
_GH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_GH)

_FLOW_ORDER = ["Export", "Import"]


def _clean_pipeline(data_dir, input_len):
    # Resolve the data root robustly: configured dir, repo root (fresh clone
    # with data/ in-repo), or repo PARENT (VM: repo /root/full, data /root/data).
    _candidates = [data_dir, str(_REPO), str(_REPO.parent)]
    data_dir = next(
        (c for c in _candidates if os.path.isdir(os.path.join(c, "data/imports"))),
        str(_REPO),
    )
    pipe = _GH.TradeDataPipeline(_GH.DataConfig(
        imports_dir=os.path.join(data_dir, "data/imports"),
        exports_dir=os.path.join(data_dir, "data/exports"),
        input_len=int(input_len),
        lag_count=12,
    ))
    pipe.load_data()
    pipe.create_splits()
    return pipe


def _kept_mask(pipe, states, comms):
    """(S, C, 2) bool: True where (state, commodity, flow) is a kept combo."""
    kept = set(map(tuple, pipe.kept_combos))  # (State, Commodity, Import/Export)
    S, C = len(states), len(comms)
    mask = np.zeros((S, C, 2), dtype=bool)
    for si, s in enumerate(states):
        for ci, c in enumerate(comms):
            for fi, f in enumerate(_FLOW_ORDER):
                if (s, c, f) in kept:
                    mask[si, ci, fi] = True
    return mask


def _kept_mask_groups(pipe, states, groups):
    """(S, G) bool: True where (state, commodity, flow) is a kept combo.

    `groups[gi]` is a `(Commodity, Import/Export)` pair (the state grid folds
    flow into the group axis), so each kept (state, commodity, flow) triple maps
    to exactly one (state, group) cell.
    """
    kept = set(map(tuple, pipe.kept_combos))  # (State, Commodity, Import/Export)
    S, G = len(states), len(groups)
    mask = np.zeros((S, G), dtype=bool)
    for si, s in enumerate(states):
        for gi, (c, f) in enumerate(groups):
            if (s, c, f) in kept:
                mask[si, gi] = True
    return mask


class TradeS4ND3D(Agg5DAll):
    """State x Commodity grid, per-flow (clean data) — 2 categorical axes + time."""
    _name_ = "s4nd_3d"

    @property
    def init_defaults(self):
        d = dict(super().init_defaults)
        d["data_dir"] = str(_REPO)
        return d

    def setup(self):
        if getattr(self, "_is_setup", False):
            return
        self._is_setup = True

        pipe = _clean_pipeline(self.data_dir, self.input_len)
        trf, vaf, tef = pipe.df_train, pipe.df_val, pipe.df_test
        feat_cols = pipe.feat_cols

        X_tr_all, states, comms, times_tr, sid_grid, cid_grid = self._build_sc_tensor_per_flow(trf, feat_cols)
        trva = pd.concat([trf, vaf], ignore_index=True)
        X_va_all, _, _, times_va, _, _ = self._build_sc_tensor_per_flow(trva, feat_cols)
        trvate = pd.concat([trf, vaf, tef], ignore_index=True)
        X_te_all, _, _, times_te, _, _ = self._build_sc_tensor_per_flow(trvate, feat_cols)

        y_tr_all, _, _, _ = self._build_targets_per_flow(trf)
        y_va_all, _, _, _ = self._build_targets_per_flow(trva)
        y_te_all, _, _, _ = self._build_targets_per_flow(trvate)

        L = int(self.input_len)
        t_val_start = pd.Timestamp(self.split_val_start)
        t_val_end = pd.Timestamp(self.split_val_end)
        t_test_start = pd.Timestamp(self.split_test_start)
        t_test_end = pd.Timestamp(self.split_test_end)

        def windows_per_flow(X_all, y_all, times, start, end):
            T = X_all.shape[0]
            idxs = [t for t in range(L, T)
                    if (start is None or times[t] >= start) and (end is None or times[t] <= end)]
            X_list, Y_list, FID_list = [], [], []
            for t in idxs:
                for fid in (0, 1):
                    x_seg = np.transpose(X_all[t - L:t, :, :, fid, :], (1, 2, 0, 3)).astype(np.float32)  # (S,C,L,F)
                    X_list.append(x_seg)
                    Y_list.append(y_all[t, :, :, fid, :].astype(np.float32))                              # (S,C,2)
                    FID_list.append(np.full((X_all.shape[1], X_all.shape[2]), fid, dtype=np.int64))
            return (np.stack(X_list), np.stack(Y_list), np.stack(FID_list))

        X_tr, y_tr, fid_tr = windows_per_flow(X_tr_all, y_tr_all, times_tr, None, pd.Timestamp(self.split_train_end))
        X_va, y_va, fid_va = windows_per_flow(X_va_all, y_va_all, times_va, t_val_start, t_val_end)
        X_te, y_te, fid_te = windows_per_flow(X_te_all, y_te_all, times_te, t_test_start, t_test_end)

        S, C = sid_grid.shape

        def tile(grid, n):
            return np.tile(grid[None, ...], (n, 1, 1))

        class _DS(Dataset):
            def __init__(self, X, Y, SID, CID, FID):
                self.X, self.Y, self.SID, self.CID, self.FID = X, Y, SID, CID, FID
            def __len__(self):
                return self.X.shape[0]
            def __getitem__(self, i):
                return (torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]),
                        torch.from_numpy(self.SID[i]), torch.from_numpy(self.CID[i]),
                        torch.from_numpy(self.FID[i]))

        self.dataset_train = _DS(X_tr, y_tr, tile(sid_grid, len(X_tr)), tile(cid_grid, len(X_tr)), fid_tr)
        self.dataset_val   = _DS(X_va, y_va, tile(sid_grid, len(X_va)), tile(cid_grid, len(X_va)), fid_va)
        self.dataset_test  = _DS(X_te, y_te, tile(sid_grid, len(X_te)), tile(cid_grid, len(X_te)), fid_te)

        self.n_states, self.n_comms = S, C
        extra = int(getattr(self, "encoder_extra_dim", 0) or 0)
        self.d_input = X_tr.shape[-1] + extra
        # l_output is a read-only property on the parent (returns 0).
        # Fair-eval support: kept-combo mask + per-test-sample flow ids.
        self.kept_mask = _kept_mask(pipe, states, comms)
        self.test_fids = fid_te[:, 0, 0]  # flow id per test sample
        self.states, self.comms = states, comms

        # Flat (key, time) aligned with the masked-eval pred order (C-order
        # over kept cells, per (month, flow) sample) + denorm map, for plots.
        self.pipe = pipe
        self._denorm = _build_denorm(pipe)
        test_idxs = [t for t in range(L, X_te_all.shape[0])
                     if t_test_start <= times_te[t] <= t_test_end]
        months = [times_te[t] for t in test_idxs for _ in (0, 1)]  # (month,flow) order
        FLOW = ["Export", "Import"]
        tk, tt = [], []
        for i, fid in enumerate(self.test_fids):
            for s, c in np.argwhere(self.kept_mask[:, :, int(fid)]):
                tk.append((states[s], comms[c], FLOW[int(fid)]))
                tt.append(months[i])
        self.test_keys, self.test_times = tk, tt


class TradeS4ND4D(Agg6DAll):
    """Flow x State x Commodity grid (clean data) — 3 categorical axes + time."""
    _name_ = "s4nd_4d"

    @property
    def init_defaults(self):
        d = dict(super().init_defaults)
        d["data_dir"] = str(_REPO)
        return d

    def setup(self):
        if getattr(self, "_is_setup", False):
            return
        self._is_setup = True

        pipe = _clean_pipeline(self.data_dir, self.input_len)
        trf, vaf, tef = pipe.df_train, pipe.df_val, pipe.df_test
        feat_cols = pipe.feat_cols

        X_tr_all, states, comms, times_tr = self._build_scflow_tensor(trf, feat_cols)
        trva = pd.concat([trf, vaf], ignore_index=True)
        X_va_all, _, _, times_va = self._build_scflow_tensor(trva, feat_cols)
        trvate = pd.concat([trf, vaf, tef], ignore_index=True)
        X_te_all, _, _, times_te = self._build_scflow_tensor(trvate, feat_cols)

        L = int(self.input_len)
        X_tr, y_tr, _ = self._windows(X_tr_all, times_tr, L, None, pd.Timestamp(self.split_train_end))
        X_va, y_va, _ = self._windows(X_va_all, times_va, L,
                                      pd.Timestamp(self.split_val_start), pd.Timestamp(self.split_val_end))
        X_te, y_te, _ = self._windows(X_te_all, times_te, L,
                                      pd.Timestamp(self.split_test_start), pd.Timestamp(self.split_test_end))

        class _DS(Dataset):
            def __init__(self, X, Y):
                self.X, self.Y = X, Y
            def __len__(self):
                return self.X.shape[0]
            def __getitem__(self, i):
                return (torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]))

        self.dataset_train = _DS(X_tr, y_tr)
        self.dataset_val = _DS(X_va, y_va)
        self.dataset_test = _DS(X_te, y_te)

        self.d_input = X_tr.shape[-1]
        self.kept_mask = _kept_mask(pipe, states, comms)  # (S, C, 2)
        self.states, self.comms = states, comms

        # Flat (key, time) aligned with the masked-eval pred order (C-order
        # over all kept (s,c,f) cells, one sample per test month) + denorm map.
        self.pipe = pipe
        self._denorm = _build_denorm(pipe)
        t_test_start = pd.Timestamp(self.split_test_start)
        t_test_end = pd.Timestamp(self.split_test_end)
        test_idxs = [t for t in range(L, X_te_all.shape[0])
                     if t_test_start <= times_te[t] <= t_test_end]
        months = [times_te[t] for t in test_idxs]
        FLOW = ["Export", "Import"]
        cells = np.argwhere(self.kept_mask)  # (s, c, f)
        tk, tt = [], []
        for t in months:
            for s, c, f in cells:
                tk.append((states[s], comms[c], FLOW[int(f)]))
                tt.append(t)
        self.test_keys, self.test_times = tk, tt


class TradeS4ND2D(Agg4DState):
    """State grid with Commodity x Flow folded into a group axis (clean data).

    The paper's "2-D by State" (1 categorical axis + time = 2-D, the genuine 2-D
    S4ND): State is the structured grid axis and each (Commodity, Import/Export)
    pair is a cell in the group axis G, with commodity and flow re-injected as
    embeddings by the `cfid` encoder. Mirrors TradeS4ND3D/4D (the higher-rank
    grids) but inherits Agg4DState's tensor builders and swaps the data
    source from the buggy /root/shared_dataloader.py to the canonical clean
    pipeline (dedup-summed, 95%-filtered -> 903 combos, per-combo min-max, reach
    back windows). Because the clean pipeline puts the same 903 combos in every
    split, the group axis G is identical across train/val/test.

    Per-sample shapes: X (S, G, L, F); y (S, G, 2); commodity_ids/flow_ids (S, G).
    Model output is (B, S, G, 2); eval masks it with an (S, G) kept_mask.
    """
    _name_ = "s4nd_2d"

    @property
    def init_defaults(self):
        d = dict(super().init_defaults)
        d["data_dir"] = str(_REPO)
        return d

    def setup(self):
        if getattr(self, "_is_setup", False):
            return
        self._is_setup = True

        pipe = _clean_pipeline(self.data_dir, self.input_len)
        trf, vaf, tef = pipe.df_train, pipe.df_val, pipe.df_test
        feat_cols = pipe.feat_cols

        # groups (the Commodity x Flow axis) come from the train call; the clean
        # pipeline carries the same 903 combos in every split, so val/test share
        # the identical groups ordering.
        X_tr_all, states, times_tr, groups = self._build_state_group_tensor(trf, feat_cols)
        trva = pd.concat([trf, vaf], ignore_index=True)
        X_va_all, states_va, times_va, groups_va = self._build_state_group_tensor(trva, feat_cols)
        trvate = pd.concat([trf, vaf, tef], ignore_index=True)
        X_te_all, states_te, times_te, groups_te = self._build_state_group_tensor(trvate, feat_cols)
        # The clean pipeline carries the same 903 combos in every split, so the
        # State axis and the Commodity x Flow group axis are identical across
        # splits; assert it so a future pipeline change fails loudly rather than
        # silently misaligning the grid / ids.
        assert states == states_va == states_te, "state sets differ across splits"
        assert groups == groups_va == groups_te, "group (commodity x flow) sets differ across splits"

        L = int(self.input_len)
        t_test_start = pd.Timestamp(self.split_test_start)
        t_test_end = pd.Timestamp(self.split_test_end)

        # y comes from X_all[t, :, :, :2] inside _windows -> (N, S, G, 2).
        X_tr, y_tr, _ = self._windows(X_tr_all, times_tr, L, None, pd.Timestamp(self.split_train_end))
        X_va, y_va, _ = self._windows(X_va_all, times_va, L,
                                      pd.Timestamp(self.split_val_start), pd.Timestamp(self.split_val_end))
        X_te, y_te, _ = self._windows(X_te_all, times_te, L, t_test_start, t_test_end)

        # (S, G) commodity/flow id grids, constant across the State axis, tiled per sample.
        commodities = sorted({c for c, _ in groups})
        commodity_to_id = {c: i for i, c in enumerate(commodities)}
        flow_to_id = {"Export": 0, "Import": 1}
        S, G = X_tr.shape[1], X_tr.shape[2]
        cid_grid = np.zeros((S, G), dtype=np.int64)
        fid_grid = np.zeros((S, G), dtype=np.int64)
        for gi, (c, f) in enumerate(groups):
            cid_grid[:, gi] = commodity_to_id.get(c, 0)
            fid_grid[:, gi] = flow_to_id.get(f, 0)

        def tile_ids(n):
            return (np.tile(cid_grid[None, ...], (n, 1, 1)),
                    np.tile(fid_grid[None, ...], (n, 1, 1)))

        cid_tr, fid_tr = tile_ids(len(X_tr))
        cid_va, fid_va = tile_ids(len(X_va))
        cid_te, fid_te = tile_ids(len(X_te))

        class _DS(Dataset):
            def __init__(self, X, Y, CID, FID):
                self.X, self.Y, self.CID, self.FID = X, Y, CID, FID
            def __len__(self):
                return self.X.shape[0]
            def __getitem__(self, i):
                return (torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]),
                        torch.from_numpy(self.CID[i]), torch.from_numpy(self.FID[i]))

        self.dataset_train = _DS(X_tr, y_tr, cid_tr, fid_tr)
        self.dataset_val   = _DS(X_va, y_va, cid_va, fid_va)
        self.dataset_test  = _DS(X_te, y_te, cid_te, fid_te)

        # cfid appends commodity(32) + flow(2) embeddings to the F core features.
        self.n_comms = len(commodities)
        self.d_input = X_tr.shape[-1] + 32 + 2
        # _collate_arg_names = ['commodity_ids', 'flow_ids'] (inherited); no state_ids.

        # Fair-eval support: (S, G) kept-combo mask + flat (key, time) order.
        self.kept_mask = _kept_mask_groups(pipe, states, groups)  # (S, G)
        self.states, self.groups = states, groups
        self.pipe = pipe
        self._denorm = _build_denorm(pipe)

        # Masked-eval flatten order: one sample per test month (ascending), and
        # within a sample x[b][mask] scans (S, G) in C-order (S slow, G fast).
        test_idxs = [t for t in range(L, X_te_all.shape[0])
                     if t_test_start <= times_te[t] <= t_test_end]
        months = [times_te[t] for t in test_idxs]
        cells = np.argwhere(self.kept_mask)  # (s, g) in C-order, matches x[b][mask]
        tk, tt = [], []
        for t in months:
            for s, g in cells:
                c, f = groups[int(g)]
                tk.append((states[int(s)], c, f))
                tt.append(t)
        self.test_keys, self.test_times = tk, tt


class TradeS4ND2DState(Agg4DState):
    """State-flat "2-D by State": State is the ONLY convolved categorical axis.

    Unlike s4nd_2d (which folds Commodity x Flow into a grid axis G and convolves
    over it), this variant folds each (Commodity, Import/Export) series into the
    BATCH dimension, so a sample is the dense State axis for one commodity/flow:
        X (S, L, F), y (S, 2), with commodity & flow re-injected as per-sample
        features via the `cfstate` encoder.
    The S4ND layer therefore convolves over (State, time) only (dim=2) and the
    model output is (B, S, 2) -- the literal Tensor(B,S,L,F) of the paper, and the
    clean "+1 categorical axis over the flatten baseline" rung of the
    dimensionality ablation (flatten -> +State -> +Commodity -> +Flow).

    Which states are real depends on the sample's (commodity, flow), so masking is
    per-sample: each sample carries a `valid_mask` (S,) of its kept states (the
    other states are zero-padded). Train loss is masked via _mask_grid_output and
    eval selects the kept states, keeping the same 4515-target comparison.
    """
    _name_ = "s4nd_2d_state"
    _collate_arg_names = ["commodity_ids", "flow_ids", "valid_mask"]

    @property
    def init_defaults(self):
        d = dict(super().init_defaults)
        d["data_dir"] = str(_REPO)
        return d

    def setup(self):
        if getattr(self, "_is_setup", False):
            return
        self._is_setup = True

        pipe = _clean_pipeline(self.data_dir, self.input_len)
        trf, vaf, tef = pipe.df_train, pipe.df_val, pipe.df_test
        feat_cols = pipe.feat_cols

        X_tr_all, states, times_tr, groups = self._build_state_group_tensor(trf, feat_cols)
        trva = pd.concat([trf, vaf], ignore_index=True)
        X_va_all, states_va, times_va, groups_va = self._build_state_group_tensor(trva, feat_cols)
        trvate = pd.concat([trf, vaf, tef], ignore_index=True)
        X_te_all, states_te, times_te, groups_te = self._build_state_group_tensor(trvate, feat_cols)
        assert states == states_va == states_te, "state sets differ across splits"
        assert groups == groups_va == groups_te, "group (commodity x flow) sets differ across splits"

        S, G = len(states), len(groups)
        L = int(self.input_len)
        t_test_start = pd.Timestamp(self.split_test_start)
        t_test_end = pd.Timestamp(self.split_test_end)

        # Per-group commodity/flow ids and per-group kept-state mask (static).
        commodities = sorted({c for c, _ in groups})
        commodity_to_id = {c: i for i, c in enumerate(commodities)}
        flow_to_id = {"Export": 0, "Import": 1}
        kept = set(map(tuple, pipe.kept_combos))
        cid_of_g = np.array([commodity_to_id.get(groups[g][0], 0) for g in range(G)], dtype=np.int64)
        fid_of_g = np.array([flow_to_id.get(groups[g][1], 0) for g in range(G)], dtype=np.int64)
        group_valid = np.zeros((G, S), dtype=bool)  # group_valid[g, s] = (state, c, f) kept
        for gi, (c, f) in enumerate(groups):
            for si, s in enumerate(states):
                if (s, c, f) in kept:
                    group_valid[gi, si] = True

        def state_windows(X_all, times, start, end):
            """One sample per (target month, group g): X (S,L,F), y (S,2),
            commodity/flow ids (S,) and valid_mask (S,). Order: month-major,
            group-minor -- the order test_keys is built in."""
            T = X_all.shape[0]  # (T, S, G, F)
            idxs = [t for t in range(L, T)
                    if (start is None or times[t] >= start) and (end is None or times[t] <= end)]
            Xs, Ys, CIDs, FIDs, VMs = [], [], [], [], []
            for t in idxs:
                for g in range(G):
                    Xs.append(np.transpose(X_all[t - L:t, :, g, :], (1, 0, 2)).astype(np.float32))  # (S,L,F)
                    Ys.append(X_all[t, :, g, :2].astype(np.float32))                                 # (S,2)
                    CIDs.append(np.full((S,), cid_of_g[g], dtype=np.int64))
                    FIDs.append(np.full((S,), fid_of_g[g], dtype=np.int64))
                    VMs.append(group_valid[g].astype(np.float32))                                    # (S,)
            if not Xs:
                F = X_all.shape[3]
                return (np.zeros((0, S, L, F), np.float32), np.zeros((0, S, 2), np.float32),
                        np.zeros((0, S), np.int64), np.zeros((0, S), np.int64),
                        np.zeros((0, S), np.float32))
            return (np.stack(Xs), np.stack(Ys), np.stack(CIDs), np.stack(FIDs), np.stack(VMs))

        X_tr, y_tr, cid_tr, fid_tr, vm_tr = state_windows(X_tr_all, times_tr, None, pd.Timestamp(self.split_train_end))
        X_va, y_va, cid_va, fid_va, vm_va = state_windows(X_va_all, times_va,
                                                          pd.Timestamp(self.split_val_start), pd.Timestamp(self.split_val_end))
        X_te, y_te, cid_te, fid_te, vm_te = state_windows(X_te_all, times_te, t_test_start, t_test_end)

        class _DS(Dataset):
            def __init__(self, X, Y, CID, FID, VM):
                self.X, self.Y, self.CID, self.FID, self.VM = X, Y, CID, FID, VM
            def __len__(self):
                return self.X.shape[0]
            def __getitem__(self, i):
                return (torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]),
                        torch.from_numpy(self.CID[i]), torch.from_numpy(self.FID[i]),
                        torch.from_numpy(self.VM[i]))

        self.dataset_train = _DS(X_tr, y_tr, cid_tr, fid_tr, vm_tr)
        self.dataset_val   = _DS(X_va, y_va, cid_va, fid_va, vm_va)
        self.dataset_test  = _DS(X_te, y_te, cid_te, fid_te, vm_te)

        # cfstate appends commodity(32) + flow(2) embeddings to the F core features.
        self.n_comms = len(commodities)
        self.d_input = X_tr.shape[-1] + 32 + 2
        # NOTE: no static kept_mask -- valid states vary per sample, so masking
        # is driven by the per-sample valid_mask (see train._mask_grid_output and
        # s4_eval). Leaving kept_mask unset makes those paths fall through to the
        # valid_mask branch.
        self.states, self.groups = states, groups
        self.pipe = pipe
        self._denorm = _build_denorm(pipe)

        # Masked-eval flatten order: sample order is (month ascending, group), and
        # within a sample x[b][valid_mask] scans S ascending. test_keys mirrors it.
        test_idxs = [t for t in range(L, X_te_all.shape[0])
                     if t_test_start <= times_te[t] <= t_test_end]
        months = [times_te[t] for t in test_idxs]
        tk, tt = [], []
        for t in months:
            for g in range(G):
                c, f = groups[g]
                for si in np.where(group_valid[g])[0]:  # kept states, ascending
                    tk.append((states[int(si)], c, f))
                    tt.append(t)
        self.test_keys, self.test_times = tk, tt
