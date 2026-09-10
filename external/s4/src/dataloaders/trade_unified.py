"""Unified S4/S4ND dataloaders that DIRECTLY call the canonical github
TradeDataPipeline (src/data/dataloader.py) instead of re-implementing data
loading. This is the single source of truth for: dedup-sum of duplicate
(combo, month) sub-records, 95% sparse-combo filter (-> 903 combos),
per-combo min-max normalization fit on train, and reach-back val/test
windows. The S4 datasets here only adapt the pipeline's output into the
(x, y, z) tuple shape the S4 LightningModule expects.

The github pipeline imports only stdlib + numpy/pandas/torch (no `src.`
deps), so it loads cleanly via importlib with no package-name clash against
the S4 repo's own `src` package.

Datasets registered:
    s4_embeddings  -> per-series, ids passed to TradeIDEncoder (1-D S4)
    s4_onehot      -> per-series, one-hot appended to features (1-D S4)
"""
import importlib.util
import os

import torch
from torch.utils.data import DataLoader

from src.dataloaders.base import SequenceDataset

# --- load the canonical github pipeline by path (no `src` clash) ---
# Resolve the repo root from this file's location so it works on any machine /
# fresh clone (was hardcoded /root/full). dataloaders/ -> src/ -> s4/ ->
# external/ -> repo root == parents[4].
import pathlib
_REPO = pathlib.Path(__file__).resolve().parents[4]
_GH_PATH = str(_REPO / "src" / "data" / "dataloader.py")
_spec = importlib.util.spec_from_file_location("github_dataloader", _GH_PATH)
_GH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_GH)


def _build_denorm(pipe):
    """(state, commodity, flow) -> (val_min, val_range, wt_min, wt_range) for
    inverse min-max, so eval can plot aggregates in real units."""
    m = {}
    for _, r in pipe.stats["exact"].iterrows():
        key = (str(r["State"]), str(r["Commodity"]), str(r["Import/Export"]))
        vr = float(r["val_max"]) - float(r["val_min"])
        wr = float(r["wt_max"]) - float(r["wt_min"])
        m[key] = (float(r["val_min"]), vr if vr else 1.0,
                  float(r["wt_min"]), wr if wr else 1.0)
    return m


class _TradePerSeriesBase(SequenceDataset):
    """Per-series S4 dataset backed by the canonical pipeline."""
    _name_ = "s4_perseries_base"
    encoding = "embeddings"  # overridden by subclasses

    @property
    def init_defaults(self):
        return {
            "data_dir": str(_REPO),   # repo root; expects <data_dir>/data/{imports,exports}
            "input_len": 36,
            "lag_count": 12,
            "aggregate": False,    # Test 1: collapse to one summed series
        }

    def setup(self):
        if getattr(self, "_is_setup", False):
            return
        self._is_setup = True
        self.l_output = 0
        self.d_output = 2

        # Resolve the data root robustly across layouts: the configured dir, the
        # repo root (fresh clone with data/ in-repo), or the repo's PARENT (VM
        # convention: repo at /root/full, data at /root/data). First hit wins.
        _candidates = [self.data_dir, str(_REPO), str(_REPO.parent)]
        data_root = next(
            (c for c in _candidates if os.path.isdir(os.path.join(c, "data/imports"))),
            str(_REPO),
        )
        cfg = _GH.DataConfig(
            imports_dir=os.path.join(data_root, "data/imports"),
            exports_dir=os.path.join(data_root, "data/exports"),
            input_len=self.input_len,
            lag_count=self.lag_count,
            aggregate=getattr(self, "aggregate", False),
        )
        self.pipe = _GH.TradeDataPipeline(cfg)
        self.pipe.load_data()
        self.pipe.create_splits()

        # One get_dataloaders call materializes id maps + the per-series
        # TradeDataset objects (with reach-back target-time windows). We
        # reuse the .dataset handles and rebuild DataLoaders per request so
        # batch_size/num_workers from the S4 loader config are honored.
        tr, va, te = self.pipe.get_dataloaders(
            batch_size=64, num_workers=0, shuffle_train=False, combo=False
        )
        self._ds = {"train": tr.dataset, "val": va.dataset, "test": te.dataset}
        self._gh_collate = tr.collate_fn

        self.n_states = len(self.pipe.state2id)
        self.n_comms = len(self.pipe.comm2id)
        self.n_flows = len(self.pipe.flow2id)
        n_feat = len(self.pipe.feat_cols)

        if self.encoding == "onehot":
            self.d_input = n_feat + self.n_states + self.n_comms + self.n_flows
        else:  # embeddings: TradeIDEncoder appends (state=8, comm=32, flow=2)
            self.d_input = n_feat + 8 + 32 + 2

        # Per-test-sample (key, time), aligned with the unshuffled test loader,
        # + denorm map -> lets s4_eval draw real-unit aggregate plots.
        id2s = {v: k for k, v in self.pipe.state2id.items()}
        id2c = {v: k for k, v in self.pipe.comm2id.items()}
        id2f = {v: k for k, v in self.pipe.flow2id.items()}
        samples = self._ds["test"].samples
        self.test_keys = [(id2s[s["state_id"]], id2c[s["comm_id"]], id2f[s["flow_id"]])
                          for s in samples]
        self.test_times = [s["target_time"] for s in samples]
        self._denorm = _build_denorm(self.pipe)

    def _collate_fn(self, batch):
        """Convert a list of github sample dicts into S4's (x, y, z) tuple."""
        gh = self._gh_collate(batch)
        x = gh["x_numeric"]                                   # (B, L, F)
        y = torch.stack([gh["target_value"], gh["target_weight"]], dim=-1)  # (B, 2)

        if self.encoding == "onehot":
            B, L, _ = x.shape
            s = torch.nn.functional.one_hot(gh["state_ids"], self.n_states).float()
            c = torch.nn.functional.one_hot(gh["comm_ids"], self.n_comms).float()
            f = torch.nn.functional.one_hot(gh["flow_ids"], self.n_flows).float()
            cat = torch.cat([s, c, f], dim=-1).unsqueeze(1).expand(-1, L, -1)
            x = torch.cat([x, cat], dim=-1)
            return x, y, {}

        return x, y, {
            "state_ids": gh["state_ids"],
            "comm_ids": gh["comm_ids"],
            "commodity_ids": gh["comm_ids"],
            "flow_ids": gh["flow_ids"],
        }

    def _dataloader(self, dataset, **kwargs):
        return DataLoader(dataset, collate_fn=self._collate_fn, **kwargs)

    def train_dataloader(self, **kwargs):
        kwargs.setdefault("shuffle", True)
        return self._dataloader(self._ds["train"], **kwargs)

    def val_dataloader(self, **kwargs):
        kwargs["shuffle"] = False
        return self._dataloader(self._ds["val"], **kwargs)

    def test_dataloader(self, **kwargs):
        kwargs["shuffle"] = False
        return self._dataloader(self._ds["test"], **kwargs)


class TradeS4Embeddings(_TradePerSeriesBase):
    _name_ = "s4_embeddings"
    encoding = "embeddings"


class TradeS4OneHot(_TradePerSeriesBase):
    _name_ = "s4_onehot_u"
    encoding = "onehot"
