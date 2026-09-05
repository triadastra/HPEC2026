"""S4/S4ND dataloaders backed by the precomputed ext/ feature_tensors.pt bridge.

Same (x, y, z) contract as trade_unified.py, but instead of re-running the
canonical pipeline it LOADS the shared tensors file that the github models also
train on -- so S4 sees the EXACT same feature-augmented inputs (apples-to-apples
feature ablation). The canonical pipeline / existing S4 dataloaders are untouched;
this is a new _name_-registered dataset only.

Registered:
    s4_feat_embeddings -> per-series, ids -> TradeIDEncoder (d_input = F + 8+32+2)
    s4_feat_onehot     -> per-series, one-hot appended  (d_input = F + nS+nC+nF)

Pick the feature set at run time:
    dataset.tensors=/root/full/ext/data/processed/feature_tensors_base.pt   # 28
    dataset.tensors=.../feature_tensors_exog.pt                              # 33
    dataset.tensors=.../feature_tensors.pt                                   # 43
"""
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataloaders.base import SequenceDataset


class _FeatTensorDataset(Dataset):
    def __init__(self, split):
        self.x = split["x"]            # (N, L, F) float32
        self.y = split["y"]            # (N, 2)
        self.sid = split["state_ids"]
        self.cid = split["comm_ids"]
        self.fid = split["flow_ids"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return (self.x[i], self.y[i],
                int(self.sid[i]), int(self.cid[i]), int(self.fid[i]))


class _TradeFeatBase(SequenceDataset):
    _name_ = "s4_feat_base"
    encoding = "embeddings"  # overridden by subclasses

    @property
    def init_defaults(self):
        return {
            "tensors": "/root/full/ext/data/processed/feature_tensors.pt",
        }

    def setup(self):
        if getattr(self, "_is_setup", False):
            return
        self._is_setup = True
        self.l_output = 0
        self.d_output = 2

        data = torch.load(self.tensors, weights_only=False)
        self._splits = {k: data[k] for k in ("train", "val", "test")}
        self.n_states = int(data["n_states"])
        self.n_comms = int(data["n_commodities"])
        self.n_flows = int(data["n_flows"])
        n_feat = len(data["feat_cols"])

        if self.encoding == "onehot":
            self.d_input = n_feat + self.n_states + self.n_comms + self.n_flows
        else:  # embeddings: TradeIDEncoder appends (state=8, comm=32, flow=2)
            self.d_input = n_feat + 8 + 32 + 2

        # eval hooks (aligned with the unshuffled test loader)
        self.test_keys = [tuple(k) for k in data["test"]["keys"]]
        self.test_times = data["test"]["times"]
        self._denorm = self._build_denorm(data["target_denorm"])
        print(f"[s4-feat] {self.tensors.split('/')[-1]}: F={n_feat} "
              f"d_input={self.d_input} train N={self._splits['train']['x'].shape[0]}")

    @staticmethod
    def _build_denorm(td):
        m = {}
        for i in range(len(td["State"])):
            key = (str(td["State"][i]), str(td["Commodity"][i]),
                   str(td["Import/Export"][i]))
            vr = float(td["val_max"][i]) - float(td["val_min"][i])
            wr = float(td["wt_max"][i]) - float(td["wt_min"][i])
            m[key] = (float(td["val_min"][i]), vr if vr else 1.0,
                      float(td["wt_min"][i]), wr if wr else 1.0)
        return m

    def _collate_fn(self, batch):
        xs = torch.stack([b[0] for b in batch])               # (B, L, F)
        ys = torch.stack([b[1] for b in batch])               # (B, 2)
        sid = torch.tensor([b[2] for b in batch], dtype=torch.long)
        cid = torch.tensor([b[3] for b in batch], dtype=torch.long)
        fid = torch.tensor([b[4] for b in batch], dtype=torch.long)
        if self.encoding == "onehot":
            B, L, _ = xs.shape
            s = torch.nn.functional.one_hot(sid, self.n_states).float()
            c = torch.nn.functional.one_hot(cid, self.n_comms).float()
            f = torch.nn.functional.one_hot(fid, self.n_flows).float()
            cat = torch.cat([s, c, f], dim=-1).unsqueeze(1).expand(-1, L, -1)
            xs = torch.cat([xs, cat], dim=-1)
            return xs, ys, {}
        return xs, ys, {
            "state_ids": sid, "comm_ids": cid, "commodity_ids": cid, "flow_ids": fid,
        }

    def _dataloader(self, split, **kwargs):
        return DataLoader(_FeatTensorDataset(self._splits[split]),
                          collate_fn=self._collate_fn, **kwargs)

    def train_dataloader(self, **kwargs):
        kwargs.setdefault("shuffle", True)
        return self._dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        kwargs["shuffle"] = False
        return self._dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        kwargs["shuffle"] = False
        return self._dataloader("test", **kwargs)


class TradeS4FeatEmbeddings(_TradeFeatBase):
    _name_ = "s4_feat_embeddings"
    encoding = "embeddings"


class TradeS4FeatOneHot(_TradeFeatBase):
    _name_ = "s4_feat_onehot"
    encoding = "onehot"
