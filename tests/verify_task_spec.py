"""Mechanical verification of the paper's task definition against the live code.

Asserts, from the actual loader and model factory (no mocks):
lattice/feature/channel dimensions, target channels, strict causality of the
input window, identical evaluation coverage for flat vs multidimensional
paths, and that every model head emits exactly (value, weight).

    CUDA_VISIBLE_DEVICES="" python tests/verify_task_spec.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data.census_loader import CensusLattice, CensusConfig  # noqa: E402
from src.models import create_model  # noqa: E402

ok = [0, 0]


def check(name, cond):
    ok[0] += bool(cond)
    ok[1] += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


cl = CensusLattice(CensusConfig(npz=str(REPO / "data/census_port/processed/census_lattice_9ch.npz")))

check("non-empty train-selected cohort", cl.N > 0)
check("cohort selection uses training months only",
      cl.metadata.get("cohort_selection", {}).get("state_and_commodity_ranking")
      == "training_months_only")
check("stored normalization range matches the floored builder formula",
      np.allclose(cl.norm_range, np.maximum(
          np.load(cl.config.npz)["norm_max"] - cl.norm_min, 1.0)))
check("192 months", cl.T == 192)
check("35 features/timestep (9ch + 24 lags + sin/cos)", cl.features_per_group == 35)
check("9 channels", cl.K == 9)
check("5 value + 4 weight channels",
      len([c for c in cl.channels if c.endswith("_value")]) == 5
      and len([c for c in cl.channels if c.endswith("_weight")]) == 4)
check("targets = channels [0,5] = agg_value, agg_weight",
      cl.target_ch == [0, 5] and cl.channels[0] == "agg_value" and cl.channels[5] == "agg_weight")

_, _, test_flat = cl.get_dataloaders(batch_size=512, num_workers=0, combo=False, shuffle_train=False)
ds = test_flat.dataset
b = ds[0]
x = b["x_numeric"].numpy()
tv, tw, tt = float(b["target_value"]), float(b["target_weight"]), int(b["target_time"])
sm = np.where((np.abs(cl.panel[:, tt, 0] - tv) < 1e-6)
              & (np.abs(cl.panel[:, tt, 5] - tw) < 1e-6))[0]
si = int(sm[0])
check("window == feat[s, t-36:t] (target month excluded)",
      np.allclose(x, cl.feat[si, tt - 36:tt, :], atol=1e-6))
check("last input row is month t-1", np.allclose(x[-1], cl.feat[si, tt - 1, :], atol=1e-6))
check("first test target month index = 168 (2024-01)", tt == 168)
check("flat test set = N x 24 samples", len(ds) == cl.N * 24)
_, _, test_combo = cl.get_dataloaders(batch_size=1, num_workers=0, combo=True, shuffle_train=False)
check("combo test set = 24 lattice windows", len(test_combo.dataset) == 24)

kw = dict(num_numeric_features=cl.features_per_group, num_states=cl.num_states,
          num_commodities=cl.num_commodities, num_flows=cl.num_flows,
          hidden_size=128, num_layers=4,
          state_embed_dim=7, comm_embed_dim=88, flow_embed_dim=2)
ckw = dict(kw, num_combos=cl.num_combos, features_per_group=cl.features_per_group,
           combo_coords=cl.combo_coords, lattice_dims=cl.lattice_dims,
           combo_encoder="embeddings")


def head_out(m):
    layer = getattr(m, "output_proj", None) or getattr(m, "output_layer", None)
    if isinstance(layer, nn.Sequential):
        lins = [x for x in layer.modules() if isinstance(x, nn.Linear)]
        return lins[-1].out_features if lins else -1
    return getattr(layer, "out_features", -1)


heads = []
for mdl, var, kws in [("gru", "embeddings", kw), ("lstm", "onehot", kw),
                      ("transformer", "embeddings", kw), ("s4", "embeddings", kw),
                      ("gru", "fa_3d", ckw), ("transformer", "asa_2d", ckw),
                      ("gru", "fa_3d", ckw),
                      ("s4nd", "grid_4d", ckw)]:
    m = create_model(mdl, var, **kws)
    heads.append((f"{mdl}/{var}", head_out(m)))
    del m
check("all heads emit exactly 2 (value, weight): "
      + ", ".join(f"{n}={o}" for n, o in heads),
      all(o == 2 for _, o in heads))

m = create_model("gru", "embeddings", **kw).eval()
bb = next(iter(test_flat))
with torch.no_grad():
    out = m(bb["x_numeric"][:1], bb["state_ids"][:1], bb["comm_ids"][:1], bb["flow_ids"][:1])
check(f"flat forward -> (1, 2), got {tuple(out.shape)}", tuple(out.shape) == (1, 2))

print(f"\n{ok[0]}/{ok[1]} checks passed")
sys.exit(0 if ok[0] == ok[1] else 1)
