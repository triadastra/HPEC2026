"""Same seed must give the same run, bit for bit.

README states "same-seed training is bit-reproducible" and the whole
seed-matrix design rests on it -- three seeds are treated as three samples of
the same procedure, which is only true if the procedure is deterministic given
the seed. Nothing tested it end to end.

Verified by hand at the script level first: two `scripts/train.py` invocations
with identical arguments produced best.pth files with the same SHA-256, zero
differing tensors of 23, and training curves identical to ten decimals. These
tests pin the mechanism that makes that hold, cheaply enough to run every time.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.census_loader import CensusConfig, CensusLattice
from src.models import create_model
from src.utils import compose_config, model_kwargs_from_config, set_seed


def _lattice(tmp_path):
    n, t, k = 12, 60, 2
    panel = np.random.default_rng(1).random((n, t, k)).astype(np.float32)
    npz = tmp_path / "l.npz"
    np.savez(npz, panel_norm=panel, panel_raw=panel,
             series_idx=np.array([[i, 0, i % 2] for i in range(n)], np.int64),
             norm_min=np.zeros((n, k)), norm_max=np.ones((n, k)),
             norm_range=np.ones((n, k)), mask=np.ones((n, 1, 2), bool))
    return CensusLattice(CensusConfig(npz=str(npz), input_len=8, lag_count=4,
                                      train_end=40, val_end=50,
                                      allow_legacy_artifact=True))


def _build(cl, model="gru", variant="embeddings"):
    kw = model_kwargs_from_config(
        compose_config(model, variant, extra=["config/census.yaml"]))
    kw.update(num_numeric_features=cl.features_per_group, num_states=cl.num_states,
              num_commodities=cl.num_commodities, num_flows=cl.num_flows)
    return create_model(model, variant, **kw)


def _weights(cl, seed):
    set_seed(seed)
    return {k: v.clone() for k, v in _build(cl).state_dict().items()}


def test_the_same_seed_initialises_the_same_weights(tmp_path):
    cl = _lattice(tmp_path)
    a, b = _weights(cl, 947), _weights(cl, 947)
    assert set(a) == set(b)
    differing = [k for k in a if not torch.equal(a[k], b[k])]
    assert not differing, f"initialisation is not deterministic: {differing[:3]}"


def test_different_seeds_initialise_differently(tmp_path):
    """Otherwise the seed matrix would be three copies of one run, and the
    seed-averaged standard deviations would be meaningless."""
    cl = _lattice(tmp_path)
    a, b = _weights(cl, 947), _weights(cl, 732)
    assert any(not torch.equal(a[k], b[k]) for k in a), "seeds are not distinct"


@pytest.mark.parametrize("seed", [947, 732, 619])
def test_a_training_epoch_is_deterministic_given_the_seed(tmp_path, seed):
    """Initialisation is not enough: the batch order and the updates
    themselves have to be reproducible too."""
    cl = _lattice(tmp_path)

    def one_epoch():
        set_seed(seed)
        model = _build(cl)
        gen = torch.Generator().manual_seed(seed)
        train, _, _ = cl.get_dataloaders(batch_size=8, num_workers=0,
                                         generator=gen, combo=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for batch in train:
            opt.zero_grad(set_to_none=True)
            out = model(batch["x_numeric"], batch["state_ids"],
                        batch["comm_ids"], batch["flow_ids"])
            tgt = torch.stack([batch["target_value"], batch["target_weight"]], -1)
            loss = torch.nn.functional.mse_loss(out.reshape(tgt.shape), tgt)
            loss.backward(); opt.step()
            losses.append(loss.item())
        return losses, {k: v.clone() for k, v in model.state_dict().items()}

    la, wa = one_epoch()
    lb, wb = one_epoch()
    assert la == lb, "per-batch losses diverged between identical runs"
    differing = [k for k in wa if not torch.equal(wa[k], wb[k])]
    assert not differing, f"weights diverged after one epoch: {differing[:3]}"


def test_the_seeds_the_benchmark_declares_are_actually_distinct():
    from scripts.sweep import DEFAULT_SEEDS
    assert len(set(DEFAULT_SEEDS)) == len(DEFAULT_SEEDS) == 3
