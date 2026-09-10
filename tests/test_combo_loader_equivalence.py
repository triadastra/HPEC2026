"""``CensusComboDataset`` must stay bitwise identical to the naive formulation.

The combo window is ``(L, G, F)`` while the panel is stored ``(G, T, F)``, so
every sample needs the time axis in front. The straightforward way -- slice
``feat[:, t-L:t, :]`` and transpose -- yields a strided view whose
``ascontiguousarray`` is a full gather of the window: 143 MB at benchmark
scale, measured at ~51 ms per sample, several times the GPU cost of the step it
feeds. The dataset instead hoists that transpose into ``__init__``.

That is a pure performance change, and these tests pin it as one: the emitted
tensors must equal the naive construction bit for bit, in value, dtype and
shape. They also pin the aliasing contract the optimisation introduces --
``__getitem__`` now returns a view into the dataset's own panel, which is only
safe because collate copies.
"""

import numpy as np
import pytest
import torch

from src.data.census_loader import CensusComboDataset
from src.data.dataloader import combo_collate_fn

G, T, F, K, L = 40, 60, 12, 9, 8
LO, HI, TV, TW = 20, 45, 0, 5


def _fixtures():
    rng = np.random.default_rng(0)
    feat = rng.random((G, T, F)).astype(np.float32)
    panel = rng.random((G, T, K)).astype(np.float32)
    return feat, panel, CensusComboDataset(feat, panel, L, LO, HI, TV, TW)


def _naive(feat, panel, t):
    """The formulation the dataset is optimising away, written out."""
    x = feat[:, t - L:t, :].transpose(1, 0, 2)
    return {
        "x_numeric": torch.from_numpy(np.ascontiguousarray(x)),
        "group_mask": torch.ones(feat.shape[0], dtype=torch.float32),
        "target_value": torch.from_numpy(panel[:, t, TV].copy()),
        "target_weight": torch.from_numpy(panel[:, t, TW].copy()),
        "target_time": int(t),
    }


def test_every_sample_matches_the_naive_transpose_bitwise():
    feat, panel, ds = _fixtures()
    assert len(ds) == HI - LO
    for idx in range(len(ds)):
        got, want = ds[idx], _naive(feat, panel, LO + idx)
        assert got["target_time"] == want["target_time"]
        for key in ("x_numeric", "group_mask", "target_value", "target_weight"):
            a, b = got[key], want[key]
            assert a.shape == b.shape, key
            assert a.dtype == b.dtype, key
            assert torch.equal(a, b), key


def test_window_carries_the_right_months_in_the_right_order():
    """Guards against a transpose that is fast but axis-swapped.

    Bitwise equality above already implies this, but only against a reference
    that could itself be wrong. This checks the window against the panel
    directly: row i of the window is month t-L+i, for every series.
    """
    feat, _, ds = _fixtures()
    for idx in (0, len(ds) // 2, len(ds) - 1):
        t = LO + idx
        x = ds[idx]["x_numeric"].numpy()          # (L, G, F)
        assert x.shape == (L, G, F)
        for i in range(L):
            assert np.array_equal(x[i], feat[:, t - L + i, :])


def test_collate_copies_so_the_returned_view_never_reaches_the_model():
    """``__getitem__`` returns a view into the dataset's own arrays.

    That is safe only because ``combo_collate_fn`` stacks, which allocates. If
    collate ever stopped copying, an in-place op downstream would silently
    corrupt the panel for every later sample -- so pin it here rather than
    rely on it implicitly.
    """
    _, _, ds = _fixtures()
    batch = combo_collate_fn([ds[0], ds[1]])
    panel_storage = torch.from_numpy(ds.feat_t).untyped_storage().data_ptr()
    assert batch["x_numeric"].untyped_storage().data_ptr() != panel_storage
    assert (batch["group_mask"].untyped_storage().data_ptr()
            != ds._group_mask.untyped_storage().data_ptr())
    # and the copy is still the right data
    assert torch.equal(batch["x_numeric"][0], ds[0]["x_numeric"])


def test_shared_group_mask_is_all_ones_and_not_mutated_between_samples():
    _, _, ds = _fixtures()
    first = ds[0]["group_mask"]
    assert first.shape == (G,)
    assert first.dtype == torch.float32
    assert torch.equal(first, torch.ones(G))
    _ = ds[1], ds[2]
    assert torch.equal(ds[0]["group_mask"], torch.ones(G))


def _lattice(tmp_path):
    from src.data.census_loader import CensusConfig, CensusLattice
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


def test_persistent_workers_stays_off_on_every_census_loader(tmp_path):
    """Pins a trap: persistent_workers is NOT numerically neutral here.

    It looks free -- the combo loader emits ~96 batches an epoch, so respawning
    workers each epoch is real overhead. But with num_workers > 0 the
    DataLoader draws a base seed from `generator` each time it builds a worker
    iterator (once per epoch); persisting the workers draws it once per run.
    That same generator drives the sampler's shuffle, so the batch order
    changes from epoch 1 onward -- verified directly: epoch 0 matched, every
    later epoch differed, and end-to-end training loss moved.

    Any future "just turn on persistent_workers" must re-derive this, not
    assume it.
    """
    cl = _lattice(tmp_path)
    gen = torch.Generator().manual_seed(947)
    for combo in (False, True):
        for loader in cl.get_dataloaders(batch_size=2, num_workers=2,
                                         generator=gen, combo=combo):
            assert loader.persistent_workers is False


def test_batch_order_is_reproducible_from_the_seed(tmp_path):
    """Same generator seed must give the same order, epoch after epoch."""
    cl = _lattice(tmp_path)

    def order():
        gen = torch.Generator().manual_seed(947)
        train, _, _ = cl.get_dataloaders(batch_size=2, num_workers=0,
                                         generator=gen, combo=True)
        return [[int(t) for t in b["target_time"]] for _ in range(3) for b in train]

    assert order() == order()


@pytest.mark.parametrize("combo,batch", [(True, 2), (False, 32)])
def test_worker_count_does_not_change_the_batch_sequence(tmp_path, combo, batch):
    """The claim the worker-count-in-the-environment design rests on.

    Worker count is passed by environment variable rather than argv precisely
    so that retuning it does not enter run_fingerprint and invalidate finished
    runs. That is only sound if it provably cannot change what is trained --
    and its close cousin persistent_workers provably DOES (see above), so this
    is asserted rather than assumed.

    The DataLoader draws its base seed from `generator` once per iterator
    construction regardless of worker count, and neither census dataset uses
    RNG in __getitem__, so every worker count yields an identical sequence.
    Verified here over two epochs at 0 and 2 workers; 1 and 4 were checked the
    same way by hand and also matched.
    """
    cl = _lattice(tmp_path)

    def order(num_workers):
        gen = torch.Generator().manual_seed(947)
        train, _, _ = cl.get_dataloaders(batch_size=batch, num_workers=num_workers,
                                         generator=gen, combo=combo)
        return [[int(t) for t in b["target_time"]]
                for _ in range(2) for b in train]

    reference = order(0)
    assert reference, "fixture produced no batches"
    assert order(2) == reference, (
        f"combo={combo}: 2 workers changed the batch order; worker count is "
        "NOT safe to pass outside the run fingerprint")


def test_worker_count_does_not_change_the_tensors_themselves(tmp_path):
    """Order is not enough: the payload must be bitwise identical too."""
    cl = _lattice(tmp_path)

    def first_batch(num_workers):
        gen = torch.Generator().manual_seed(947)
        train, _, _ = cl.get_dataloaders(batch_size=2, num_workers=num_workers,
                                         generator=gen, combo=True)
        return next(iter(train))

    reference = first_batch(0)
    got = first_batch(2)
    assert set(got) == set(reference)
    for key, value in reference.items():
        if torch.is_tensor(value):
            assert torch.equal(got[key], value), f"{key} differs with workers"
