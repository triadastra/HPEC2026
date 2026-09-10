"""The burn-in contract: first valid target == start + lag_count + input_len.

A windowed pipeline with lag features has two distinct history requirements --
`lag_count` months so the lag columns hold real values instead of `fillna(0)`,
and `input_len` months so a full window fits behind the first target. Their sum
is the receptive field of one prediction, and it is the correct first-target
offset. It is NOT the number of rows to delete from the start of the panel: the
window builder already reserves the `input_len` part, so deleting it as well
applies that requirement twice and silently costs `input_len` target months on
every series.

`src/data/dataloader.py` (the WCTR-lineage `TradeDataPipeline`) had exactly that
defect: it trimmed `input_len + lag_count` months at YEAR granularity while
`TradeDataset._build_sequences` also emitted `target_idx = i + input_len`. With
input_len=36 / lag_count=12 and a 2008-01 start, the first target landed at
2015-01 instead of 2012-01.

These tests use a synthetic monthly timeline where the value at month index i is
i, so a lag's provenance is checkable by subtraction: lag_k at index i must be
i - k, and a fabricated lag appears as an exact 0.0 where a nonzero index was
expected. They depend on neither the real dataset nor any cohort/filtering step.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataloader import (DataConfig, TradeDataPipeline, TradeDataset,
                                 VALUE_CANON, WEIGHT_CANON)
from src.data.census_loader import CensusConfig, CensusLattice


def _panel(start, n_months):
    """One series, value at month index i == i."""
    times = pd.date_range(start, periods=n_months, freq="MS")
    return pd.DataFrame([
        {"State": "CA", "Commodity": "01", "Import/Export": "Import",
         "Time": t, VALUE_CANON: float(i), WEIGHT_CANON: float(i)}
        for i, t in enumerate(times)
    ])


def _build(start, input_len, lag_count, n_months=120):
    df = _panel(start, n_months)
    cfg = DataConfig(start=start, end=str(df["Time"].iloc[-1].date()),
                     input_len=input_len, lag_count=lag_count,
                     imports_dir="/nonexistent", exports_dir="/nonexistent")
    pipeline = TradeDataPipeline(cfg)
    lagged = pipeline.add_lags(df)
    feat_cols = [c for c in pipeline.get_feature_columns() if c in lagged.columns]
    dataset = TradeDataset(lagged, feat_cols, input_len,
                           {"CA": 0}, {"01": 0}, {"Import": 0})
    return lagged, dataset


# Non-January starts and lag counts that are not multiples of 12 are the cases
# the previous year-granularity trim got wrong in both directions: `// 12`
# rounded a 6-month burn-in down to zero years (admitting fabricated lags) and
# ignored the start month entirely.
GRID = [("2008-01-01", 36, 12), ("2008-01-01", 36, 6), ("2008-01-01", 24, 18),
        ("2008-07-01", 36, 12), ("2009-03-01", 12, 6), ("2008-01-01", 36, 1)]


@pytest.mark.parametrize("start,input_len,lag_count", GRID)
def test_first_target_is_start_plus_lag_count_plus_input_len(start, input_len, lag_count):
    n_months = 120
    _, dataset = _build(start, input_len, lag_count, n_months)
    targets = sorted({s["target_time"] for s in dataset.samples})
    expected = pd.Timestamp(start) + pd.DateOffset(months=lag_count + input_len)
    assert targets, "no windows were produced at all"
    assert targets[0] == expected
    # Every month from the first valid target to the end of the panel is usable.
    assert len(targets) == n_months - lag_count - input_len


@pytest.mark.parametrize("start,input_len,lag_count", GRID)
def test_no_retained_row_carries_a_fabricated_lag(start, input_len, lag_count):
    lagged, _ = _build(start, input_len, lag_count)
    origin = pd.Timestamp(start).to_period("M")
    for _, row in lagged.iterrows():
        i = (row["Time"].to_period("M") - origin).n
        for k in range(1, lag_count + 1):
            # value at index i-k; a fillna(0.0) fabrication shows up as 0.0
            assert row[f"value_lag_{k}"] == float(i - k)
            assert row[f"weight_lag_{k}"] == float(i - k)


def test_val_and_test_windows_reach_back_into_real_history():
    """Splits are formed by filtering target time, not by lagging each slice.

    If each split were lagged separately, the first `lag_count` rows of val and
    of test would carry fabricated zeros -- an evaluation-affecting defect.
    """
    start, input_len, lag_count = "2008-01-01", 36, 12
    lagged, _ = _build(start, input_len, lag_count)
    feat_cols = [c for c in TradeDataPipeline(
        DataConfig(start=start, end="2017-12-01", input_len=input_len,
                   lag_count=lag_count, imports_dir="/x", exports_dir="/x")
    ).get_feature_columns() if c in lagged.columns]

    val_start = pd.Timestamp("2015-01-01")
    val = TradeDataset(lagged, feat_cols, input_len, {"CA": 0}, {"01": 0},
                       {"Import": 0}, target_time_start=val_start)
    assert val.samples
    first = min(val.samples, key=lambda s: s["target_time"])
    assert first["target_time"] == val_start
    # The window feeding the first val target must be real history, and its
    # earliest row must itself carry real lags.
    window = first["x_numeric"].numpy()
    assert window.shape[0] == input_len
    assert not np.allclose(window[0], 0.0)


def test_census_lattice_reserves_history_instead_of_deleting_it(tmp_path):
    """The primary Census path satisfies the same contract.

    It never deletes rows: `burn` is the first TARGET index while every earlier
    month stays available as window history. This pins that behaviour so a
    future "trim the panel" refactor cannot reintroduce the defect.
    """
    n_series, n_months, n_channels = 2, 96, 2
    input_len, lag_count = 36, 12
    train_end, val_end = 72, 84

    panel = np.zeros((n_series, n_months, n_channels), np.float32)
    for t in range(n_months):
        panel[:, t, :] = t

    npz = tmp_path / "syn.npz"
    np.savez(npz, panel_norm=panel, panel_raw=panel,
             series_idx=np.array([[0, 0, 0], [0, 0, 1]], np.int64),
             norm_min=np.zeros((n_series, n_channels)),
             norm_max=np.ones((n_series, n_channels)),
             norm_range=np.ones((n_series, n_channels)),
             mask=np.ones((1, 1, 2), bool))

    lattice = CensusLattice(CensusConfig(
        npz=str(npz), input_len=input_len, lag_count=lag_count,
        train_end=train_end, val_end=val_end, allow_legacy_artifact=True))
    train, _, _ = lattice.get_dataloaders(batch_size=8, num_workers=0,
                                          shuffle_train=False)
    burn = input_len + lag_count
    target_indices = {int(s[1]) for s in train.dataset.samples}
    assert min(target_indices) == burn
    assert len(target_indices) == train_end - burn

    # The earliest row any window touches must already have real lags.
    assert burn - input_len >= lag_count

    # Lag provenance across every row a window can reach.
    col = n_channels
    for ch in lattice.lag_channels:
        for k in range(1, lag_count + 1):
            for t in range(lag_count, n_months):
                assert float(lattice.feat[0, t, col]) == float(t - k)
            col += 1
