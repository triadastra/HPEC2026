"""Regression tests for the benchmark review findings."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_census_lattice import (_national_aggregate, _select_cohort,
                                          accumulate)
from scripts.dump_errors import observation_losses
from scripts.evaluate import (_metrics, _pooled_rolling_rows, _rank_quartiles,
                              parse_run, rolling_score_errors,
                              seed_matrix_errors)
from scripts.model_cost import (AGGREGATE, COMBO, COMBO_ID, FLAT,
                                cost_exit_code)
from scripts.sweep import (RETRAIN_SETS, _protocol_argv, build_matrix,
                           flat_accum, rolling_aggregate_folds, run_state,
                           schedule, select)
from scripts.xgb_agg import window_xy
from src.data.census_loader import CensusConfig, CensusLattice, ExactGroupBatchSampler
from src.training.trainer import Trainer
from src.utils.run_manifest import (run_fingerprint, validate_completion,
                                    write_completion)


def _raw_series(values):
    out = np.zeros((len(values), 7), dtype=np.float64)
    out[:, 0] = values
    return out


def test_cohort_selection_ignores_validation_and_test_availability():
    base = {
        ("a", "AA", "export"): _raw_series([1, 1, 1, 1, 0, 0]),
        ("b", "AA", "export"): _raw_series([0, 0, 0, 0, 1, 1]),
    }
    changed_future = {key: value.copy() for key, value in base.items()}
    changed_future[("a", "AA", "export")][4:, 0] = 999
    changed_future[("b", "AA", "export")][4:, 0] = 0
    kwargs = dict(train_months=4, n_states=1, target_density=0.5, filter_frac=0.75)
    first = _select_cohort(base, **kwargs)[2]
    second = _select_cohort(changed_future, **kwargs)[2]
    assert first == second == [("a", "AA", "export")]


def test_inverse_uses_stored_floored_normalization_range():
    cl = object.__new__(CensusLattice)
    cl.target_ch = [0, 1]
    cl.norm_min = np.array([[2.0, 3.0]])
    cl.norm_range = np.array([[1.0, 4.0]])
    got = cl.inverse_targets(np.array([[0.5, 0.25]]), np.array([0]))
    np.testing.assert_allclose(got, np.expm1([[2.5, 4.0]]))


def _write_test_lattice(tmp_path, *, safe):
    npz = tmp_path / "lattice.npz"
    panel_raw = np.arange(12, dtype=np.float32).reshape(1, 6, 2)
    logp = np.log1p(panel_raw)
    norm_min = logp[:, :4].min(axis=1)
    norm_max = logp[:, :4].max(axis=1)
    norm_range = np.maximum(norm_max - norm_min, 1.0)
    arrays = dict(
        panel_norm=(logp - norm_min[:, None]) / norm_range[:, None],
        panel_raw=panel_raw,
        # Deliberately differs from the selected panel so the rolling test
        # proves that aggregate_raw, not panel_raw, is used.
        aggregate_raw=panel_raw[0] + 100.0,
        series_idx=np.zeros((1, 3), dtype=np.int32),
        norm_min=norm_min,
        norm_max=norm_max,
        mask=np.ones((1, 1, 1), dtype=bool),
    )
    if safe:
        arrays["norm_range"] = norm_range
    np.savez(npz, **arrays)
    metadata = {
        "channels": ["agg_value", "agg_weight"],
        "target_channels": [0, 1],
        "splits": {"train": [0, 4], "val": [4, 5], "test": [5, 6]},
    }
    if safe:
        metadata["cohort_selection"] = {
            "n_months": 4,
            "state_and_commodity_ranking": "training_months_only",
        }
        metadata["aggregate_source"] = {
            "scope": "all_raw_series_before_cohort_selection",
            "channels": ["agg_value", "agg_weight"],
            "n_source_series": 2,
        }
    npz.with_suffix(".json").write_text(json.dumps(metadata))
    return npz


def test_official_loader_rejects_pre_fix_lattice(tmp_path):
    npz = _write_test_lattice(tmp_path, safe=False)
    cfg = CensusConfig(npz=str(npz), input_len=2, lag_count=1,
                       train_end=4, val_end=5)
    with pytest.raises(ValueError, match="unsafe Census lattice"):
        CensusLattice(cfg)


def test_official_loader_accepts_train_only_contract(tmp_path):
    npz = _write_test_lattice(tmp_path, safe=True)
    cfg = CensusConfig(npz=str(npz), input_len=2, lag_count=1,
                       train_end=4, val_end=5)
    lattice = CensusLattice(cfg)
    assert lattice.metadata["cohort_selection"]["n_months"] == 4


def test_aggregate_loader_rejects_unproven_pre_cohort_target(tmp_path):
    npz = _write_test_lattice(tmp_path, safe=True)
    metadata = json.loads(npz.with_suffix(".json").read_text())
    metadata.pop("aggregate_source")
    npz.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="before cohort selection"):
        CensusLattice(CensusConfig(
            npz=str(npz), input_len=2, lag_count=1,
            train_end=4, val_end=5, aggregate=True,
        ))


def test_rolling_aggregate_refits_train_only_and_bounds_test_window(tmp_path):
    npz = _write_test_lattice(tmp_path, safe=True)
    cfg = CensusConfig(
        npz=str(npz), input_len=1, lag_count=1,
        train_end=3, val_end=4, test_end=5,
        aggregate=True, refit_normalization=True,
    )
    lattice = CensusLattice(cfg)
    expected_log = np.log1p(lattice.panel_raw[0, :3])
    np.testing.assert_allclose(lattice.norm_min[0], expected_log.min(axis=0))
    assert lattice.T == lattice.test_end == 5
    train, val, test = lattice.get_dataloaders(
        batch_size=32, num_workers=0, shuffle_train=False,
    )
    assert len(train.dataset) == len(val.dataset) == len(test.dataset) == 1
    assert max(test.dataset.samples[:, 1]) < lattice.test_end


def test_national_aggregate_is_independent_of_cohort_selection():
    first = _raw_series([1, 1, 1, 1, 0, 0])
    second = _raw_series([0, 0, 0, 0, 9, 9])
    raw = {
        ("kept", "AA", "export"): first,
        ("future_sparse", "AA", "export"): second,
    }
    aggregate = _national_aggregate(raw, 6)
    selected = _select_cohort(
        raw, train_months=4, n_states=1, target_density=0.5, filter_frac=0.75,
    )[2]
    assert selected == [("kept", "AA", "export")]
    np.testing.assert_array_equal(aggregate[:, 0], first[:, 0] + second[:, 0])


def test_missing_monthly_archives_are_unknown_not_zero_trade(tmp_path):
    with pytest.raises(FileNotFoundError, match="unknown data"):
        accumulate(tmp_path, {}, 6)
    months, raw, missing = accumulate(
        tmp_path, {}, 6, allow_missing_months=True
    )
    assert len(months) == 192
    assert not raw
    assert len(missing) == 384


def test_official_loader_rejects_diagnostic_missing_month_build(tmp_path):
    npz = _write_test_lattice(tmp_path, safe=True)
    metadata = json.loads(npz.with_suffix(".json").read_text())
    metadata["source_missing_archives"] = ["missing-export.ZIP"]
    npz.with_suffix(".json").write_text(json.dumps(metadata))
    cfg = CensusConfig(npz=str(npz), input_len=2, lag_count=1,
                       train_end=4, val_end=5)
    with pytest.raises(ValueError, match="missing monthly source"):
        CensusLattice(cfg)


def test_official_loader_rejects_mismatched_split_contract(tmp_path):
    npz = _write_test_lattice(tmp_path, safe=True)
    metadata = json.loads(npz.with_suffix(".json").read_text())
    metadata["splits"]["val"] = [4, 6]
    npz.with_suffix(".json").write_text(json.dumps(metadata))
    cfg = CensusConfig(npz=str(npz), input_len=2, lag_count=1,
                       train_end=4, val_end=5)
    with pytest.raises(ValueError, match="val split"):
        CensusLattice(cfg)


def test_legacy_lattice_requires_explicit_opt_in(tmp_path):
    npz = _write_test_lattice(tmp_path, safe=False)
    cfg = CensusConfig(npz=str(npz), input_len=2, lag_count=1,
                       train_end=4, val_end=5, allow_legacy_artifact=True)
    lattice = CensusLattice(cfg)
    assert lattice.norm_range.shape == lattice.norm_min.shape


def test_smape_uses_explicit_raw_values_and_combined_rmse_is_pooled():
    m = _metrics(np.array([0.1]), np.array([0.0]),
                 smape_pred=np.array([110.0]), smape_true=np.array([100.0]))
    assert np.isclose(m["sMAPE"], 2 * 10 / 210 * 100)
    pooled = _metrics(np.array([[1.0, 3.0]]), np.zeros((1, 2)))
    assert np.isclose(pooled["RMSE"], np.sqrt(5.0))


def test_incomplete_seed_cohort_is_rejected():
    configs = {
        "gru_embeddings_1d_s947": {
            "model": "gru", "variant": "embeddings", "enc": "embeddings", "seed": 947,
        }
    }
    errors = seed_matrix_errors(configs, [947, 732, 619])
    assert errors and "incomplete seeds" in errors[0]


def test_profile_quartiles_never_split_ties():
    values = np.array([1, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4], dtype=float)
    bins = _rank_quartiles(values)
    for value in np.unique(values):
        assert len(np.unique(bins[values == value])) == 1
    assert np.all(_rank_quartiles(np.ones(20)) == 0)


def test_exact_group_sampler_retains_all_samples_and_matches_steps():
    sampler = ExactGroupBatchSampler(
        dataset_size=18, micro_batch_size=4, effective_batch_size=6,
        shuffle=False,
    )
    batches = list(sampler)
    assert [len(batch) for batch in batches] == [4, 2, 4, 2, 4, 2]
    assert sorted(index for batch in batches for index in batch) == list(range(18))
    assert len(batches) // sampler.micro_batches_per_group == 3


def test_exact_census_groups_drive_exact_optimizer_steps(tmp_path):
    n, t, k = 6, 8, 2
    log_panel = np.linspace(0.0, 1.0, n * t * k, dtype=np.float32).reshape(n, t, k)
    panel_raw = np.expm1(log_panel)
    norm_min = log_panel[:, :5].min(axis=1)
    norm_range = np.maximum(log_panel[:, :5].max(axis=1) - norm_min, 1.0)
    panel = (log_panel - norm_min[:, None, :]) / norm_range[:, None, :]
    series_idx = np.array(
        [(commodity, 0, flow) for commodity in range(3) for flow in range(2)],
        dtype=np.int32,
    )
    mask = np.ones((3, 1, 2), dtype=bool)
    npz = tmp_path / "tiny.npz"
    np.savez(
        npz, panel_norm=panel, panel_raw=panel_raw, series_idx=series_idx,
        mask=mask, norm_min=norm_min, norm_max=norm_min + norm_range,
        norm_range=norm_range,
    )
    npz.with_suffix(".json").write_text(json.dumps({
        "channels": ["agg_value", "agg_weight"], "target_channels": [0, 1],
        "states": ["AA"], "commodities": ["a", "b", "c"],
        "flows": ["export", "import"],
        "cohort_selection": {
            "n_months": 5,
            "state_and_commodity_ranking": "training_months_only",
        },
        "splits": {"train": [0, 5], "val": [5, 6], "test": [6, 8]},
    }))
    cl = CensusLattice({
        "npz": str(npz), "input_len": 2, "lag_count": 1,
        "train_end": 5, "val_end": 6,
    })
    train, val, _ = cl.get_dataloaders(
        batch_size=4, num_workers=0, shuffle_train=False,
        effective_batch_size=6,
    )
    full_train, _, _ = cl.get_dataloaders(
        batch_size=6, num_workers=0, shuffle_train=False,
    )

    class TinyModel(torch.nn.Module):
        requires_combo_loader = False

        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(cl.features_per_group, 2)

        def forward(self, x, state_ids, comm_ids, flow_ids):
            return self.head(x[:, -1])

        def get_num_params(self):
            return sum(parameter.numel() for parameter in self.parameters())

    model = TinyModel()
    reference = TinyModel()
    reference.load_state_dict(model.state_dict())
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    for batch in full_train:
        reference_optimizer.zero_grad()
        prediction = reference(
            batch["x_numeric"], batch["state_ids"], batch["comm_ids"], batch["flow_ids"]
        )
        target = torch.stack([batch["target_value"], batch["target_weight"]], dim=-1)
        torch.nn.functional.mse_loss(prediction, target).backward()
        reference_optimizer.step()

    trainer = Trainer({
        "training": {
            "device": "cpu", "batch_size": 4, "grad_accum": 2,
            "effective_batch_size": 6, "scheduler": None,
            "optimizer": "sgd", "lr": 0.01,
        },
        "checkpointing": {"save_dir": str(tmp_path / "checkpoints")},
        "logging": {"log_dir": str(tmp_path / "logs")},
    })
    trainer.setup(model, train, val)
    trainer.train_epoch()
    assert trainer.steps_per_epoch == 2
    assert len(train) == 4
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_sweep_routes_npz_and_exact_effective_batch():
    accum = flat_accum(28_292, 2_048)
    runs = build_matrix({"2"}, [947], 2_048, 1, accum=accum,
                        npz="custom/lattice.npz", effective_batch_size=28_292)
    # 6 flat 1-D + 3 hosts x |MECHS| x 3 dims. fa_local left the roster, so
    # this tracks two mechanisms, not three.
    assert len(runs) == 24
    for _, argv in runs:
        assert argv[argv.index("--npz") + 1] == "custom/lattice.npz"
    flat = [argv for name, argv in runs if "_1d_" in name]
    assert flat and all(argv[argv.index("--effective-batch-size") + 1] == "28292"
                        for argv in flat)


def test_test_1_1_declares_six_annual_aggregate_folds():
    folds = rolling_aggregate_folds()
    assert folds == [
        {"test_year": year, "train_end": 108 + 12 * offset,
         "val_end": 120 + 12 * offset, "test_end": 132 + 12 * offset}
        for offset, year in enumerate(range(2020, 2026))
    ]
    runs = build_matrix({"1.1"}, [947, 732, 619], 2_048, 1)
    assert len(runs) == 108
    assert len({name for name, _ in runs}) == 108
    for name, argv in runs:
        assert "_aggregate_roll_y" in name
        assert "--aggregate" in argv and "--refit-normalization" in argv
        assert "--fresh-model-session" in argv
        assert int(argv[argv.index("--test-end") + 1]) - int(
            argv[argv.index("--val-end") + 1]
        ) == 12


def test_test_1_1_run_name_parses_to_embedding_model_and_year():
    cfg = parse_run("gru_aggregate_roll_y2023_s947")
    assert cfg["rolling"] and cfg["test_year"] == 2023
    assert cfg["variant"] == "aggregate_roll_y2023"
    assert cfg["build_variant"] == "embeddings"


def test_rolling_pool_uses_pooled_mse_for_rmse():
    rows = []
    for year, mse in ((2020, 1.0), (2021, 9.0)):
        row = {
            "run": f"gru_aggregate_roll_y{year}_s947", "model": "gru",
            "variant": f"aggregate_roll_y{year}", "enc": "aggregate", "seed": 947,
        }
        for channel in ("value", "weight", "both"):
            row.update({
                f"{channel}_MAE": mse,
                f"{channel}_MSE": mse,
                f"{channel}_RMSE": np.sqrt(mse),
                f"{channel}_sMAPE": mse,
                f"{channel}_MASE": mse,
            })
        rows.append(row)
    pooled = _pooled_rolling_rows(rows)[0]
    assert pooled["n_folds"] == 2
    assert np.isclose(pooled["both_MSE"], 5.0)
    assert np.isclose(pooled["both_RMSE"], np.sqrt(5.0))


def test_every_rolling_run_requires_one_twelve_month_test_score():
    configs = {
        "gru_aggregate_roll_y2020_s947": {
            "rolling": True,
        }
    }
    assert rolling_score_errors(configs, [])
    row = {
        "run": "gru_aggregate_roll_y2020_s947",
        "val_end": 120,
        "test_end": 132,
        "n_test_months": 12,
        "model_session_id": "session-a",
    }
    assert rolling_score_errors(configs, [row]) == []
    assert "instead of 12" in rolling_score_errors(
        configs, [{**row, "n_test_months": 13}]
    )[0]


def test_filtered_execution_still_declares_complete_experiment(tmp_path):
    declared = build_matrix({"2"}, [947], 4, 1, accum=2,
                            npz="custom/lattice.npz", effective_batch_size=6)
    schedule([], [], epochs=1, logdir=tmp_path / "sweep" / "logs", dry=False,
             reset_manifest=True, declared_runs=declared)
    manifest = json.loads((tmp_path / "sweep" / "manifest.json").read_text())
    assert manifest["schema_version"] == 3
    assert len(manifest["runs"]) == 24
    assert all("fingerprint" in entry and "input_fingerprint" in entry
               for entry in manifest["runs"])


def test_manifest_does_not_mix_old_and_new_training_protocols(tmp_path):
    first = build_matrix({"2"}, [947], 4, 1, accum=2,
                         npz="custom/lattice.npz", effective_batch_size=6)
    logdir = tmp_path / "sweep" / "logs"
    schedule([], [], epochs=30, logdir=logdir, dry=False,
             reset_manifest=True, declared_runs=first)
    second = build_matrix({"3"}, [947], 8, 1, accum=1,
                          npz="replacement/lattice.npz", effective_batch_size=6)
    schedule([], [], epochs=200, logdir=logdir, dry=False,
             declared_runs=second)
    manifest = json.loads((tmp_path / "sweep" / "manifest.json").read_text())
    assert {entry["name"] for entry in manifest["runs"]} == {
        name for name, _ in second
    }
    for entry in manifest["runs"]:
        argv = entry["argv"]
        assert argv[argv.index("--epochs") + 1] == "200"
        assert argv[argv.index("--npz") + 1] == "replacement/lattice.npz"


def test_manifest_protocol_ignores_only_the_python_environment():
    base = {"argv": ["/env-a/python", "scripts/train.py", "--epochs", "200"]}
    other_env = {"argv": ["/env-b/python", "scripts/train.py", "--epochs", "200"]}
    other_script = {"argv": ["/env-a/python", "scripts/other.py", "--epochs", "200"]}
    assert _protocol_argv(base) == _protocol_argv(other_env)
    assert _protocol_argv(base) != _protocol_argv(other_script)


def test_completion_record_binds_checkpoint_to_run_fingerprint(tmp_path):
    (tmp_path / "base.yaml").write_text("training: {}\n")
    (tmp_path / "census.yaml").write_text("data: {}\n")
    (tmp_path / "lattice.npz").write_bytes(b"npz-v1")
    (tmp_path / "lattice.json").write_text(json.dumps({"n_series": 6}))
    argv = ["python", "train.py", "--config", "base.yaml", "--data-config",
            "census.yaml", "--npz", "lattice.npz", "--out-dir", "runs/demo"]
    entry = {"name": "demo", "argv": argv,
             "fingerprint": run_fingerprint(argv, tmp_path)}
    checkpoint = tmp_path / "runs" / "demo" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-v1")
    write_completion(tmp_path / "runs", entry)
    assert validate_completion(tmp_path / "runs", entry) is None
    checkpoint.write_bytes(b"checkpoint-v2")
    assert "hash" in validate_completion(tmp_path / "runs", entry)


def test_rolling_completion_is_bound_to_its_fresh_model_session(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "rolling"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pth").write_bytes(b"checkpoint")
    (run_dir / "model_session.json").write_text(json.dumps({
        "session_id": "session-a",
        "fresh_initialization": True,
        "checkpoint_loaded": False,
    }))
    entry = {
        "name": "rolling", "fingerprint": "fingerprint",
        "argv": ["python", "train.py", "--fresh-model-session"],
    }
    write_completion(runs_dir, entry)
    assert validate_completion(runs_dir, entry) is None
    (run_dir / "model_session.json").write_text(json.dumps({
        "session_id": "session-b",
        "fresh_initialization": True,
        "checkpoint_loaded": False,
    }))
    assert "different model session" in validate_completion(runs_dir, entry)


def test_retrain_slices_track_current_mixer_matrix():
    runs = build_matrix(
        {"1", "1.1", "2", "3", "4", "6"},
        [947, 732, 619], 2_048, 1, accum=14,
        effective_batch_size=28_292,
    )
    # fa_local was dropped from the roster: every FA claim now rests on the
    # authors' own components, so the declared matrix is two mixers, not three.
    # 477 = 18 (t1) + 108 (t1.1) + 72 (t2) + 81 (t3) + 108 (t4) + 90 (t6).
    # Test 5's 144 are gone: Exp 0 replaced it (see sweep.RETIRED_TESTS).
    assert len(runs) == 477
    assert {name: len(select(runs, pattern))
            for name, pattern in RETRAIN_SETS.items()} == {
                "f1": 36, "f2": 6, "f3": 63, "mandatory": 99,
            }


def test_sweep_status_uses_provenance_completion_state(tmp_path):
    runs_dir = tmp_path / "sweep"
    logdir = runs_dir / "logs"
    logdir.mkdir(parents=True)
    entry = {"name": "demo", "fingerprint": "declared-fingerprint"}
    checkpoint = runs_dir / "demo" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    write_completion(runs_dir, entry)
    (logdir / "demo.log").write_text("Final val loss: 0.1\n")
    assert run_state(entry, runs_dir, logdir) == "done"

    checkpoint.write_bytes(b"different checkpoint")
    assert run_state(entry, runs_dir, logdir) == "stale"
    checkpoint.unlink()
    assert run_state(entry, runs_dir, logdir) == "partial"
    (logdir / "demo.log").write_text("Traceback: training failed\n")
    assert run_state(entry, runs_dir, logdir) == "failed"
    assert run_state({"name": "new", "fingerprint": "new"},
                     runs_dir, logdir) == "todo"


_TRITON_WARNING = (
    "/repo/src/models/__init__.py:28: UserWarning: Mamba-2 unavailable "
    "(ModuleNotFoundError(\"No module named 'triton'\")); "
    "'mamba2' model not registered.\n"
    "  _warnings.warn(f\"Mamba-2 unavailable ({_mamba_err!r}); ...\")\n"
)


def test_sweep_status_ignores_exception_names_quoted_in_warnings(tmp_path):
    """A finished run must not read as ``failed`` over benign warning text.

    ``src/models/__init__.py`` warns about the unregistered Mamba models on
    EVERY run of an environment without Triton -- exactly the non-Mamba half of
    the split-environment sweep RETRAIN.md describes -- and that warning quotes
    ``ModuleNotFoundError``. A substring test for "Error" therefore reported
    every not-yet-``done`` cell in that environment as ``failed``.
    """
    runs_dir = tmp_path / "sweep"
    logdir = runs_dir / "logs"
    logdir.mkdir(parents=True)
    entry = {"name": "gru_aggregate_s947", "fingerprint": "declared"}
    checkpoint = runs_dir / entry["name"] / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    write_completion(runs_dir, entry)
    log = logdir / f"{entry['name']}.log"
    log.write_text(_TRITON_WARNING + "Final val loss: 0.1\n")
    assert run_state(entry, runs_dir, logdir) == "done"

    # Same run under a moved declaration (e.g. a different --epochs): the run
    # completed, only its manifest entry changed.
    moved = {"name": entry["name"], "fingerprint": "redeclared"}
    assert run_state(moved, runs_dir, logdir) == "stale"

    # An exception name inside ordinary progress output is not a failure.
    log.write_text(_TRITON_WARNING
                   + "Epoch 1 | Errors: 0 | no OutOfMemoryError seen\n"
                   + "Final val loss: 0.1\n")
    assert run_state(moved, runs_dir, logdir) == "stale"

    # Real failure signals still register, under the same warning preamble.
    log.write_text(_TRITON_WARNING
                   + "Traceback (most recent call last):\n"
                   + '  File "scripts/train.py", line 1, in <module>\n'
                   + "FileNotFoundError: no lattice\n")
    assert run_state(moved, runs_dir, logdir) == "failed"
    log.write_text(_TRITON_WARNING
                   + "\nSWEEP-FAIL: process produced no new best.pth\n")
    assert run_state(moved, runs_dir, logdir) == "failed"

    # A traceback flushed straight after a progress-bar line has no newline
    # before it, only a carriage return.
    log.write_text("100%|####| 5/5\rtorch.cuda.OutOfMemoryError: CUDA OOM\n")
    assert run_state(moved, runs_dir, logdir) == "failed"


def test_cost_panel_covers_experiments_one_two_and_three():
    assert len(AGGREGATE) == 6
    assert {(model, enc) for model, enc in FLAT} == {
        (model, enc)
        for model in ("gru", "lstm", "transformer", "s4", "mamba2", "mamba3")
        for enc in ("onehot", "embeddings")
    }
    for model in ("gru", "lstm", "transformer"):
        for mech in ("asa", "fa"):
            for dims in (2, 3, 4):
                variant = f"{mech}_{dims}d"
                assert (model, variant, "onehot") in COMBO
                assert (model, variant, "embeddings") in COMBO


def test_cost_panel_includes_identity_aware_test_four_cells():
    # Test 4 = the axial grid x --axis-identity. The panel must cover every
    # default mechanism, plus the opt-in arms: the cost table is static, so
    # measuring it is what makes "is ACA worth 144 runs?" (and "is the
    # identity-aware hybrid arm worth 72?") an informed decision.
    required = {
        (model, f"{mechanism}_{dims}d", encoder)
        for model in ("gru", "lstm", "transformer")
        for mechanism in ("asa", "fa")
        for dims in (2, 3, 4)
        for encoder in ("onehot", "embeddings")
    }
    assert len(required) == 36
    assert required <= set(COMBO_ID)
    optional_aca = {
        (model, f"aca_{dims}d", encoder)
        for model in ("gru", "lstm", "transformer")
        for dims in (2, 3, 4)
        for encoder in ("onehot", "embeddings")
    }
    optional_hybrid_id = {
        (model, f"{mechanism}_{dims}d", encoder)
        for model in ("mamba2", "mamba3")
        for mechanism in ("asa", "aca", "fa")
        for dims in (2, 3, 4)
        for encoder in ("onehot", "embeddings")
    }
    assert set(COMBO_ID) == required | optional_aca | optional_hybrid_id


def test_cost_generation_fails_if_any_required_cell_is_unmeasured():
    assert cost_exit_code([("mamba3", "embeddings", "oom")]) == 1
    assert cost_exit_code([], allow_failures=False) == 0
    assert cost_exit_code([("mamba3", "embeddings", "oom")],
                          allow_failures=True) == 0


def test_xgboost_uses_the_same_full_history_window_as_neural_models():
    features = np.arange(24, dtype=np.float32).reshape(8, 3)
    targets = np.arange(16, dtype=np.float32).reshape(8, 2)
    x, y = window_xy(features, targets, 3, 6, input_len=3)
    np.testing.assert_array_equal(x[0], features[0:3].reshape(-1))
    np.testing.assert_array_equal(x[-1], features[2:5].reshape(-1))
    np.testing.assert_array_equal(y, targets[3:6])


def test_absolute_significance_loss_is_mae_not_root_mse():
    predictions = np.array([[[3.0, 4.0]]])
    squared, absolute = observation_losses(predictions, np.zeros_like(predictions))
    assert squared.item() == 12.5
    assert absolute.item() == 3.5
    assert not np.isclose(absolute.item(), np.sqrt(squared.item()))


def test_the_tabular_arm_module_imports_without_xgboost():
    """A hard top-level `import xgboost` in scripts/xgb_agg.py made THIS whole
    module fail to collect on any box without xgboost installed.

    The cost was invisible: pytest reported one collection ERROR, and roughly
    thirty unrelated regression tests -- the declared-matrix size, the retrain
    slices, the step-matching contract -- simply did not run. Found on a fresh
    GPU box where xgboost was not yet installed.

    src/models/__init__.py already treats xgboost as optional; the script has
    to as well, because its pure helpers (window_xy) are imported for tests
    that have nothing to do with gradient boosting.
    """
    source = (Path(__file__).resolve().parent.parent / "scripts/xgb_agg.py").read_text()
    body = source.split('"""', 2)[-1]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("import xgboost") or stripped.startswith("from xgboost"):
            assert line.startswith(" "), (
                "xgboost must be imported inside the function that needs it, "
                "not at module scope"
            )


def test_the_cuda_requirements_declare_what_the_roster_needs():
    """`pip install -r requirements-cuda.txt` must yield an environment where
    every roster model registers.

    It did not: pytorch_lightning (external/s4 callbacks -> s4, s4nd) and
    huggingface_hub (external/mamba -> mamba2, mamba3, mamba_nd) were both
    absent, so 5 of the 8 models silently failed to register. src/models
    reports that as a UserWarning, not an error, so the gap only surfaced once
    a sweep was already running and every SSM cell failed.
    """
    reqs = (Path(__file__).resolve().parent.parent / "requirements-cuda.txt").read_text()
    for package in ("pytorch_lightning", "huggingface_hub"):
        assert package in reqs, f"{package} is needed by the vendored SSMs"


def test_the_builder_and_the_docs_agree_on_the_source_filenames():
    """README said PORTHS6IM, the builder opens PORTHS6MM.

    An operator following the README downloads the wrong archives and the
    build fails with "missing N expected Census monthly archive(s)" -- after
    the download. The code is authoritative here because it is what opens the
    files, so the docs must match it.
    """
    from scripts.build_census_lattice import FLOWS

    repo = Path(__file__).resolve().parent.parent
    prefixes = {prefix for _, prefix in FLOWS.values()}
    assert prefixes == {"PORTHS6MM", "PORTHS6XM"}
    for doc in ("README.md", "data/README.md", "PLAN.md"):
        text = (repo / doc).read_text()
        if "PORTHS6" not in text:
            continue
        assert "PORTHS6IM" not in text, f"{doc} names an archive the builder never opens"
        for prefix in prefixes:
            pass        # presence is not required in every doc, absence of the
                        # wrong spelling is


def test_the_api_fetcher_is_not_mistaken_for_the_lattice_source():
    """scripts/fetch_census_ports.py hits the API and writes CSVs into
    census_ports/ (with an s); the builder reads fixed-width ZIPs from
    census_port/. Nothing consumes the fetcher's output, and the docs must say
    so rather than leaving the name to imply otherwise."""
    repo = Path(__file__).resolve().parent.parent
    text = (repo / "data/README.md").read_text()
    assert "fetch_census_ports.py" in text
    assert "not" in text.lower().split("fetch_census_ports.py")[1][:120].lower()


def test_the_bulk_fetcher_targets_exactly_what_the_builder_reads():
    """The fetcher and the builder must not drift on folder or prefix.

    They are the two halves of one contract -- one writes the layout the other
    opens -- and the repo has already shipped one filename disagreement
    (PORTHS6IM in the docs vs PORTHS6MM in the code).
    """
    from scripts.build_census_lattice import FLOWS
    import scripts.fetch_census_bulk as fetcher

    source = (Path(__file__).resolve().parent.parent
              / "scripts/fetch_census_bulk.py").read_text()
    body = source.split('"""', 2)[-1]
    for folder, prefix in FLOWS.values():
        assert f'"{folder}"' not in body, (
            f"{folder} is hard-coded in the fetcher instead of coming from FLOWS")
        assert f'"{prefix}"' not in body, (
            f"{prefix} is hard-coded in the fetcher instead of coming from FLOWS")
    assert "from scripts.build_census_lattice import FLOWS" in source

    plan = fetcher.BULK.format(year=2010, folder="im_hs6_m",
                               prefix="PORTHS6MM", yy="10", mm="01")
    assert plan.endswith("/2010/Port/im_hs6_m/PORTHS6MM1001.ZIP")


def test_the_fetcher_writes_schedule_d_under_the_name_the_builder_opens():
    import inspect

    from scripts.build_census_lattice import load_state_map
    import scripts.fetch_census_bulk as fetcher

    assert "scheduleD_dist3.txt" in inspect.getsource(load_state_map)
    assert fetcher.SCHEDULE_D_NAME == "scheduleD_dist3.txt"


def test_the_fetcher_sends_a_browser_user_agent():
    """census.gov answers urllib's default User-Agent with 403 while serving
    the same URL to curl. Found by running the fetcher, not by a HEAD check
    made with a different client."""
    import scripts.fetch_census_bulk as fetcher
    assert "Mozilla" in fetcher._UA


def test_the_month_range_covers_the_declared_panel():
    import scripts.fetch_census_bulk as fetcher
    all_months = list(fetcher.months("2010-01", "2025-12"))
    assert len(all_months) == 192, "the panel is 192 monthly points"
    assert all_months[0] == (2010, 1) and all_months[-1] == (2025, 12)


def test_the_builder_default_name_is_what_the_pipeline_opens():
    """The builder writes {--name}.npz; config/census.yaml and
    sweep.DEFAULT_NPZ both read census_lattice_9ch.npz. A default that
    disagrees only surfaces after the multi-GB download and the build have
    already been paid for."""
    import subprocess

    repo = Path(__file__).resolve().parent.parent
    from scripts.sweep import DEFAULT_NPZ

    expected = Path(str(DEFAULT_NPZ)).stem
    source = (repo / "scripts/build_census_lattice.py").read_text()
    assert f'"--name", default="{expected}"' in source, (
        f"the builder's --name default does not produce {expected}.npz")

    # and the operator can see it without reading the source
    help_text = subprocess.run(
        [sys.executable, str(repo / "scripts/build_census_lattice.py"), "--help"],
        capture_output=True, text=True, cwd=str(repo)).stdout
    assert expected in help_text, "--help hides the default it will write"

    census_yaml = (repo / "config/census.yaml").read_text()
    assert f"{expected}.npz" in census_yaml
