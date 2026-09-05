import json

import numpy as np
import pytest
import torch

from scripts.dump_errors import census_config_from_entry
from scripts.model_cost import forecast_normalized_flops
from src.data.census_loader import CensusConfig, CensusLattice
from src.models.tabular_common import fallback_feature_names
from src.utils.run_manifest import (
    _repository_record,
    run_fingerprint,
    run_input_fingerprint,
)


def _write_lattice(tmp_path, *, fit_end):
    raw = np.arange(24, dtype=np.float32).reshape(2, 6, 2)
    log_panel = np.log1p(raw)
    norm_min = log_panel[:, :fit_end].min(axis=1)
    norm_range = np.maximum(
        log_panel[:, :fit_end].max(axis=1) - norm_min, 1.0
    )
    panel_norm = (log_panel - norm_min[:, None]) / norm_range[:, None]
    npz = tmp_path / "lattice.npz"
    np.savez(
        npz,
        panel_norm=panel_norm,
        panel_raw=raw,
        series_idx=np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int32),
        mask=np.ones((1, 1, 2), dtype=bool),
        norm_min=norm_min,
        norm_max=norm_min + norm_range,
        norm_range=norm_range,
    )
    npz.with_suffix(".json").write_text(json.dumps({
        "channels": ["agg_value", "agg_weight"],
        "target_channels": [0, 1],
        "cohort_selection": {
            "n_months": 4,
            "state_and_commodity_ranking": "training_months_only",
        },
        "splits": {"train": [0, 4], "val": [4, 5], "test": [5, 6]},
    }))
    return npz


def test_official_loader_rejects_full_timeline_normalization(tmp_path):
    npz = _write_lattice(tmp_path, fit_end=6)
    with pytest.raises(ValueError, match="training-only log1p fit"):
        CensusLattice(CensusConfig(
            npz=str(npz), input_len=2, lag_count=1, train_end=4, val_end=5,
        ))


def test_run_fingerprint_changes_when_source_changes(tmp_path):
    source = tmp_path / "src" / "model.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    first = run_fingerprint([], tmp_path)
    first_inputs = run_input_fingerprint([], tmp_path)
    assert first != first_inputs
    source.write_text("VALUE = 2\n")
    _repository_record.cache_clear()
    second = run_fingerprint([], tmp_path)
    second_inputs = run_input_fingerprint([], tmp_path)
    assert first != second
    assert first_inputs != second_inputs


def test_manifest_entry_controls_significance_data_contract(tmp_path):
    entry = {
        "name": "gru_embeddings_1d_s947",
        "argv": [
            "python", "scripts/train.py", "--model", "gru", "--variant",
            "embeddings", "--config", "config/base.yaml", "--data-config",
            "config/census.yaml", "--npz", str(tmp_path / "custom.npz"),
            "--input-len", "24", "--lag-count", "6",
        ],
    }
    config = census_config_from_entry(entry)
    assert config.npz == str((tmp_path / "custom.npz").resolve())
    assert config.input_len == 24
    assert config.lag_count == 6


def test_manifest_reconstructs_rolling_aggregate_contract(tmp_path):
    entry = {
        "name": "gru_aggregate_roll_y2020_s947",
        "argv": [
            "python", "scripts/train.py", "--model", "gru", "--variant",
            "embeddings", "--config", "config/base.yaml", "--data-config",
            "config/census.yaml", "--npz", str(tmp_path / "custom.npz"),
            "--aggregate", "--refit-normalization", "--train-end", "108",
            "--val-end", "120", "--test-end", "132",
        ],
    }
    config = census_config_from_entry(entry)
    assert config.aggregate and config.refit_normalization
    assert (config.train_end, config.val_end, config.test_end) == (108, 120, 132)


def test_cost_panel_reports_flops_per_forecast():
    flat = {"target_value": torch.zeros(8)}
    combo = {"target_value": torch.zeros(2, 4)}
    assert forecast_normalized_flops(800, flat) == (8, 100)
    assert forecast_normalized_flops(800, combo) == (8, 100)


@pytest.mark.parametrize("variant", ("embeddings", "onehot"))
def test_census_tabular_feature_names_match_matrix_width(variant):
    names = fallback_feature_names(variant, 35, 14, 1263, 2)
    categorical_width = 3 if variant == "embeddings" else 14 + 1263 + 2
    assert len(names) == 35 + categorical_width



def _cost_trainer(val_loss, steps, observations, *, mode="min", measured=None):
    """Bare Trainer with only the fields _record_cost() reads."""
    from src.training.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.history = {"train_loss": [0.0] * len(val_loss), "val_loss": list(val_loss),
                 "lr": [1e-3] * len(val_loss), "steps": list(steps),
                 "observations": list(observations)}
    t.mode = mode
    t.total_steps = steps[-1]
    t.total_observations = observations[-1]
    t.effective_batch_size = 28292
    t.batch_size, t.grad_accum = 2048, 14
    t.logger = type("L", (), {"log_info": staticmethod(lambda *_a, **_k: None)})()
    t._measure_step_flops = lambda: dict(
        measured or {"status": "measured", "flops_uncounted_modules": [],
                     "flops_per_observation": 1000.0}
    )
    return t


def test_cost_charges_flops_to_the_best_checkpoint_not_the_last_epoch():
    # Best val at epoch 2; training then burns the patience epochs out to 5.
    # The Pareto axis must bill epoch 2's work, not epoch 5's.
    t = _cost_trainer([0.5, 0.2, 0.3, 0.4, 0.45], [96, 192, 288, 384, 480],
                      [1000, 2000, 3000, 4000, 5000])
    t._record_cost()
    assert t.cost["epochs_to_best"] == 2
    assert t.cost["observations_to_best"] == 2000
    assert t.cost["train_flops_to_best"] == 2000 * 1000.0
    assert t.cost["train_flops_total"] == 5000 * 1000.0
    assert t.cost["optimizer_steps_to_best"] == 192   # steps still reported
    assert t.history["cost"] is t.cost


def test_cost_scales_by_observations_not_steps_so_short_groups_are_not_overcharged():
    # 3 micro-batches at grad_accum=2 => train_epoch closes groups of 2 and 1.
    # Billing 2 x a full group would overcharge; observations are exact.
    t = _cost_trainer([0.5], [2], [300])
    t.grad_accum = 2
    t._measure_step_flops = lambda: {
        "status": "measured", "flops_uncounted_modules": [],
        "flops_per_observation": 1000.0, "observations_measured": 200,
        "flops_per_optimizer_step": 200_000.0,
    }
    t._record_cost()
    assert t.cost["train_flops_total"] == 300_000.0            # 300 observations
    per_step = t.cost["flops_per_optimizer_step"] * t.total_steps
    assert per_step == 400_000.0                               # would overcharge 33%


def test_cost_honours_max_mode_monitors():
    t = _cost_trainer([0.1, 0.9, 0.4], [96, 192, 288], [1000, 2000, 3000],
                      mode="max")
    t._record_cost()
    assert t.cost["epochs_to_best"] == 2
    assert t.cost["train_flops_to_best"] == 2000 * 1000.0


def test_cost_degrades_without_crashing_when_measurement_fails():
    t = _cost_trainer([0.5, 0.2], [96, 192], [1000, 2000],
                      measured={"status": "failed", "reason": "CUDA OOM"})
    t._record_cost()
    assert t.cost["status"] == "failed"
    assert t.cost["train_flops_total"] is None
    assert t.cost["train_flops_to_best"] is None
    assert t.cost["observations_to_best"] == 2000   # counts still reported


def _tiny_trainer(dropout=0.0):
    """Trainer over a real nn.Module, enough to exercise _measure_step_flops."""
    from src.training.trainer import Trainer

    d_in, d_out = 8, 2
    model = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                torch.nn.Linear(d_in, d_out, bias=False))
    t = Trainer.__new__(Trainer)
    t.model, t.device, t.combo, t.use_amp = model, torch.device("cpu"), False, False
    t.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    t.seen_training_mode = []
    t.train_loader = [{"x_numeric": torch.randn(4, d_in),
                       "target_value": torch.zeros(4),
                       "target_weight": torch.zeros(4)} for _ in range(3)]

    def fwd(batch):
        t.seen_training_mode.append(model.training)
        return (model(batch["x_numeric"]) ** 2).mean()

    t._forward_and_loss = fwd
    return t


def test_measurement_runs_in_training_mode_and_restores_eval():
    # validate() runs last and leaves the model in eval mode. Counting a
    # backward there breaks cuDNN RNN backward outright (every GRU/LSTM run
    # would report "failed") and drops dropout from the counted graph.
    t = _tiny_trainer(dropout=0.5)
    t.grad_accum = 2
    t.model.eval()
    out = t._measure_step_flops()
    assert out["status"] == "measured"
    assert t.seen_training_mode == [True, True]      # measured in train mode
    assert t.model.training is False                 # prior mode restored


def test_measurement_accepts_a_loader_shorter_than_grad_accum():
    t = _tiny_trainer()
    t.grad_accum = 10                                 # loader only has 3
    out = t._measure_step_flops()
    assert out["status"] == "measured"
    assert out["micro_batches_measured"] == 3
    assert out["observations_measured"] == 12
    assert out["flops_per_observation"] > 0


def test_triton_backed_models_are_reported_partial_not_measured():
    # FlopCounterMode sees only registered ATen ops; the Mamba SSD scan runs as
    # raw Triton launches, so its dominant term is absent from the total.
    t = _tiny_trainer()
    t.grad_accum = 1
    t._dispatcher_blind_modules = lambda: ["mamba_ssm.modules.mamba2.Mamba2"]
    out = t._measure_step_flops()
    assert out["status"] == "partial"
    assert out["flops_uncounted_modules"] == ["mamba_ssm.modules.mamba2.Mamba2"]


def test_dispatcher_blind_detection_keys_on_module_origin_not_a_name_list():
    from src.training.trainer import Trainer

    class Mamba2(torch.nn.Identity):
        pass
    Mamba2.__module__ = "mamba_ssm.modules.mamba2"     # as the vendored kernel presents

    t = Trainer.__new__(Trainer)
    t.model = torch.nn.Sequential(torch.nn.Linear(2, 2), Mamba2())
    t.device = torch.device("cpu")
    assert t._dispatcher_blind_modules() == ["mamba_ssm.modules.mamba2.Mamba2"]

    t.model = torch.nn.Sequential(torch.nn.Linear(2, 2))   # pure-torch fallback
    assert t._dispatcher_blind_modules() == []             # stops flagging on its own
