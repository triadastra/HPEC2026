"""Regressions for the repository-wide review, with no remote side effects."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import select_lr
from scripts.significance_tests import wilcoxon_series
from src.data.dataloader import DataConfig, TradeDataPipeline
from src.models import create_model
from src.models.tabular_common import extract_tabular_xy
from src.training.trainer import Trainer
from tools import coordinate
from tools.publication import drop_invalid_runs, publication_runs, withdrawn_remote_files

REPO = Path(__file__).resolve().parents[1]


def plant(root, name, argv=None):
    d = root / name
    d.mkdir(parents=True)
    (d / 'best.pth').write_bytes(b'weights')
    entry = dict(name=name, fingerprint='fp', argv=argv or [])
    (d / 'run_complete.json').write_text(json.dumps(dict(
        fingerprint='fp', checkpoint_sha256=hashlib.sha256(b'weights').hexdigest())))
    return entry


def manifest(root, entries):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'manifest.json').write_text(json.dumps({'runs': entries}))


def test_deleted_source_cannot_survive_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinate, 'LOCAL', tmp_path)
    entry = plant(tmp_path / 'boxA', 'gru_embeddings_1d_s947')
    manifest(tmp_path / 'boxA', [entry])
    manifest(tmp_path / 'boxB', [])
    coordinate.merge(SimpleNamespace())
    shutil.rmtree(tmp_path / 'boxA' / entry['name'])
    coordinate.merge(SimpleNamespace())
    assert not (tmp_path / 'sweep' / entry['name']).exists()


def test_publication_requires_session_membership(tmp_path):
    valid = plant(tmp_path, 'valid')
    plant(tmp_path, 'undeclared')
    invalid = plant(tmp_path, 'invalid')
    (tmp_path / 'invalid' / 'best.pth').write_bytes(b'corrupted')
    nested = plant(tmp_path / 'exp0', 'valid')
    manifest(tmp_path, [valid, invalid])
    manifest(tmp_path / 'exp0', [nested])
    kept, dropped = drop_invalid_runs(tmp_path)
    assert kept == {'valid', 'exp0/valid'}
    assert {d.name for d, _ in dropped} == {'undeclared', 'invalid'}
    assert not (tmp_path / 'undeclared').exists()
    assert not (tmp_path / 'invalid').exists()


def test_missing_publication_manifest_fails_closed(tmp_path):
    plant(tmp_path, 'old')
    with pytest.raises(FileNotFoundError):
        publication_runs(tmp_path)
    assert (tmp_path / 'old/best.pth').exists()


def test_remote_withdrawal_includes_old_completion_records():
    remote = ['README.md', 'valid/best.pth', 'valid/run_complete.json',
              'old/best.pth', 'old/run_complete.json', 'old/logs/train.log',
              'exp0/old/best.pth', 'exp0/old/run_complete.json',
              'orphan/best.pth']
    assert set(withdrawn_remote_files(remote, {'valid', 'exp0/old'})) == {
        'old/best.pth', 'old/run_complete.json', 'old/logs/train.log', 'orphan/best.pth'}


@pytest.mark.parametrize('failure', ['pull', 'merge', 'stage', 'validate', 'prune', 'upload'])
def test_publication_aborts_failed_stage(tmp_path, failure):
    tools = tmp_path / 'tools'; tools.mkdir()
    shutil.copy(REPO / 'tools/publish_checkpoints.sh', tools)
    (tmp_path / 'outputs/merged/sweep').mkdir(parents=True)
    bins = tmp_path / 'bin'; bins.mkdir()
    # All commands that could reach a network are replaced by local stubs.
    python = bins / 'python3'
    python.write_text('''#!/bin/bash
case "$*" in
  *coordinate.py*pull*) stage=pull ;;
  *coordinate.py*merge*) stage=merge ;;
  *Celsia/HPEC2026*) stage=prune ;;
  *)
    source=$(cat)
    if [[ "$source" == *copytree* ]]; then stage=stage; else stage=validate; fi ;;
esac
echo "$stage" >> "$TRACE"
[ "$FAILURE" != "$stage" ] || exit 7
exit 0
''')
    hf = bins / 'hf'
    hf.write_text('''#!/bin/bash
echo upload >> "$TRACE"
[ "$FAILURE" != upload ] || exit 7
exit 0
''')
    python.chmod(0o755); hf.chmod(0o755)
    trace = tmp_path / 'trace'
    env = {**os.environ, 'PATH': str(bins) + ':' + os.environ['PATH'],
           'VMPW_A': 'test', 'VMPW_B': 'test', 'FAILURE': failure, 'TRACE': str(trace)}
    result = subprocess.run(['bash', str(tools / 'publish_checkpoints.sh')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 7, result.stdout + result.stderr
    assert trace.read_text().splitlines()[-1] == failure
    assert '[pub] https://' not in result.stdout


def test_divergent_seed_disqualifies_rate(tmp_path):
    entries = []
    for lr, tag, loss in [('0.001', 'lr1e3', 0.2), ('0.01', 'lr1e2', 0.01)]:
        for seed in (947, 732):
            name = f'exp0_flat_gru_{tag}_s{seed}'
            entries.append(plant(tmp_path, name, ['python', '--lr', lr]))
            d = tmp_path / name
            (d / 'logs').mkdir()
            (d / 'logs/metrics.json').write_text(json.dumps(
                [{'epoch': i, 'val_loss': loss} for i in range(4)]))
            if lr == '0.01' and seed == 732:
                (d / 'run_diverged.json').write_text('{}')
    manifest(tmp_path, entries)
    assert select_lr.main(['--runs', str(tmp_path)]) == 0
    result = json.loads((tmp_path / 'lr_selection.json').read_text())
    assert result['selected']['flat']['gru'] == '0.001'
    assert result['diagnostics']['flat/gru']['best_val']['0.01'] is None
    # No survivor-only selection even when ALL candidates have failed seeds.
    (tmp_path / 'exp0_flat_gru_lr1e3_s732/run_diverged.json').write_text('{}')
    assert select_lr.main(['--runs', str(tmp_path)]) == 1


def trade_pipeline():
    cfg = DataConfig(start='2010-01-01', end='2011-12-01', input_len=3,
                     lag_count=2, train_end='2011-06-01', val_start='2011-07-01',
                     val_end='2011-09-01', test_start='2011-10-01', test_end='2011-12-01')
    p = TradeDataPipeline(cfg)
    p.df_full = pd.DataFrame({'State': 'S', 'Commodity': 'C', 'Import/Export': 'Import',
        'Country': 'X', 'Time': pd.date_range(cfg.start, cfg.end, freq='MS'),
        'Value': range(1, 25), 'Weight': range(2, 26)})
    p.create_splits()
    return p


def test_tabular_targets_match_neural_dataset():
    p = trade_pipeline()
    for loader in p.get_dataloaders(batch_size=2, num_workers=0, shuffle_train=False):
        x, y, _ = extract_tabular_xy(loader, variant='embeddings',
            num_numeric_features=len(p.feat_cols), num_states=1, num_commodities=1, num_flows=1)
        expected = np.array([[sample['target_value'].item(), sample['target_weight'].item()]
                             for sample in loader.dataset])
        np.testing.assert_allclose(y, expected)
        assert len(x) == len(expected)


def test_csv_train_constructs_nonempty_vocabularies(tmp_path, monkeypatch):
    from scripts import train
    p = trade_pipeline()
    monkeypatch.setattr(train, 'TradeDataPipeline', lambda config: p)
    monkeypatch.setattr(p, 'load_data', lambda: p.df_full)
    monkeypatch.setattr(p, 'save_artifacts', lambda path: None)
    monkeypatch.setenv('CENSUS_NUM_WORKERS', '0')
    monkeypatch.setattr(sys, 'argv', ['train.py', '--model', 'gru', '--variant',
        'embeddings', '--data-config', '', '--batch-size', '2', '--out-dir', str(tmp_path)])
    class Checked(Exception):
        pass
    def check_fit(self, model, tr, va):
        assert (model.num_states, model.num_commodities, model.num_flows) == (1, 1, 1)
        b = next(iter(tr))
        out = model(b['x_numeric'], b['state_ids'], b['comm_ids'], b['flow_ids'])
        assert out.shape == (2, 2)
        raise Checked()
    monkeypatch.setattr(train.Trainer, 'fit', check_fit)
    with pytest.raises(Checked):
        train.main()


@pytest.mark.parametrize('scipy_available', [True, False])
def test_wilcoxon_tie_correction(monkeypatch, scipy_available):
    scipy = pytest.importorskip('scipy.stats')
    from scripts import significance_tests as sig
    monkeypatch.setattr(sig, 'HAVE_SCIPY', scipy_available)
    diff = np.array([1.] * 15 + [-1.] * 5 + [0.] * 3)
    _, p = wilcoxon_series(diff, np.zeros(len(diff)))
    assert p == pytest.approx(scipy.wilcoxon(diff, method='approx').pvalue)


def test_s4_training_cost_is_partial():
    from src.models import ModelFactory
    if 's4' not in ModelFactory.list_models():
        pytest.skip('S4 unavailable')
    model = create_model('s4', 'embeddings', num_numeric_features=2, num_states=1,
        num_commodities=1, num_flows=1, d_model=4, n_layers=1, d_state=4, dropout=0)
    from src.training.losses import MSELoss
    trainer = Trainer({'training': {'device': 'cpu'}})
    trainer.model = model; trainer.combo = False; trainer.criterion = MSELoss()
    trainer.optimizer = torch.optim.Adam(model.parameters())
    trainer.train_loader = [dict(x_numeric=torch.randn(2, 4, 2),
        state_ids=torch.zeros(2, dtype=torch.long), comm_ids=torch.zeros(2, dtype=torch.long),
        flow_ids=torch.zeros(2, dtype=torch.long), target_value=torch.zeros(2),
        target_weight=torch.zeros(2))]
    cost = trainer._measure_step_flops()
    assert cost['status'] == 'partial'
    assert any('FFTConv' in name for name in cost['flops_uncounted_modules'])


@pytest.mark.parametrize('save_last', [True, False])
def test_last_checkpoint_tracks_nonimproving_final_epoch(tmp_path, monkeypatch, save_last):
    model = create_model('gru', 'embeddings', num_numeric_features=2, num_states=1,
        num_commodities=1, hidden_size=4, num_layers=1)
    trainer = Trainer({'training': {'device': 'cpu', 'epochs': 3, 'scheduler': None},
        'checkpointing': {'save_dir': str(tmp_path), 'save_best_only': False, 'save_last': save_last},
        'logging': {'log_dir': str(tmp_path / 'logs')}})
    losses = iter([1., 2., 3.])
    monkeypatch.setattr(trainer, 'train_epoch', lambda: 0.5)
    monkeypatch.setattr(trainer, 'validate', lambda: next(losses))
    monkeypatch.setattr(trainer, '_record_cost', lambda: None)
    trainer.fit(model, [None], [None])
    assert (tmp_path / 'best.pth').exists()
    assert (tmp_path / 'last.pth').exists() == save_last
    if save_last:
        last = torch.load(tmp_path / 'last.pth', weights_only=True)
        assert last['epoch'] == 2
        assert last['score'] == 3.
        assert last['optimizer_state_dict'] is not None


def test_cuda_rnn_training_cost_is_flagged_without_running_cuda():
    trainer = Trainer({'training': {'device': 'cpu'}})
    trainer.model = torch.nn.GRU(2, 4)
    trainer.device = torch.device('cuda')
    assert any('GRU' in name for name in trainer._dispatcher_blind_modules())
