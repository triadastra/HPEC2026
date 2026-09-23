#!/usr/bin/env python3
"""Build the complete result tables and figures from committed archival inputs.

Run from any directory: python scripts/results_gallery.py [--dpi 300] [--tables-only]
No training, inference, GPU profiling, or remote downloads occur.
"""
import argparse
import csv
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gallery'
INPUT = OUT / 'inputs'
TABLES = OUT / 'tables'
FIGS = OUT / 'figures'
sys.path.insert(0, str(ROOT))
from scripts.significance_tests import wilcoxon_series

SEEDS = [947, 732, 619]
METRICS = ['MSE', 'MSE_std', 'MAE', 'RMSE', 'sMAPE', 'MASE', 'val_loss']
NAMES = {'gru': 'GRU', 'lstm': 'LSTM', 'transformer': 'Transformer', 's4': 'S4',
         's4nd': 'S4ND', 'mamba2': 'Mamba-2', 'mamba3': 'Mamba-3', 'mamba_nd': 'Mamba-ND',
         'random_walk': 'Persistence', 'moving_average': 'Moving average (12)',
         'seasonal_naive': 'Seasonal naive', 'xgboost': 'XGBoost'}
PAPER_ARMS = ['persistence', 's4_embeddings_1d', 'mamba2_embeddings_1d',
              's4nd_grid_3d_embeddings', 'mamba_nd_grid_2d_embeddings',
              'gru_embeddings_1d', 'lstm_embeddings_1d', 'transformer_embeddings_1d',
              'gru_asa_2d_embeddings', 'gru_fa_local_2d_embeddings', 'gru_fa_3d_embeddings',
              'lstm_asa_4d_embeddings', 'lstm_fa_local_4d_embeddings', 'lstm_fa_4d_embeddings',
              'transformer_asa_3d_embeddings', 'transformer_fa_local_2d_embeddings',
              'transformer_fa_3d_embeddings']
COLORS = {'gru': '#2874A6', 'lstm': '#C67824', 'transformer': '#27845D', 's4': '#BF4B45',
          's4nd': '#BF4B45', 'mamba2': '#8056A3', 'mamba_nd': '#8056A3',
          'mamba3': '#208D93', 'xgboost': '#383F4A'}


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    return None if value in ('', None) else float(value)


def write_csv(name, rows):
    if not rows:
        raise ValueError(f'Empty output table {name}')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (TABLES / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def holm(p):
    """Adjust a declared family; reject missing p-values, preserve zero tails."""
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Holm requires finite probabilities in [0,1]')
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return adjusted


def dm_from_months(values, lag=0):
    """HLN h=1 SE with Bartlett HAC; float32 lag-0 replays archived arithmetic."""
    d = np.asarray(values) if lag == 0 else np.asarray(values, dtype=float)
    n = len(d)
    if n < 2 or not 0 <= lag < n:
        raise ValueError('Invalid monthly sample or HAC lag')
    mean = float(d.mean())
    c = d - mean
    variance = float((c * c).mean())
    if lag:
        variance += 2 * sum((1 - k / (lag + 1)) * float(np.dot(c[k:], c[:-k])) / n for k in range(1, lag + 1))
    if variance <= 0:
        raise ValueError('Degenerate differential; do not claim significance')
    se = math.sqrt(variance / n) / math.sqrt((n - 1) / n)
    statistic = mean / se
    return mean, se, statistic, float(2 * stats.t.sf(abs(statistic), n - 1))


def tost(mean, se, delta, n=24):
    return float(max(stats.t.sf((mean + delta) / se, n - 1),
                     stats.t.sf(-(mean - delta) / se, n - 1)))


def arm_label(arm):
    if arm == 'persistence':
        return 'Persistence'
    model = next(m for m in sorted(NAMES, key=len, reverse=True) if arm.startswith(m + '_'))
    rest = arm[len(model) + 1:].replace('_embeddings', '')
    rest = rest.replace('embeddings_1d', 'flat').replace('fa_local', 'CoFA').replace('fa_sm', 'FA-softmax')
    rest = rest.replace('asa', 'ASA').replace('fa', 'FA').replace('grid_', '').replace('_id', ' + identity')
    return NAMES[model] + ' ' + rest.replace('_', ' ').replace('2d', '2D').replace('3d', '3D').replace('4d', '4D')


def row_label(r):
    model, variant, enc = r['model'], r['variant'], r['enc']
    if model in ('random_walk', 'moving_average', 'seasonal_naive', 'xgboost') or r['panel'] != 'series':
        return NAMES.get(model, model)
    if variant in ('embeddings', 'onehot'):
        return NAMES.get(model, model) + ' flat / ' + ('one-hot' if variant == 'onehot' else 'embeddings')
    return arm_label(model + '_' + variant + '_embeddings') + (' / one-hot' if enc == 'onehot' else '')


def full_results():
    sources = ['results/flat_and_nd_results.csv', 'results/aggregate_and_identity_results.csv',
               'results/rolling_annual.csv', 'results/rolling_pooled.csv',
               'artifacts/exp7/results.csv', 'artifacts/exp8/results.csv']
    merged = {}
    for source in sources:
        for original in read_csv(ROOT / source):
            key = tuple(original[k] for k in ('panel', 'model', 'variant', 'enc'))
            row = {k: original[k] for k in ('panel', 'model', 'variant', 'enc')}
            row['n'] = int(original['n'])
            row.update({m: number(original.get(m)) for m in METRICS})
            row['source'] = source
            if key in merged:
                for field in ['n'] + METRICS:
                    if merged[key][field] != row[field]:
                        raise ValueError(f'Conflicting results for {key}: {field}')
                merged[key]['source'] += '; ' + source
            else:
                merged[key] = row
    archived = json.loads((INPUT / 'figure_inputs_archived.json').read_text())
    # Fixed aggregate baseline rows are absent from the root result CSVs.
    for original in archived['table1']:
        reverse = {'Moving avg.': 'moving_average', 'T-1': 'random_walk', 'Seasonal naive': 'seasonal_naive'}
        if original['model'] not in reverse:
            continue
        model = reverse[original['model']]
        r = dict(panel='aggregate', model=model, variant='baseline', enc='aggregate', n=1,
                 MSE=original['mse'], MSE_std=0.0, MAE=original['mae'], RMSE=original['rmse'],
                 sMAPE=original['smape'], MASE=original['mase'], val_loss=None,
                 source='results/gallery/inputs/figure_inputs_archived.json')
        merged[('aggregate', model, 'baseline', 'aggregate')] = r
    xgb = json.loads((INPUT / 'xgboost_all.json').read_text())
    groups = {}
    for r in xgb['rows']:
        groups.setdefault((r['panel'], r['variant']), []).append(r)
    # Equal-length rolling test folds: pool metrics within seed, then across seeds.
    pooled = []
    for seed in SEEDS:
        rs = [r for r in xgb['rows'] if r['panel'] == 'aggregate_rolling' and r['seed'] == seed]
        weights = np.array([r['n_test_months'] for r in rs])
        r = {m: float(np.average([row[m] for row in rs], weights=weights))
             for m in ['both_MSE', 'both_MAE', 'both_sMAPE', 'both_MASE']}
        # Match evaluate._pooled_rolling_rows: sqrt(pooled MSE) within seed.
        r['both_RMSE'] = math.sqrt(r['both_MSE'])
        pooled.append(r)
    groups[('aggregate_rolling', 'aggregate_roll_pooled')] = pooled
    for (panel, variant), rs in groups.items():
        r = dict(panel=panel, model='xgboost', variant=variant, enc='aggregate', n=len(rs),
                 **{m: float(np.mean([rr['both_' + m] for rr in rs])) for m in METRICS if m not in ('MSE_std', 'val_loss')},
                 MSE_std=float(np.std([rr['both_MSE'] for rr in rs])), val_loss=None,
                 source='results/gallery/inputs/xgboost_all.json')
        merged[(panel, 'xgboost', variant, 'aggregate')] = r
    costs = {(r['model'], r['variant'], r['enc']): r for r in read_csv(ROOT / 'results/model_cost_FINAL.csv')}
    for row in merged.values():
        key = (row['model'], row['variant'], row['enc'])
        c = costs.get(key) if row['panel'] != 'aggregate_rolling' else None
        row.update(params=None, flops_per_forecast=None, flops_upper_bound=None,
                   analytic_flops=None, tree_comparisons_upper_bound=None, peak_mem_GB=None,
                   cost_status='not recorded', cost_source=None)
        if c:
            row.update(params=int(c['params']), flops_per_forecast=float(c['flops_per_forecast']),
                       peak_mem_GB=float(c['peak_mem_GB']), cost_status='archived profiler + analytic top-ups',
                       cost_source='results/model_cost_FINAL.csv')
        elif row['model'] == 'xgboost':
            row.update(flops_upper_bound=2 * xgb['hparams']['n_estimators'],
                       tree_comparisons_upper_bound=2 * xgb['hparams']['n_estimators'] * xgb['hparams']['max_depth'],
                       cost_status='analytical arithmetic upper bound; not measured',
                       cost_source='results/gallery/inputs/xgboost_all.json; documented counting convention')
        elif row['model'] in ('random_walk', 'seasonal_naive', 'moving_average'):
            row.update(analytic_flops=24 if row['model'] == 'moving_average' else 0,
                       cost_status='analytical arithmetic count; not runtime',
                       cost_source='copy only, or 11 additions + 1 division per each of two outputs')
        row['label'] = row_label(row)
    result = sorted(merged.values(), key=lambda r: (r['panel'], r['variant'] if r['panel'] == 'aggregate_rolling' else '', r['MSE'], r['model'], r['enc']))
    write_csv('all_results.csv', result)
    write_csv('all_costs.csv', read_csv(ROOT / 'results/model_cost_FINAL.csv'))
    write_csv('xgboost_per_seed.csv', xgb['rows'])
    return result


def comparisons():
    data = np.load(INPUT / 'loss_summaries.npz', allow_pickle=False)
    names = data['names'].tolist()
    provenance = json.loads((INPUT / 'loss_provenance.json').read_text())
    lookup = {r['arm']: r for r in provenance['arms']}
    arms = [dict(arm=a, label=arm_label(a), MSE=float(data['global_mean_squared'][i]),
                 MAE=float(data['global_mean_absolute'][i]), **{k: lookup[a][k] for k in ['provenance', 'seeds']}) for i, a in enumerate(names)]
    write_csv('loss_arms.csv', arms)
    write_csv('loss_per_seed.csv', provenance['per_seed_metrics'])
    rows = []
    for k, (i, j) in enumerate(data['pairs']):
        a, b = names[i], names[j]
        r = {'a': a, 'b': b, 'MSE_a': arms[i]['MSE'], 'MSE_b': arms[j]['MSE'],
             'MAE_a': arms[i]['MAE'], 'MAE_b': arms[j]['MAE'],
             'delta_MSE': arms[i]['MSE'] - arms[j]['MSE'],
             'delta_MAE': arms[i]['MAE'] - arms[j]['MAE'],
             'MSE_percent_vs_b': 100 * (arms[i]['MSE'] / arms[j]['MSE'] - 1),
             'MSE_percent_vs_a_reverse': 100 * (arms[j]['MSE'] / arms[i]['MSE'] - 1)}
        for lag in (0, 1, 3, 6):
            mean, se, statistic, p = dm_from_months(data['pair_monthly_squared'][k], lag)
            r.update({f'DM_lag{lag}': statistic, f'p_DM_lag{lag}': p})
            if lag == 0:
                r.update(monthly_mean_delta_MSE=mean, DM_standard_error=se,
                         TOST_p_reference_b=tost(mean, se, .05 * arms[j]['MSE']),
                         TOST_p_reference_a=tost(mean, se, .05 * arms[i]['MSE']))
        _, _, r['DM_MAE_lag0'], r['p_DM_MAE_lag0'] = dm_from_months(data['pair_monthly_absolute'][k])
        r['wilcoxon_z'], r['p_wilcoxon'] = wilcoxon_series(data['series_mean_absolute'][i], data['series_mean_absolute'][j])
        # Stable log tail records magnitude even if ordinary p underflows to zero.
        r['wilcoxon_log10_p'] = float((math.log(2) + stats.norm.logsf(abs(r['wilcoxon_z']))) / math.log(10))
        rows.append(r)
    for field in ['p_DM_lag0', 'p_DM_lag1', 'p_DM_lag3', 'p_DM_lag6', 'p_DM_MAE_lag0', 'p_wilcoxon']:
        for r, p in zip(rows, holm([r[field] for r in rows])):
            r[field + '_Holm_all'] = float(p)
        subset = [r for r in rows if r['a'] in PAPER_ARMS and r['b'] in PAPER_ARMS]
        for r, p in zip(subset, holm([r[field] for r in subset])):
            r[field + '_Holm_paper'] = float(p)
    old = {(r['model_a'], r['model_b']): r for r in read_csv(INPUT / 'significance_all_pairs_archived.csv')}
    mismatch = []
    for r in rows:
        previous = old.get((r['a'], r['b']))
        if previous is None or abs(r['DM_lag0'] - float(previous['dm_sq'])) > .00051 or abs(r['wilcoxon_z'] - float(previous['wilcoxon_z_abs'])) > .00051:
            mismatch.append([r['a'], r['b']])
    if mismatch:
        raise ValueError(f'Archived comparison mismatch: {mismatch[:5]}')
    write_csv('all_pairwise_tests.csv', rows)
    write_csv('paper_pairwise_tests.csv', [r for r in rows if r['a'] in PAPER_ARMS and r['b'] in PAPER_ARMS])
    summary = dict(configurations=len(arms), unique_pairs=len(rows), source_shape=data['shape'].tolist(),
                   archived_statistics_reproduced=len(rows),
                   revived_configurations=sum('revived;' in a['provenance'] for a in arms),
                   DM_unadjusted_significant=sum(r['p_DM_lag0'] < .05 for r in rows),
                   DM_HAC3_Holm_all_significant=sum(r['p_DM_lag3_Holm_all'] < .05 for r in rows),
                   paper_DM_unadjusted_significant=sum(r['p_DM_lag0'] < .05 for r in rows if r['a'] in PAPER_ARMS and r['b'] in PAPER_ARMS),
                   paper_DM_HAC3_Holm136_significant=sum(r.get('p_DM_lag3_Holm_paper', 1) < .05 for r in rows))
    (TABLES / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    return arms, rows, summary


def make_plots(results, arms, pairs, dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#303946',
                         'text.color': '#1C2736', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'savefig.facecolor': 'white'})
    catalog = []

    def save(fig, name, title, description):
        for fmt in ('png', 'svg', 'pdf'):
            fig.savefig(FIGS / f'{name}.{fmt}', dpi=dpi, bbox_inches='tight', pad_inches=.22,
                        metadata={'Creator': 'HPEC2026 results gallery'} if fmt == 'pdf' else None)
            if fmt == 'svg':
                path = FIGS / f'{name}.{fmt}'
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        plt.close(fig)
        catalog.append(dict(file=name, title=title, description=description))
        print('Rendered ' + name, flush=True)

    def header(fig, title, subtitle):
        fig.suptitle(title, x=.035, y=.995, ha='left', fontsize=19, weight='bold')
        fig.text(.035, .957, subtitle, fontsize=10, ha='left', va='top', color='#596579')

    def heatmap(roster, kind, name, title):
        n = len(roster)
        index = {a: i for i, a in enumerate(roster)}
        matrix = np.full((n, n), np.nan)
        for r in pairs:
            if r['a'] not in index or r['b'] not in index:
                continue
            i, j = index[r['a']], index[r['b']]
            if kind == 'relative':
                matrix[i, j], matrix[j, i] = r['MSE_percent_vs_b'], r['MSE_percent_vs_a_reverse']
            elif kind == 'equivalence':
                matrix[i, j], matrix[j, i] = int(r['TOST_p_reference_b'] < .05), int(r['TOST_p_reference_a'] < .05)
            elif kind == 'decisions':
                key = 'p_DM_lag3_Holm_' + ('paper' if n == 17 else 'all')
                decision = np.sign(r['DM_lag3']) if r[key] < .05 else 0
                matrix[i, j], matrix[j, i] = decision, -decision
            else:
                matrix[i, j], matrix[j, i] = r[kind], -r[kind]
        size = 21 if n > 20 else 13
        fig, ax = plt.subplots(figsize=(size, size * .90))
        fig.subplots_adjust(left=.21, right=.90, bottom=.22, top=.90)
        if kind == 'decisions':
            cmap = ListedColormap(['#286E9F', '#EEEEEE', '#C6534A'])
            image = ax.imshow(matrix, cmap=cmap, norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3))
            unit = 'HAC lag 3 + Holm: blue = row lower MSE; grey = not detected; red = row higher MSE'
            subtitle = f'{n} configurations, {n * (n - 1) // 2:,} tests in this correction family. No equivalence inference from grey cells.'
        elif kind == 'equivalence':
            image = ax.imshow(matrix, cmap=ListedColormap(['#EFEFEF', '#527898']), vmin=0, vmax=1)
            unit = 'Blue = equivalence detected; grey = not detected'
            subtitle = 'MSE TOST, unadjusted alpha 0.05; margin = 5% of COLUMN MSE; directional margins explain asymmetry.'
        else:
            limit = float(np.nanmax(np.abs(matrix)))
            image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit)
            unit = {'DM_lag0': 'Signed DM statistic (squared loss)', 'wilcoxon_z': 'Signed Wilcoxon z (per-series MAE)',
                    'delta_MSE': 'MSE: row minus column', 'relative': 'MSE change relative to column (%)'}[kind]
            subtitle = f'Full linear color range, no magnitude clipping. Blue = row lower; red = row higher. Diagonal omitted.'
        labels = [arm_label(a) for a in roster]
        ax.set_xticks(range(n), labels, rotation=55, ha='right', rotation_mode='anchor', fontsize=8 if n > 20 else 9)
        ax.set_yticks(range(n), labels, fontsize=8 if n > 20 else 9)
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.35)
        ax.tick_params(which='both', length=0)
        # Full numeric cells are included even when the color scale is dominated by large values.
        if kind not in ('decisions', 'equivalence'):
            limit = float(np.nanmax(np.abs(matrix)))
            for i, j in itertools.product(range(n), repeat=2):
                if i == j:
                    continue
                v = matrix[i, j]
                text = f'{v:.2f}' if kind == 'delta_MSE' else f'{v:.1f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=5 if n > 20 else 6.8,
                        color='white' if abs(v) > .58 * limit else '#172534')
        bar = fig.colorbar(image, ax=ax, fraction=.026, pad=.025)
        bar.set_label(unit, fontsize=10)
        if kind == 'decisions':
            bar.set_ticks([-1, 0, 1], labels=['Row lower', 'Not detected', 'Row higher'])
        elif kind == 'equivalence':
            bar.set_ticks([0, 1], labels=['Not detected', 'Detected'])
        fig.suptitle(title, x=.035, y=.98, ha='left', fontsize=21, weight='bold')
        fig.text(.035, .943, subtitle, fontsize=10)
        fig.text(.035, .025, 'Archived losses: 30,087 series x 24 months; 3-seed mean losses. Eight configurations use revived checkpoints.\nStatistical assumptions and provenance limits: results/gallery/README.md. Full-precision values: tables/all_pairwise_tests.csv.', fontsize=9)
        save(fig, name, title, subtitle + ' ' + unit)

    roster = [a['arm'] for a in sorted(arms, key=lambda a: a['MSE'])]
    for kind, suffix, title in [('DM_lag0', 'dm', 'Diebold-Mariano: full signed magnitudes'),
                                ('wilcoxon_z', 'wilcoxon', 'Wilcoxon: full signed magnitudes'),
                                ('delta_MSE', 'mse_difference', 'Absolute MSE differences'),
                                ('relative', 'mse_relative', 'Relative MSE differences'),
                                ('decisions', 'hac_holm', 'Difference-test sensitivity: HAC + Holm'),
                                ('equivalence', 'equivalence', 'MSE equivalence: separate from difference tests')]:
        heatmap(roster, kind, 'heatmap_all_' + suffix, title + ' / 46 configurations')
    for kind, suffix, title in [('DM_lag0', 'dm', 'Paper configurations: full DM magnitudes'),
                                ('wilcoxon_z', 'wilcoxon', 'Paper configurations: full Wilcoxon magnitudes'),
                                ('decisions', 'hac_holm', 'Paper configurations: HAC + Holm sensitivity')]:
        heatmap(PAPER_ARMS, kind, 'heatmap_paper_' + suffix, title)

    aggregate = [r for r in results if r['panel'] == 'aggregate']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, metric in zip(axes, ['MSE', 'MAE']):
        rr = sorted(aggregate, key=lambda r: r[metric])
        ax.barh(range(len(rr)), [r[metric] for r in rr], color=[COLORS.get(r['model'], '#929BA8') for r in rr], height=.64)
        if metric == 'MSE':
            ax.errorbar([r[metric] for r in rr], range(len(rr)), xerr=[r['MSE_std'] for r in rr], fmt='none', color='#283445', capsize=3)
        ax.set_yticks(range(len(rr)), [r['label'] for r in rr]); ax.invert_yaxis()
        ax.set_xlabel(metric + ' (normalized targets)'); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
        for y, r in enumerate(rr):
            ax.text(r[metric] + (r['MSE_std'] if metric == 'MSE' else 0) + max(x[metric] for x in rr) * .025,
                    y, f'{r[metric]:.5f}', va='center', fontsize=9)
        ax.set_xlim(0, max(r[metric] + (r['MSE_std'] if metric == 'MSE' else 0) for r in rr) * 1.25)
    header(fig, 'Fixed aggregate forecast', 'All 10 models. MSE whiskers are population SD across 3 seeds; deterministic baselines have no seed variation.')
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, 'aggregate_accuracy', 'Fixed aggregate accuracy', 'MSE and MAE; full uncertainty whiskers; all ten models.')

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, metric in zip(axes, ['MSE', 'MAE']):
        for r in aggregate:
            if r['model'] in ('random_walk', 'moving_average', 'seasonal_naive'):
                ax.axhline(r[metric], color='#929BA8', ls=':', lw=1, label=r['label'] + ' (accuracy reference)')
                continue
            bound = r['flops_upper_bound'] is not None
            x = (r['flops_upper_bound'] if bound else r['flops_per_forecast']) / 1e6
            ax.errorbar(x, r[metric], yerr=r['MSE_std'] if metric == 'MSE' else None,
                        fmt='D' if bound else 'o', mfc='white' if bound else COLORS[r['model']],
                        color=COLORS[r['model']], ms=8, capsize=3, label=r['label'] + (' (upper bound)' if bound else ''))
        ax.set_xscale('log'); ax.set_xlabel('Arithmetic MFLOPs per joint forecast (log scale)')
        ax.set_ylabel(metric); ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    header(fig, 'Accuracy and inference arithmetic', 'Hollow XGBoost diamond is a <=0.001 MFLOP bound, not a measurement. Comparisons and data movement are excluded.')
    fig.tight_layout(rect=(0, .04, 1, .90))
    save(fig, 'aggregate_cost_accuracy', 'Aggregate cost versus accuracy', 'Neural costs retain archived profiling conventions; XGBoost is explicitly an analytical upper bound.')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    annual = [r for r in results if r['panel'] == 'aggregate_rolling' and 'pooled' not in r['variant']]
    for ax, metric in zip(axes, ['MSE', 'MASE']):
        for model in dict.fromkeys(r['model'] for r in aggregate):
            rr = sorted([r for r in annual if r['model'] == model], key=lambda r: r['variant'])
            ax.plot([int(r['variant'][-4:]) for r in rr], [r[metric] for r in rr], marker='o', ms=4,
                    color=COLORS.get(model, {'random_walk': '#515B65', 'moving_average': '#87959E', 'seasonal_naive': '#BABEC3'}.get(model)),
                    ls='--' if model in ('random_walk', 'moving_average', 'seasonal_naive') else '-', label=NAMES[model])
        ax.set_xlabel('Test year'); ax.set_ylabel(metric); ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    header(fig, 'Rolling-origin aggregate forecasts', 'Six annual folds, 2020-2025. Seed-mean metrics; each fold refits normalization using its own training history.')
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, 'rolling_comparison', 'Rolling-origin comparison', 'All ten models across all six annual test folds.')

    series = [r for r in results if r['panel'] == 'series']
    def find(model, variant, enc='embeddings'):
        return next(r for r in series if r['model'] == model and r['variant'] == variant and r['enc'] == enc)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    models = ['s4', 'mamba2', 'gru', 'transformer', 'lstm', 'mamba3']
    changes = []
    for k, (variant, color) in enumerate([('embeddings', '#397AA6'), ('onehot', '#D08B45')]):
        rr = [find(m, variant, variant) for m in models]
        axes[0].bar(np.arange(6) + (k-.5)*.34, [r['MSE'] for r in rr], width=.32,
                    yerr=[r['MSE_std'] for r in rr], capsize=3, label=variant, color=color)
    for m in models:
        a, b = find(m, 'embeddings'), find(m, 'onehot', 'onehot')
        changes.append(dict(model=m, embedding_MSE=a['MSE'], onehot_MSE=b['MSE'],
                            percent_change_of_means=100*(b['MSE']/a['MSE']-1)))
    write_csv('encoder_changes.csv', changes)
    axes[0].set_yscale('log'); axes[0].set_ylabel('MSE (log scale); mean +/- population SD'); axes[0].legend()
    axes[1].bar(range(6), [r['percent_change_of_means'] for r in changes], color='#397AA6')
    axes[1].axhline(0, color='#666', lw=.8); axes[1].set_ylabel('One-hot versus embeddings: MSE change (%)')
    for ax in axes:
        ax.set_xticks(range(6), [NAMES[m] for m in models], rotation=25, ha='right'); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    header(fig, 'Flat encoders and cross-seed spread', 'Percentage changes are ratios of means, without uncertainty whiskers. Paired per-seed percentages are not available for every arm.')
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, 'encoder_comparison', 'Embedding versus one-hot encoders', 'No propagated uncertainty is presented as empirical paired-seed SD.')

    mixes = [('asa', 'ASA', '#347CAE'), ('fa', 'FA', '#D2944B'), ('fa_sm', 'FA-softmax', '#9873B4'), ('fa_local', 'CoFA', '#328570')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    for ax, model in zip(axes, ['gru', 'lstm', 'transformer']):
        for prefix, label, color in mixes:
            rr = [find(model, f'{prefix}_{d}d') for d in [2, 3, 4]]
            ax.errorbar([2, 3, 4], [r['MSE'] for r in rr], yerr=[r['MSE_std'] for r in rr],
                        marker='o', capsize=4, label=label, color=color)
        ax.set_title(NAMES[model]); ax.set_xticks([2,3,4], ['2D','3D','4D']); ax.set_ylabel('MSE'); ax.grid(alpha=.15)
        ax.margins(y=.20)
    axes[-1].legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02,1))
    header(fig, 'Attention-operator comparison', 'All 36 embedding configurations; mean +/- population SD over 3 seeds. Axes include every whisker endpoint.')
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, 'attention_ablation', 'ASA, FA, FA-softmax and CoFA', 'Independent panel scales; all uncertainty endpoints visible.')

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, model in zip(axes.flat, ['gru', 'lstm', 'transformer']):
        base = find(model, 'embeddings')
        for prefix, label, color in mixes:
            rr = [base] + [find(model, f'{prefix}_{d}d') for d in [2,3,4]]
            ax.errorbar([1,2,3,4], [r['MSE'] for r in rr], yerr=[r['MSE_std'] for r in rr], marker='o', capsize=3, label=label, color=color)
        ax.set_title(NAMES[model]); ax.set_xticks([1,2,3,4], ['Flat','2D','3D','4D']); ax.set_ylabel('MSE'); ax.grid(alpha=.15); ax.margins(y=.15)
    ax = axes[1,1]
    for native, flat, label in [('s4nd','s4','S4 / S4ND'), ('mamba_nd','mamba2','Mamba-2 / Mamba-ND')]:
        rr = [find(flat, 'embeddings')] + [find(native, f'grid_{d}d') for d in [2,3,4]]
        ax.errorbar([1,2,3,4], [r['MSE'] for r in rr], yerr=[r['MSE_std'] for r in rr], marker='s', capsize=3, label=label, color=COLORS[native])
    for model in ('moving_average','random_walk'):
        r = next(r for r in series if r['model'] == model)
        ax.axhline(r['MSE'], color='#8B929C', ls=':' if model=='moving_average' else '--', label=r['label'])
    ax.set_title('Native multidimensional SSMs'); ax.set_ylabel('MSE (separate scale)'); ax.set_xticks([1,2,3,4], ['Flat','2D','3D','4D']); ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9, ncol=2); ax.legend(fontsize=8)
    header(fig, 'Dimensionality and model architecture', 'Embedding encoders; population SD across 3 seeds. Flat references are shared across operator curves, not separate runs.')
    fig.tight_layout(rect=(0, 0, 1, .925))
    save(fig, 'dimensionality', 'Dimensionality comparison', 'Flat, 2D, 3D, and 4D means and full SD whiskers.')

    fig, ax = plt.subplots(figsize=(13, 8))
    for m in sorted(set(r['model'] for r in series)):
        rr = [r for r in series if r['model'] == m and r['flops_per_forecast'] is not None]
        if not rr:
            continue
        ax.scatter([r['flops_per_forecast']/1e6 for r in rr], [r['MSE'] for r in rr],
                   color=COLORS.get(m, '#697584'), label=NAMES[m], alpha=.75, s=45)
    for m in ['moving_average','random_walk','seasonal_naive']:
        r=next(r for r in series if r['model']==m)
        ax.axhline(r['MSE'], color='#9BA2A9', ls=':', lw=.8)
        ax.text(.015, r['MSE'], r['label'], transform=ax.get_yaxis_transform(), fontsize=8, va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Archived arithmetic MFLOPs per joint forecast (log scale)'); ax.set_ylabel('MSE (log scale)')
    ax.grid(alpha=.15); ax.legend(loc='upper left', bbox_to_anchor=(1.01,1))
    header(fig, 'All available series-level cost and accuracy results', 'Every series configuration with an archived cost entry. Statistical baselines are accuracy-reference lines, not zero-cost measurements.')
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, 'series_cost_accuracy', 'All-configuration cost versus accuracy', 'Colors identify model families; full configuration identifiers and values are in all_results.csv.')

    # Ranked overview includes all series configurations, not only the loss-dump subset.
    ranked = sorted(series, key=lambda r:r['MSE'])
    chunks = [ranked[k:k+40] for k in range(0,len(ranked),40)]
    for page, rr in enumerate(chunks,1):
        fig, ax = plt.subplots(figsize=(13, max(7, len(rr)*.30 + 2)))
        ax.errorbar([r['MSE'] for r in rr], range(len(rr)), xerr=[r['MSE_std'] for r in rr], fmt='none', ecolor='#AFB7BF', capsize=2)
        ax.scatter([r['MSE'] for r in rr], range(len(rr)), c=[COLORS.get(r['model'],'#626E7C') for r in rr], s=32)
        ax.set_yticks(range(len(rr)), [f'{(page-1)*40+i+1:03d}  '+r['label'] for i,r in enumerate(rr)], fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(left=0); ax.set_xlabel('MSE; mean +/- population SD across seeds'); ax.grid(axis='x',alpha=.15)
        fig.suptitle(f'All series configurations / rank {(page-1)*40+1}-{(page-1)*40+len(rr)}',fontsize=17,weight='bold',x=.03,ha='left')
        fig.tight_layout(rect=(0,0,1,.955))
        save(fig,f'series_ranking_{page:02d}',f'Series MSE ranking, page {page}', 'Descriptive ordering, not a statistical superiority claim. Full whiskers shown.')

    lr=read_csv(INPUT/'lr_direction.csv')
    categories=list(dict.fromkeys(r[k] for r in lr for k in ['seed947','seed732','seed619']))
    mat=np.array([[categories.index(r[k]) for k in ['seed947','seed732','seed619']] for r in lr])
    fig,ax=plt.subplots(figsize=(9,14))
    ax.imshow(mat,cmap=ListedColormap(['#90B6A0','#E3B67E','#B1BCCC'][:len(categories)]),aspect='auto',vmin=-.5,vmax=len(categories)-.5)
    ax.set_yticks(range(len(lr)),[r['cell'] for r in lr],fontsize=9);ax.set_xticks([0,1,2],['Seed 947','Seed 732','Seed 619'])
    for i,r in enumerate(lr):
        for j,k in enumerate(['seed947','seed732','seed619']):ax.text(j,i,r[k],ha='center',va='center',fontsize=7)
    ax.set_title('Archived learning-rate diagnostic judgments',loc='left',pad=20)
    fig.text(.05,.015,'38 displayed configurations. These categories are diagnostic hypotheses, not demonstrated optimal learning rates.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    save(fig,'learning_rate_diagnostics','Learning-rate diagnostics','Original categories are preserved verbatim; roster does not cover every model.')
    return catalog


def make_html(results, catalog, summary):
    cards = '\n'.join(f'<article><h2>{html.escape(c["title"])}</h2><a href="figures/{c["file"]}.png"><img loading="lazy" src="figures/{c["file"]}.png" alt="{html.escape(c["title"])}"></a><p>{html.escape(c["description"])}</p><p>' + ' / '.join(f'<a href="figures/{c["file"]}.{ext}">{ext.upper()}</a>' for ext in ['png','svg','pdf']) + '</p></article>' for c in catalog)
    fields = ['panel','label','variant','enc','n'] + METRICS + ['flops_per_forecast','flops_upper_bound','analytic_flops','params','peak_mem_GB','cost_status','source']
    body = ''.join('<tr>' + ''.join('<td>'+html.escape('Not recorded' if r.get(k) is None else str(r[k]))+'</td>' for k in fields) + '</tr>' for r in results)
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPEC 2026 results</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#1c2736;background:#f4f6f9;margin:0}}main{{max-width:1380px;margin:auto;padding:40px 24px}}h1{{font-size:38px;line-height:1.15}}h2{{font-size:23px}}a{{color:#226294}}article{{background:white;margin:24px 0;padding:24px;border:1px solid #d9dfe7;border-radius:10px}}img{{max-width:100%;height:auto}}.table{{overflow:auto;max-height:75vh}}table{{border-collapse:collapse;font-size:12px}}th,td{{padding:9px 12px;border-bottom:1px solid #e1e6eb;white-space:nowrap;text-align:right}}th{{position:sticky;top:0;background:#e6edf5}}td:nth-child(-n+4){{text-align:left}}input{{padding:12px;width:min(500px,90%);font:inherit}}.note{{border-left:4px solid #527898;padding-left:18px}}code{{background:#e9edf2;padding:2px 5px}}</style>
<main><p>IEEE HPEC 2026 / supplementary result gallery / 23 September 2026</p><h1>Accuracy, compute and multidimensional structure</h1>
<p>{len(results)} result rows; {summary['configurations']} configurations with paired loss evidence; {summary['unique_pairs']:,} unique pairwise comparisons.</p>
<p class="note">This is a re-analysis and rendering of archived results, not a new training run or publisher correction. Heatmaps retain the full magnitude range. Statistical robustness, seed variation and checkpoint provenance are separate qualifications. Read <a href="README.md">methods and limitations</a>.</p>
<p><a href="hpec_results.xlsx">Excel workbook</a> / <a href="tables/all_results.csv">All results (CSV)</a> / <a href="tables/all_pairwise_tests.csv">All comparisons (CSV)</a> / <a href="tables/all_costs.csv">All archived costs (CSV)</a></p>
{cards}<article><h2>Complete result table</h2><p>Stored precision is shown; it is not a claim of statistical precision. Blank source metrics remain unavailable. Search models, configurations or panels.</p><input id="search" aria-label="Filter result table" placeholder="Filter rows"><div class="table"><table><thead><tr>{''.join('<th>'+html.escape(k)+'</th>' for k in fields)}</tr></thead><tbody id="rows">{body}</tbody></table></div></article></main>
<script>document.getElementById('search').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('#rows tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});</script></html>'''
    (OUT/'index.html').write_text(text)
    (OUT/'figure_catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--tables-only',action='store_true')
    args=parser.parse_args()
    for p in [TABLES,FIGS]:p.mkdir(parents=True,exist_ok=True)
    for record in json.loads((INPUT/'SOURCES.json').read_text()):
        with (INPUT/record['file']).open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=record['sha256']:
                raise ValueError('Source checksum mismatch: '+record['file'])
    results=full_results()
    arms,pairs,summary=comparisons()
    print(json.dumps(summary,indent=2),flush=True)
    if not args.tables_only:
        catalog=make_plots(results,arms,pairs,args.dpi)
        make_html(results,catalog,summary)
    data={'results':results,'arms':arms,'pairs':pairs,'costs':read_csv(ROOT/'results/model_cost_FINAL.csv'),
          'xgboost':json.loads((INPUT/'xgboost_all.json').read_text())['rows'],
          'seed_metrics':json.loads((INPUT/'loss_provenance.json').read_text())['per_seed_metrics'],
          'training':read_csv(INPUT/'training_diagnostics.csv'),'lr':read_csv(INPUT/'lr_direction.csv')}
    (TABLES/'workbook_data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    inputs = [p for p in INPUT.iterdir() if p.is_file()] + [ROOT/'results'/p for p in ['flat_and_nd_results.csv','aggregate_and_identity_results.csv','rolling_annual.csv','rolling_pooled.csv','model_cost_FINAL.csv']] + [ROOT/'artifacts'/p/'results.csv' for p in ['exp7','exp8']]
    (OUT/'MANIFEST.json').write_text(json.dumps({'source_files':[{ 'file':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
                                               'numpy':np.__version__, 'scipy':__import__('scipy').__version__, 'seed_sd_ddof':0, 'dpi':args.dpi},indent=2)+'\n')


if __name__=='__main__':main()
