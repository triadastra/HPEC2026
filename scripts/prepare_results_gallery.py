#!/usr/bin/env python3
"""Extract compact, reproducible loss statistics from the recovered archive.

No training or inference. The large loss archive remains external to Git.
Usage: python scripts/prepare_results_gallery.py --archive-dir PATH
PATH must contain FINAL_BATCH/ and xgboost_all.json.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import significance_tests as sig


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True)
    args = parser.parse_args()
    out = ROOT / 'results/gallery/inputs'
    out.mkdir(parents=True, exist_ok=True)
    files = {
        'xgboost_all.json': args.archive_dir / 'xgboost_all.json',
        'figure_inputs_archived.json': args.archive_dir / 'FINAL_BATCH/code/fig_data.json',
        'significance_all_pairs_archived.csv': args.archive_dir / 'FINAL_BATCH/significance_all_pairs.csv',
        'training_diagnostics.csv': args.archive_dir / 'FINAL_BATCH/training_diagnostics.csv',
        'lr_direction.csv': args.archive_dir / 'FINAL_BATCH/lr_direction.csv',
    }
    sources = []
    for dest, source in files.items():
        (out / dest).write_bytes(source.read_bytes())
        sources.append({'file': dest, 'source_archive_member': str(source.relative_to(args.archive_dir)),
                        'sha256': sha(source)})
    path = args.archive_dir / 'FINAL_BATCH/errdump_ALL_final.npz'
    z, meta, shape = sig.load_errors(path)
    names = sorted({key.rsplit('_s', 1)[0] for key in meta['runs']}) + ['persistence']
    arrays = {loss: [] for loss in ('squared', 'absolute')}
    seed_metrics, provenance = [], []
    for name in names:
        if name == 'persistence':
            provenance.append({'arm': name, 'provenance': 'archived baseline loss arrays', 'seeds': 0})
        else:
            runs = [meta['runs'][f'{name}_s{s}'] for s in (947, 732, 619)]
            revived = any(r.get('revived', False) for r in runs)
            provenance.append({'arm': name, 'provenance': 'revived; tree fingerprint not revalidated; five-decimal logged-MSE gate' if revived else 'archive declares fingerprint-validated inference', 'seeds': 3})
            sq = sig.seedstack(z, meta, shape, name, 'squared')
            ab = sig.seedstack(z, meta, shape, name, 'absolute')
            for seed, a, b in zip((947, 732, 619), sq, ab):
                seed_metrics.append({'arm': name, 'seed': seed, 'MSE': float(a.mean()), 'MAE': float(b.mean())})
        for loss in arrays:
            matrix = sig.baseline_matrix(z, shape, loss) if name == 'persistence' else sig.seedavg(z, meta, shape, name, loss)
            if matrix.shape != shape or not np.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError(f'Invalid {name} {loss} loss array')
            arrays[loss].append(matrix)
        print(f'Extracted {name}', flush=True)
    pairs = np.array(list(itertools.combinations(range(len(names)), 2)))
    payload = {'names': np.array(names), 'pairs': pairs, 'shape': np.array(shape)}
    for loss, mats in arrays.items():
        # Preserve original float32 subtraction/reduction for exact DM replay.
        payload[f'pair_monthly_{loss}'] = np.stack([(mats[i] - mats[j]).mean(axis=0) for i, j in pairs])
        payload[f'series_mean_{loss}'] = np.stack([m.mean(axis=1) for m in mats])
        payload[f'global_mean_{loss}'] = np.array([float(m.mean()) for m in mats])
    np.savez_compressed(out / 'loss_summaries.npz', **payload)
    (out / 'loss_provenance.json').write_text(json.dumps({
        'source_archive': path.name, 'source_sha256': sha(path), 'shape': list(shape),
        'seed_roster': [947, 732, 619], 'arms': provenance, 'per_seed_metrics': seed_metrics,
        'merged_from': [{'file': Path(r['file']).name, 'provenance': r['provenance'], 'runs': r['runs']} for r in meta['merged_from']],
        'runs': meta['runs'],
        'scope': 'Loss-summary extraction only; no fresh checkpoint inference or provenance revalidation.'
    }, indent=2) + '\n')
    sources.extend({'file': f, 'source_archive_member': path.name + ' (derived compact statistics)',
                    'sha256': sha(out / f)} for f in ['loss_summaries.npz', 'loss_provenance.json'])
    (out / 'SOURCES.json').write_text(json.dumps(sources, indent=2) + '\n')
    print(f'{len(names)} arms, {len(pairs)} pairs; compact inputs saved to {out}')


if __name__ == '__main__':
    main()
