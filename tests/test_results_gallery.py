"""Regression tests for the supplementary tables and plotting conventions."""
import csv
import hashlib
import json
import math
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from scripts import results_gallery as gallery

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gallery'


def rows(name):
    return gallery.read_csv(OUT / 'tables' / name)


def test_holm_known_family_and_invalid_values():
    assert gallery.holm([.01, .04, .03]).tolist() == pytest.approx([.03, .06, .06])
    with pytest.raises(ValueError):
        gallery.holm([.1, float('nan')])


def test_dm_matches_paired_t_for_lag_zero_and_reverses_sign():
    differences = np.array([.4,.1,-.2,.6,.3,.7,.2,.5], dtype=float)
    mean, se, statistic, p = gallery.dm_from_months(differences)
    assert mean == pytest.approx(differences.mean())
    assert se == pytest.approx(differences.std(ddof=1) / math.sqrt(len(differences)))
    assert statistic == pytest.approx(stats.ttest_1samp(differences,0).statistic)
    assert p == pytest.approx(stats.ttest_1samp(differences,0).pvalue)
    assert gallery.dm_from_months(-differences)[2:] == pytest.approx((-statistic,p))
    with pytest.raises(ValueError):
        gallery.dm_from_months(np.ones(24))


def test_hac_uses_bartlett_weights_and_hln_scaling():
    d = np.array([.01,.02,.08,.10,.04,.09,.12,.08,.16,.19])
    c=d-d.mean(); n=len(d); lag=3
    lrv=sum(c*c)/n + 2*sum((1-k/(lag+1))*sum(c[k:]*c[:-k])/n for k in range(1,lag+1))
    _,se,_,_=gallery.dm_from_months(d,lag)
    assert se == pytest.approx(math.sqrt(lrv/(n-1)))


def test_input_and_original_source_checksums():
    for record in json.loads((OUT/'inputs/SOURCES.json').read_text()):
        assert hashlib.sha256((OUT/'inputs'/record['file']).read_bytes()).hexdigest()==record['sha256']
    source = json.loads((ROOT/'results/SOURCES.json').read_text())
    for record in source['files']:
        assert hashlib.sha256((ROOT/'results'/record['file']).read_bytes()).hexdigest()==record['sha256']


def test_master_table_complete_and_costs_never_imputed_as_zero():
    rr=rows('all_results.csv')
    assert len(rr)==218
    keys=[tuple(r[k] for k in ['panel','model','variant','enc']) for r in rr]
    assert len(set(keys))==len(keys)
    assert sum(r['panel']=='series' for r in rr)==138
    assert sum(r['panel']=='aggregate' for r in rr)==10
    for r in rr:
        if r['model']=='xgboost':
            assert r['flops_per_forecast']==''
            assert float(r['flops_upper_bound'])==1000
            assert float(r['tree_comparisons_upper_bound'])==4000
        if r['cost_status']=='not recorded':
            assert r['flops_per_forecast']==''
        if r['model'] in ('random_walk','seasonal_naive','moving_average'):
            assert r['flops_per_forecast']==''
            assert float(r['analytic_flops'])==(24 if r['model']=='moving_average' else 0)


def test_master_table_preserves_original_accuracy_values():
    master={tuple(r[k] for k in ['panel','model','variant','enc']):r for r in rows('all_results.csv')}
    for path in ['results/flat_and_nd_results.csv','results/aggregate_and_identity_results.csv',
                 'results/rolling_pooled.csv','results/rolling_annual.csv',
                 'artifacts/exp7/results.csv','artifacts/exp8/results.csv']:
        for r in gallery.read_csv(ROOT/path):
            matched=master[tuple(r[k] for k in ['panel','model','variant','enc'])]
            for m in gallery.METRICS:
                assert gallery.number(matched[m])==gallery.number(r.get(m))


def test_xgboost_pooled_rmse_and_seed_sd():
    raw=json.loads((OUT/'inputs/xgboost_all.json').read_text())['rows']
    seed_means=[np.mean([r['both_MSE'] for r in raw if r['panel']=='aggregate_rolling' and r['seed']==s]) for s in gallery.SEEDS]
    pooled=next(r for r in rows('all_results.csv') if r['model']=='xgboost' and r['variant']=='aggregate_roll_pooled')
    assert float(pooled['MSE'])==pytest.approx(np.mean(seed_means))
    assert float(pooled['MSE_std'])==pytest.approx(np.std(seed_means,ddof=0))
    assert float(pooled['RMSE'])==pytest.approx(np.mean(np.sqrt(seed_means)))


def test_all_pairs_reproduce_archived_statistics_and_separate_families():
    rr=rows('all_pairwise_tests.csv')
    assert len(rr)==46*45//2
    original={(r['model_a'],r['model_b']):r for r in gallery.read_csv(OUT/'inputs/significance_all_pairs_archived.csv')}
    for r in rr:
        old=original[r['a'],r['b']]
        assert abs(float(r['DM_lag0'])-float(old['dm_sq'])) <= .00051
        assert abs(float(r['wilcoxon_z'])-float(old['wilcoxon_z_abs'])) <= .00051
        assert float(r['p_DM_lag3_Holm_all']) >= float(r['p_DM_lag3'])-1e-15
        if r['p_DM_lag3_Holm_paper']:
            assert float(r['p_DM_lag3_Holm_all']) >= float(r['p_DM_lag3_Holm_paper'])-1e-15
    assert sum(bool(r['p_DM_lag3_Holm_paper']) for r in rr)==136
    assert sum(float(r['p_DM_lag3_Holm_all'])<.05 for r in rr)==965
    assert sum(float(r['p_DM_lag3_Holm_paper'])<.05 for r in rr if r['p_DM_lag3_Holm_paper'])==130


def test_loss_summary_shape_roster_and_provenance():
    z=np.load(OUT/'inputs/loss_summaries.npz',allow_pickle=False)
    assert z['shape'].tolist()==[30087,24]
    assert z['pair_monthly_squared'].shape==(1035,24)
    assert z['series_mean_absolute'].shape==(46,30087)
    assert set(gallery.PAPER_ARMS)<=set(z['names'])
    provenance=json.loads((OUT/'inputs/loss_provenance.json').read_text())
    assert len(provenance['per_seed_metrics'])==135
    assert sum('revived;' in r['provenance'] for r in provenance['arms'])==8


def test_figure_downloads_exist_and_are_not_empty():
    catalog=json.loads((OUT/'figure_catalog.json').read_text())
    assert len(catalog)>=21
    for item in catalog:
        for ext in ['png','svg','pdf']:
            assert (OUT/'figures'/f'{item["file"]}.{ext}').stat().st_size>1000


def test_html_local_links_and_complete_result_table():
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows=0
            self.links=[]

        def handle_starttag(self,tag,attrs):
            if tag=='tr': self.rows+=1
            self.links.extend(value for key,value in attrs if key in ('src','href'))

    page=Links()
    page.feed((OUT/'index.html').read_text())
    assert page.rows==219  # Header plus every result row.
    for link in page.links:
        if not link.startswith(('https://','http://','#')):
            assert (OUT/link).is_file(),link
