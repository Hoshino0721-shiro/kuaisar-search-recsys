"""Validation-only aggregation and independent artifact consistency checks.

Activity bins are descriptive post-training diagnostics, never model selection.
No training or test-label reads occur here.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.ctr_features import CACHE
from src.ranking import metrics
from src.v2.ranking_sensitivity import paired_interval
from src.v2.train_rankers import sha


def summarize(root, pool, output):
    if output.exists():
        raise FileExistsError('Preserve completed report')
    training = json.loads((root/'models/metrics.json').read_text())
    evaluation = json.loads((root/'evaluation/metrics.json').read_text())
    rows = json.loads((root/'evaluation/per_user.json').read_text())
    meta = evaluation['candidate_metadata']
    assert not training['test_used'] and not evaluation['test_used']
    assert sha(pool/'candidates.jsonl') == meta['candidate_sha256']
    assert sorted(r['user_id'] for r in rows) == meta['users']
    assert len({r['user_id'] for r in rows}) == len(rows)
    data_sha = sha(root/'cache/metadata.json')
    for name, run in training['runs'].items():
        assert sha(root/'models'/name/'model.pt') == run['checkpoint_sha256']
        assert run['metadata']['data_sha256'] == data_sha
        assert run['metadata']['model_source_sha256'] == sha(Path(__file__).with_name('enhanced_model.py'))
        assert run['metadata']['trainer_source_sha256'] == sha(Path(__file__).with_name('enhanced_train.py'))
        assert sum(r['methods'][name]['hit_events'] for r in rows) == evaluation['direct'][name]['hit_events']
    for name, stages in evaluation['cascade'].items():
        for stage, result in stages.items():
            assert sum(r['cascade'][name][stage]['hit_events'] for r in rows) == result['hit_events']
    freeze = json.loads(Path('experiments/final_test/freeze.json').read_text())
    assert all(sha(p) == digest for p, digest in freeze['asset_hashes'].items())
    groups = {}
    for name in training['runs']:
        groups.setdefault(name.rsplit('_s',1)[0], []).append(name)
    summary = {}
    for group, names in groups.items():
        summary[group] = {'seeds': names, 'exposure': {}, 'direct': {}}
        for section, source, fields in [
                ('exposure', {n:training['runs'][n]['validation']['all'] for n in names},
                 ['auc','gauc','logloss','brier','ece_10_bins']),
                ('direct', evaluation['direct'], ['hit_events','ndcg_at20','event_recall_at20'])]:
            for field in fields:
                values = [source[n][field] for n in names]
                summary[group][section][field] = {'mean':float(np.mean(values)),
                    'std_across_training_seeds':float(np.std(values)), 'values':values}
        summary[group]['parameters'] = training['runs'][names[0]]['parameters']
        summary[group]['checkpoint_bytes'] = [training['runs'][n]['checkpoint_bytes'] for n in names]
        summary[group]['selected_epochs'] = [training['runs'][n]['metadata']['selected_epoch'] for n in names]
        summary[group]['exposure_groups'] = {}
        for subgroup in training['runs'][names[0]]['validation']:
            summary[group]['exposure_groups'][subgroup] = {
                m:float(np.mean([training['runs'][n]['validation'][subgroup][m] for n in names]))
                for m in ['rows','auc','gauc','logloss']}
    pairs = {}
    for before, after in [('A','B'),('B','item_only'),('item_only','D'),('D','E')]:
        delta = []
        for row in rows:
            values = [[row['methods'][n]['user_period_ndcg_at_20'] for n in groups[g]] for g in (before,after)]
            if all(v is not None for group_values in values for v in group_values):
                delta.append(np.mean(values[1])-np.mean(values[0]))
        pairs[after+' minus '+before] = paired_interval(delta)
    # User activity = full training recommendation exposure count, not valid clicks.
    users = np.load(CACHE/'valid_users.npy')
    y = np.load(CACHE/'valid_y.npy')
    counts = np.bincount(np.load(CACHE/'train_users.npy'))
    frequency = np.zeros(len(users),dtype=np.int64)
    present = users < len(counts)
    frequency[present] = counts[users[present]]
    bins = {'unseen':frequency==0, '1_to_99':(frequency>0)&(frequency<100),
            '100_to_499':(frequency>=100)&(frequency<500), '500_plus':frequency>=500}
    activity = {}
    for name in training['runs']:
        prediction = np.load(root/'models'/name/'valid_predictions.npy')
        assert len(prediction) == len(y)
        activity[name] = {g:metrics(y[m],prediction[m],users[m]) for g,m in bins.items() if m.any()}
    report = {'test_used':False, 'integrity_checks_passed':True, 'users':len(rows),
              'training_runs':len(training['runs']), 'groups':summary,
              'paired_ndcg':pairs, 'user_activity':activity,
              'activity_scope':'post-training descriptive bins from full training exposures; not selection or preregistered thresholds',
              'interval_scope':'exploratory user bootstrap; seeds averaged first; reused one validation day, no multiple-testing correction or independent time-window guarantee',
              'candidate_sha256':meta['candidate_sha256'], 'data_sha256':data_sha,
              'source_sha256':sha(__file__), 'frozen_v1_assets_unchanged':True}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Enhanced summary and integrity checks passed:',output,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    summarize(a.run,a.pool,a.out)
