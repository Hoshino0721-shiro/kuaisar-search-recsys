"""Frozen-model time/user sensitivity and grouped permutation diagnostics.

No retraining or serving changes. Constructed day-level list metrics retain
historical-exposure bias. Perturbation importance is not causal importance.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from src.ctr_features import CACHE, FEATURES, FeatureState, connection
from src.v2.cascade import observed_positive_metrics
from src.v2.ranking_diagnostics import rank_pool
from src.v2.ranking_root_cause import weighted_exposure_metrics
from src.v2.train_rankers import Ranker, probability_metrics, sha


GROUPS = {
    'user_context': [1, 5, 11],
    'item_statistics': [2, 6, 10],
    'category_and_user_category': [3, 4, 7, 8, 9],
    'time': [12, 13, 14, 15],
    'search': [16, 17, 18, 19],
}


def derangement(users, seed=2026):
    users = np.asarray(users, dtype=np.int64)
    if len(users) < 2 or len(set(users.tolist())) != len(users):
        raise ValueError('Need at least two distinct users')
    order = np.random.default_rng(seed).permutation(users)
    return dict(zip(order.tolist(), np.roll(order, 1).tolist()))


def permute_group(x, columns, seed):
    """Keep a group's columns together, leaving all other columns untouched."""
    result = np.array(x, copy=True)
    permutation = np.random.default_rng(seed).permutation(len(x))
    result[:, columns] = np.asarray(x)[permutation][:, columns]
    return result


def paired_interval(delta, seed=2026, repeats=1000):
    delta = np.asarray(delta, dtype=float)
    if not len(delta):
        return {'users': 0, 'mean_delta': None, 'exploratory_95pct_user_bootstrap': None}
    rng = np.random.default_rng(seed)
    bootstrap = [float(delta[rng.integers(len(delta), size=len(delta))].mean())
                 for _ in range(repeats)]
    return {'users': len(delta), 'mean_delta': float(delta.mean()),
            'exploratory_95pct_user_bootstrap': np.percentile(bootstrap, [2.5, 97.5]).tolist()}


def diagnose(pool, output):
    if output.exists():
        raise FileExistsError('Choose a new sensitivity output directory')
    saved = json.loads((pool / 'metrics.json').read_text(encoding='utf-8'))
    if saved['split'] != 'valid' or saved['test_used']:
        raise ValueError('Only validation archives are allowed')
    if sha(pool / 'candidates.jsonl') != saved['candidate_sha256']:
        raise ValueError('Frozen candidate hash mismatch')
    torch.set_num_threads(4)
    model = Ranker(saved['config']['fine_model'])
    if sha(saved['config']['fine_model']) != saved['fine_sha256']:
        raise ValueError('Model differs from frozen comparison')
    if sha(CACHE / 'state.npz') != model.metadata['feature_state_sha256']:
        raise ValueError('Feature state differs from model')
    state = FeatureState.load(CACHE / 'state.npz')
    swap = derangement(saved['users'])
    xv = np.load(CACHE / 'valid_X.npy', mmap_mode='r')
    yv = np.load(CACHE / 'valid_y.npy')
    uv = np.load(CACHE / 'valid_users.npy')
    iv = np.load(CACHE / 'valid_items.npy')
    tv = np.load(CACHE / 'valid_times.npy')
    # Same uniformly sampled observed exposures for all feature perturbations.
    rng = np.random.default_rng(2026)
    sample = np.sort(rng.choice(len(yv), min(100000, len(yv)), replace=False))
    x, y = np.asarray(xv[sample]), yv[sample]
    reference = probability_metrics(y, model.predict(x))
    permutation_report = {}
    for group, columns in GROUPS.items():
        results = []
        for repeat in range(5):
            result = probability_metrics(y, model.predict(permute_group(x, columns, 2026 + repeat)))
            results.append({'seed': 2026 + repeat, **result})
        permutation_report[group] = {
            'columns': [FEATURES[c] for c in columns], 'runs': results,
            'mean_auc_drop': reference['auc'] - float(np.mean([v['auc'] for v in results])),
            'mean_logloss_increase': float(np.mean([v['logloss'] for v in results])) - reference['logloss']}

    # A coherent user replacement: recompute *all* user-dependent features from
    # another real user's history, rather than independently shuffling counters.
    swapped_exposure = []
    swapped_midnight_exposure = []
    for lo in range(0, len(yv), 16384):
        sl = slice(lo, lo + 16384)
        other = np.array([swap[int(u)] for u in uv[sl]])
        swapped_exposure.append(model.predict(state.features(other, iv[sl], tv[sl])))
        swapped_midnight_exposure.append(model.predict(state.features(
            other, iv[sl], np.full(len(other), saved['snapshot_ms']))))
    swapped_exposure = probability_metrics(yv, np.concatenate(swapped_exposure))
    swapped_midnight_exposure = probability_metrics(yv, np.concatenate(swapped_midnight_exposure))

    db = connection()
    positives, exposures = defaultdict(Counter), defaultdict(Counter)
    try:
        for uid, iid, label in db.execute(
                "SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
            exposures[uid][iid] += 1
            if label:
                positives[uid][iid] += 1
    finally:
        db.close()
    names = ['hour_00', 'hour_06', 'hour_12', 'hour_18', 'other_user_hour_00']
    aggregate = {name: {'hits': 0, 'ndcg': [], 'overlap': [], 'same_order': 0,
                        'matched_p': []} for name in names}
    matched_n, matched_y = [], []
    records = []
    for line in (pool / 'candidates.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        uid = row['user_id']
        items = np.asarray(row['items'], dtype=np.int64)
        x = state.features(np.full(len(items), uid), items,
                           np.full(len(items), saved['snapshot_ms']))
        baseline = None
        n = np.array([exposures[uid][int(i)] for i in items])
        y = np.array([positives[uid][int(i)] for i in items])
        known = n > 0
        matched_n.extend(n[known].tolist())
        matched_y.extend(y[known].tolist())
        record = {'user_id': uid, 'methods': {}}
        for name in names:
            if name == 'other_user_hour_00':
                values = state.features(np.full(len(items), swap[uid]), items,
                                        np.full(len(items), saved['snapshot_ms']))
            elif name == 'hour_00':
                values = x
            else:
                hour = int(name[-2:])
                values = state.features(np.full(len(items), uid), items,
                                        np.full(len(items), saved['snapshot_ms'] + hour * 3600000))
            p = model.predict(values)
            order = rank_pool(items.tolist(), p)[:20]
            if baseline is None:
                baseline = order
            quality = observed_positive_metrics(order, positives[uid])
            record['methods'][name] = quality
            a = aggregate[name]
            a['hits'] += quality['hit_events']
            if quality['user_period_ndcg_at_20'] is not None:
                a['ndcg'].append(quality['user_period_ndcg_at_20'])
            a['overlap'].append(len(set(order) & set(baseline)) / max(1, len(baseline)))
            a['same_order'] += int(order == baseline)
            a['matched_p'].extend(p[known].tolist())
        records.append(record)
        if len(records) % 1000 == 0:
            print(f'Sensitivity audit {len(records)}/{len(saved["users"])} users', flush=True)
    total_positive = sum(sum(positives[u].values()) for u in saved['users'])
    result = {}
    for name, a in aggregate.items():
        deltas = [r['methods'][name]['user_period_ndcg_at_20'] -
                  r['methods']['hour_00']['user_period_ndcg_at_20'] for r in records
                  if r['methods']['hour_00']['user_period_ndcg_at_20'] is not None]
        result[name] = {
            'hit_events': a['hits'], 'event_recall_at20': a['hits'] / total_positive,
            'ndcg_at20_positive_users': float(np.mean(a['ndcg'])),
            'top20_overlap_fraction_vs_midnight': float(np.mean(a['overlap'])),
            'identical_top20_order_user_fraction': a['same_order'] / len(records),
            'paired_ndcg_difference_vs_midnight': paired_interval(deltas),
            'matched_logged_exposure_metrics': weighted_exposure_metrics(a['matched_p'], matched_n, matched_y)}
    report = {'test_used': False, 'diagnosis_only': True,
              'users': len(records), 'positive_events': total_positive,
              'candidate_sha256': saved['candidate_sha256'], 'fine_sha256': saved['fine_sha256'],
              'source_sha256': sha(__file__), 'snapshot_ms': saved['snapshot_ms'],
              'time_list_controls': result,
              'exposure_feature_permutation': {'seed': 2026, 'sample_rows': len(sample),
                  'baseline': reference, 'groups': permutation_report},
              'coherent_user_swap_exposure_metrics': {
                  'actual_logged_time': swapped_exposure, 'midnight': swapped_midnight_exposure},
              'warnings': [
                  'Permutation breaks feature dependencies; it is sensitivity, not causal attribution.',
                  'Future hours keep counters and candidates frozen; same all-day labels, not request-level outcomes.',
                  'User replacement tests identity sensitivity on original pools, not a new recommendation policy.',
                  'Bootstrap intervals are exploratory; no best hour or parameter is selected.',
                  'No weights, source features, business configuration or final test labels changed.']}
    output.mkdir(parents=True)
    (output / 'per_user.json').write_text(json.dumps(records), encoding='utf-8')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'time_list_controls': result,
                      'coherent_user_swap_exposure_metrics': report['coherent_user_swap_exposure_metrics'],
                      'permutation_auc_drop': {k: v['mean_auc_drop'] for k, v in permutation_report.items()}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.pool, args.out)


if __name__ == '__main__':
    main()
