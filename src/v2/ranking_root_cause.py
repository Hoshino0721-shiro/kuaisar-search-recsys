"""Read-only model/data diagnosis on archived validation pools, not a fix.

Only actual validation exposure rows receive binary labels. Non-overlapping
candidates are never treated as negatives. All interventions below are offline
diagnostic comparisons and do not change production configuration or weights.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.ctr_features import CACHE, FEATURES, FeatureState, connection
from src.v2.features import coarse_features
from src.v2.models import COARSE_COLUMNS
from src.v2.ranking_diagnostics import rank_pool
from src.v2.cascade import observed_positive_metrics
from src.v2.train_rankers import Ranker, probability_metrics, sha


def weighted_exposure_metrics(scores, exposures, clicks):
    """Pair predictions weighted by *logged* positive and negative event counts."""
    scores = np.asarray(scores, dtype=float)
    exposures = np.asarray(exposures, dtype=np.int64)
    clicks = np.asarray(clicks, dtype=np.int64)
    if not (scores.shape == exposures.shape == clicks.shape):
        raise ValueError('Pair arrays must align')
    if np.any(clicks < 0) or np.any(clicks > exposures) or np.any(exposures < 0):
        raise ValueError('Invalid logged exposure counts')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Expected finite probabilities')
    n = int(exposures.sum())
    positive = int(clicks.sum())
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    if not n:
        return {'rows': 0, 'clicks': 0, 'auc': None, 'logloss': None, 'observed_ctr': None}
    auc = None
    if 0 < positive < n:
        weights = np.r_[clicks, exposures - clicks]
        labels = np.r_[np.ones(len(scores)), np.zeros(len(scores))]
        predictions = np.r_[scores, scores]
        keep = weights > 0
        auc = float(roc_auc_score(labels[keep], predictions[keep], sample_weight=weights[keep]))
    return {'rows': n, 'clicks': positive, 'auc': auc,
            'logloss': float(-(clicks * np.log(clipped) +
                               (exposures - clicks) * np.log1p(-clipped)).sum() / n),
            'observed_ctr': positive / n}


def feature_summary(x, mean, std):
    x = np.asarray(x)
    z = np.abs((x - mean) / std)
    return {name: {'mean': float(x[:, j].mean()),
                   'p50': float(np.median(x[:, j])),
                   'p99': float(np.percentile(x[:, j], 99)),
                   'abs_z_gt5_fraction': float(np.mean(z[:, j] > 5))}
            for j, name in enumerate(FEATURES)}


def diagnose(pool_dir, output):
    if output.exists():
        raise FileExistsError('Choose a new diagnostic output directory')
    original = json.loads((pool_dir / 'metrics.json').read_text(encoding='utf-8'))
    if original['split'] != 'valid' or original['test_used']:
        raise ValueError('Only validation pools are allowed')
    pool_path = pool_dir / 'candidates.jsonl'
    if sha(pool_path) != original['candidate_sha256']:
        raise ValueError('Archived candidates changed')
    torch.set_num_threads(4)
    state = FeatureState.load(CACHE / 'state.npz')
    coarse = Ranker(original['config']['coarse_model'])
    fine = Ranker(original['config']['fine_model'])
    for name, model in [('coarse', coarse), ('fine', fine)]:
        if sha(original['config'][name + '_model']) != original[name + '_sha256']:
            raise ValueError('Model changed since candidate diagnosis')
        if model.metadata['feature_state_sha256'] != sha(CACHE / 'state.npz'):
            raise ValueError('Model and feature states differ')

    # Check the complete validation cache, not just a small random fixture.
    xv = np.load(CACHE / 'valid_X.npy', mmap_mode='r')
    yv = np.load(CACHE / 'valid_y.npy')
    uv = np.load(CACHE / 'valid_users.npy')
    iv = np.load(CACHE / 'valid_items.npy')
    tv = np.load(CACHE / 'valid_times.npy')
    cached_p = fine.predict(xv)
    snapshot_p = []
    max_feature_error = 0.
    max_coarse_error = 0.
    max_prediction_error = 0.
    for lo in range(0, len(yv), 16384):
        sl = slice(lo, lo + 16384)
        fresh = state.features(uv[sl], iv[sl], tv[sl])
        max_feature_error = max(max_feature_error, float(np.max(np.abs(fresh - xv[sl]))))
        cheap = coarse_features(state, uv[sl], iv[sl])
        max_coarse_error = max(max_coarse_error, float(np.max(np.abs(cheap - fresh[:, COARSE_COLUMNS]))))
        max_prediction_error = max(max_prediction_error,
                                   float(np.max(np.abs(fine.predict(fresh) - cached_p[sl]))))
        frozen = state.features(uv[sl], iv[sl], np.full(len(fresh), original['snapshot_ms']))
        snapshot_p.append(fine.predict(frozen))
    snapshot_p = np.concatenate(snapshot_p)

    db = connection()
    exposure = defaultdict(Counter)
    positive = defaultdict(Counter)
    recent_counts = Counter()
    try:
        aligned = np.asarray(db.execute(
            "SELECT user_id,item_id,timestamp_ms,label FROM clean_events "
            "WHERE scene='rec' AND split='valid' ORDER BY id").fetchall())
        cache_alignment = {name: bool(np.array_equal(aligned[:, j], values))
                           for j, (name, values) in enumerate(
                               [('users', uv), ('items', iv), ('times', tv), ('labels', yv)])}
        if not all(cache_alignment.values()):
            raise ValueError('Database/cache alignment failed')
        del aligned
        for u, i, label in db.execute(
                "SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
            exposure[u][i] += 1
            if label:
                positive[u][i] += 1
        for i, n in db.execute(
                "SELECT item_id,count(*) FROM clean_events WHERE scene='rec' AND split='train' "
                "AND label=1 AND timestamp_ms>=? GROUP BY item_id",
                (original['snapshot_ms'] - 86400000,)):
            recent_counts[i] = n
    finally:
        db.close()
    if sum(sum(v.values()) for v in exposure.values()) != len(yv):
        raise ValueError('Database and validation cache row counts differ')
    methods = ('rrf', 'recent_count', 'train_click_count', 'item_ctr_only', 'lr', 'dcn')
    selections = {name: {'rows': 0, 'clicks': 0, 'pairs': 0, 'overlap_pairs': 0,
                         'hits': 0, 'ndcg': [], 'train_n': [], 'train_p': [],
                         'recent_p': [], 'predicted': [], 'ctr': []} for name in methods}
    matched = {'lr': [], 'dcn': [], 'item_ctr_only': []}
    matched_n, matched_y = [], []
    candidate_samples = []
    rng = np.random.default_rng(2026)
    user_agreement = 0
    total_users = 0
    logged_only = {group: {name: [] for name in methods}
                   for group in ('all_positive_users', 'at_least_40_logged_pairs')}
    for line in pool_path.open(encoding='utf-8'):
        row = json.loads(line)
        uid, ids, timestamp = row['user_id'], row['items'], row['timestamp_ms']
        if timestamp != original['snapshot_ms']:
            raise ValueError('Pool timestamp differs from diagnostic contract')
        user = np.full(len(ids), uid)
        item = np.asarray(ids, dtype=np.int64)
        x = state.features(user, item, np.full(len(ids), timestamp))
        cheap = coarse_features(state, user, item)
        pc = coarse.predict_selected(cheap)
        pf = fine.predict(x)
        cscore = np.array([recent_counts[i] for i in ids], dtype=float)
        scores = {'rrf': -np.arange(len(ids)), 'recent_count': cscore,
                  'train_click_count': state.ip[item], 'item_ctr_only': x[:, 2],
                  'lr': pc, 'dcn': pf}
        n = np.array([exposure[uid][i] for i in ids])
        y = np.array([positive[uid][i] for i in ids])
        overlap = n > 0
        matched_n.extend(n[overlap].tolist())
        matched_y.extend(y[overlap].tolist())
        for name in matched:
            matched[name].extend(scores[name][overlap].tolist())
        # A second task: rank *only* candidates with observed exposure records.
        # It is still a constructed user-period list, not a recovered request.
        overlap_ids = item[overlap].tolist()
        overlap_positive = {i: positive[uid][i] for i in overlap_ids}
        for name in methods:
            order = rank_pool(overlap_ids, scores[name][overlap])[:20]
            q = observed_positive_metrics(order, overlap_positive)['user_period_ndcg_at_20']
            if q is not None:
                logged_only['all_positive_users'][name].append(q)
                if len(overlap_ids) >= 40:
                    logged_only['at_least_40_logged_pairs'][name].append(q)
        if len(x):
            sample = rng.choice(len(x), min(16, len(x)), replace=False)
            candidate_samples.append(x[sample])
        index = {iid: j for j, iid in enumerate(ids)}
        orders = {}
        for name in methods:
            order = (ids if name == 'rrf' else rank_pool(ids, scores[name]))[:20]
            orders[name] = order
            ix = [index[i] for i in order]
            selected = selections[name]
            selected['pairs'] += len(order)
            selected['rows'] += int(n[ix].sum())
            selected['clicks'] += int(y[ix].sum())
            selected['overlap_pairs'] += int((n[ix] > 0).sum())
            quality = observed_positive_metrics(order, positive[uid])
            selected['hits'] += quality['hit_events']
            if quality['user_period_ndcg_at_20'] is not None:
                selected['ndcg'].append(quality['user_period_ndcg_at_20'])
            selected['train_n'].extend(state.inum[np.asarray(order, dtype=np.int64)].tolist())
            selected['train_p'].extend(state.ip[np.asarray(order, dtype=np.int64)].tolist())
            selected['recent_p'].extend(cscore[ix].tolist())
            selected['predicted'].extend(pf[ix].tolist())
            selected['ctr'].extend(x[ix, 2].tolist())
        user_agreement += len(set(orders['dcn']) & set(orders['recent_count']))
        total_users += 1
        if total_users % 1000 == 0:
            print(f'Root-cause audit {total_users} users', flush=True)

    total_positive = int(yv.sum())
    selection_report = {}
    for name, value in selections.items():
        selection_report[name] = {
            'selected_pairs': value['pairs'], 'logged_overlap_pairs': value['overlap_pairs'],
            'logged_overlap_pair_fraction': value['overlap_pairs'] / value['pairs'],
            'validation_exposure_rows_on_selected_pairs': value['rows'],
            'validation_clicks_on_selected_pairs': value['clicks'],
            'observed_ctr_on_overlapping_logs': value['clicks'] / value['rows'] if value['rows'] else None,
            'event_recall_at20': value['hits'] / total_positive,
            'ndcg_at20_positive_users': float(np.mean(value['ndcg'])),
            'item_train_exposures_median': float(np.median(value['train_n'])),
            'item_train_exposures_lt20_fraction': float(np.mean(np.asarray(value['train_n']) < 20)),
            'item_train_clicks_median': float(np.median(value['train_p'])),
            'item_recent_clicks_median': float(np.median(value['recent_p'])),
            'mean_smoothed_item_ctr': float(np.mean(value['ctr'])),
            'mean_predicted_dcn_ctr': float(np.mean(value['predicted']))}
    xt = np.load(CACHE / 'train_X.npy', mmap_mode='r')
    train_sample = xt[rng.choice(len(xt), min(100000, len(xt)), replace=False)]
    valid_sample = xv[rng.choice(len(xv), min(100000, len(xv)), replace=False)]
    # Audit normalization used by the actual 20-column fine model.
    distributions = {name: feature_summary(x, fine.mean, fine.std) for name, x in (
        ('train_sample', train_sample), ('valid_exposure_sample', valid_sample),
        ('uniform_per_user_candidate_sample', np.concatenate(candidate_samples)))}
    report = {
        'test_used': False, 'diagnosis_only': True, 'users': total_users,
        'candidate_sha256': original['candidate_sha256'],
        'coarse_sha256': original['coarse_sha256'], 'fine_sha256': original['fine_sha256'],
        'diagnostic_source_sha256': sha(__file__),
        'checks': {'valid_rows': len(yv), 'fresh_vs_cached_feature_max_abs_error': max_feature_error,
                   'database_cache_alignment': cache_alignment,
                   'cheap_vs_full_feature_max_abs_error': max_coarse_error,
                   'fresh_vs_cached_prediction_max_abs_error': max_prediction_error,
                   'model_feature_state_hashes_match': True},
        'exposure_task': {'dcn_actual_logged_time': probability_metrics(yv, cached_p),
                          'dcn_fixed_midnight': probability_metrics(yv, snapshot_p),
                          'single_feature_auc_controls': {
                              name: probability_metrics(yv, xv[:, column])['auc']
                              for name, column in [('user_ctr', 1), ('item_ctr', 2),
                                                   ('user_category_ctr', 4)]}},
        'candidate_overlapping_logged_exposures': {
            name: weighted_exposure_metrics(values, matched_n, matched_y)
            for name, values in matched.items()},
        'top20_selection': selection_report,
        'logged_only_candidate_ranking': {
            group: {name: {'positive_users': len(values),
                           'mean_ndcg_at20': float(np.mean(values)) if values else None}
                    for name, values in comparisons.items()}
            for group, comparisons in logged_only.items()},
        'coarse_standardized_coefficients': dict(zip(
            coarse.metadata['feature_names'],
            coarse.model.linear.weight.detach().numpy().ravel().astype(float).tolist())),
        'dcn_recent_top20_mean_overlap': user_agreement / total_users,
        'feature_distributions': distributions,
        'warnings': [
            'Candidate non-overlap means no observed label, not irrelevance.',
            'Selected-log CTR is conditional on historical exposure, not online policy CTR.',
            'One fixed user-period list measures logged-click mass, not request-level ranking.',
            'Feature distributions describe shift, not a causal attribution.',
            'No model retraining, threshold tuning, or deployment changes were performed.']}
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'feature_distributions'}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.pool, args.out)


if __name__ == '__main__':
    main()
