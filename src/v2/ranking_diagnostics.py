"""Paired ranker diagnostics on identical frozen recall pools; validation only.

Candidates without observed clicks remain unlabeled. These list metrics are
user-period proxies, not request CTR or unbiased relevance measurements.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.v2.cascade import Cascade, observed_positive_metrics, ordered_scores
from src.v2.features import coarse_features
from src.v2.recommend import Backend, connection
from src.v2.train_rankers import sha


def rank_pool(ids, scores):
    return [iid for iid, _ in ordered_scores(ids, scores)]


def compare_pool(ids, coarse_scores, fine_scores, recent_ranks, popular_ranks,
                 positives, budgets=(200, 400, 800)):
    """Pure paired comparison: only order/truncation changes, never the pool."""
    if len(set(ids)) != len(ids):
        raise ValueError('Frozen pool must contain unique items')
    coarse = rank_pool(ids, coarse_scores)
    fine = rank_pool(ids, fine_scores)
    rankings = {
        'rrf': list(ids),
        'recent_in_pool': sorted(ids, key=lambda i: (recent_ranks.get(i, float('inf')), i)),
        'popular_in_pool': sorted(ids, key=lambda i: (popular_ranks.get(i, float('inf')), i)),
        'lr_direct': coarse, 'dcn_direct': fine,
    }
    for budget in budgets:
        if not isinstance(budget, int) or budget < 20:
            raise ValueError('Coarse budgets must be integers >= 20')
        kept = set(coarse[:budget])
        rankings[f'lr{budget}_dcn'] = [i for i in fine if i in kept]
    metrics = {name: observed_positive_metrics(order[:20], positives)
               for name, order in rankings.items()}
    teacher = set(fine[:20])
    retention = {}
    for budget in budgets:
        kept = set(coarse[:budget])
        retention[str(budget)] = {
            'positive_events_retained': observed_positive_metrics(coarse[:budget], positives)['hit_events'],
            'dcn_top20_retained': len(teacher & kept),
            'dcn_top20_count': len(teacher),
        }
    # Label-informed upper bound for diagnosis only; never a serving ranker.
    oracle = sorted(ids, key=lambda i: (-int(positives.get(i, 0) > 0), i))
    metrics['oracle_binary_in_pool'] = observed_positive_metrics(oracle[:20], positives)
    return metrics, retention


def evaluate(backend, output, max_users=256):
    if output.exists():
        raise FileExistsError('Choose a new diagnostic directory')
    if max_users < 0:
        raise ValueError('max-users must be >= 0; 0 means all validation users')
    db = connection()
    positives = defaultdict(Counter)
    users = set()
    try:
        for u, i, label in db.execute(
                "SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
            users.add(u)
            if label:
                positives[u][i] += 1
    finally:
        db.close()
    users = sorted(users)
    if max_users and len(users) > max_users:
        users = sorted(np.random.default_rng(2026).choice(users, max_users, replace=False).tolist())
    if not users:
        raise ValueError('No validation users')
    output.mkdir(parents=True)
    recent = {i: j for j, i in enumerate(backend.recent)}
    popular = {i: j for j, i in enumerate(backend.popular)}
    records = []
    pool_digest = hashlib.sha256()
    timings = []
    with (output / 'candidates.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for uid in users:
            routes = backend.retrieve(uid, backend.snapshot)
            # Reuse the actual fusion/filtering code, without invoking either model.
            executor = Cascade(lambda u, ids, t: [0.] * len(ids),
                               lambda u, ids, t: [0.] * len(ids),
                               replace(backend.budget, category_penalty=0., max_per_author=None))
            trace = executor.run(uid, backend.snapshot, routes,
                                 route_weights=backend.config.get('route_weights'))
            ids = trace['stages'][0]['items']
            serialized = json.dumps({'user_id': uid, 'timestamp_ms': backend.snapshot,
                                     'items': ids}, separators=(',', ':')) + '\n'
            stream.write(serialized)
            pool_digest.update(serialized.encode('utf-8'))
            tick = perf_counter()
            user = np.full(len(ids), uid)
            items = np.asarray(ids, dtype=np.int64)
            coarse = backend.coarse.predict_selected(coarse_features(backend.state, user, items))
            fine = backend.fine.predict(backend.state.features(
                user, items, np.full(len(ids), backend.snapshot, dtype=np.int64)))
            timings.append((perf_counter() - tick) * 1000)
            metrics, retention = compare_pool(ids, coarse, fine, recent, popular, positives[uid])
            records.append({'user_id': uid, 'warm': bool(backend.state.up[uid] > 0),
                            'candidate_count': len(ids),
                            'pool': observed_positive_metrics(ids, positives[uid]),
                            'metrics': metrics, 'retention': retention})
            if len(records) % 100 == 0:
                print(f'Diagnosed {len(records)}/{len(users)} users', flush=True)
    def summarize(rows):
        total = sum(r['pool']['positive_events'] for r in rows)
        pool_hits = sum(r['pool']['hit_events'] for r in rows)
        methods = {}
        for name in records[0]['metrics']:
            values = [r['metrics'][name] for r in rows]
            ndcg = [v['user_period_ndcg_at_20'] for v in values
                    if v['user_period_ndcg_at_20'] is not None]
            hits = sum(v['hit_events'] for v in values)
            methods[name] = {'hit_events_at_20': hits,
                             'event_recall_at_20': hits / total if total else None,
                             'ndcg_at_20_positive_users': float(np.mean(ndcg)) if ndcg else None}
        retention = {}
        for budget in ('200', '400', '800'):
            retained = sum(r['retention'][budget]['positive_events_retained'] for r in rows)
            teacher = sum(r['retention'][budget]['dcn_top20_count'] for r in rows)
            retention[budget] = {
                'positive_event_retention': retained / pool_hits if pool_hits else None,
                'dcn_top20_retention': sum(r['retention'][budget]['dcn_top20_retained']
                                         for r in rows) / teacher if teacher else None}
        return {'users': len(rows), 'positive_events': total, 'pool_hit_events': pool_hits,
                'mean_candidates': float(np.mean([r['candidate_count'] for r in rows])) if rows else None,
                'methods': methods, 'coarse_retention': retention}
    report = {'split': 'valid', 'test_used': False, 'seed': 2026, 'users': users,
              'config': backend.config, 'candidate_sha256': pool_digest.hexdigest(),
              'coarse_sha256': sha(backend.config['coarse_model']),
              'fine_sha256': sha(backend.config['fine_model']),
              'snapshot_ms': backend.snapshot,
              'scope': 'fixed training-cutoff lists; observed positive coverage, not online CTR',
              'unobserved_candidates': 'unlabeled; never pseudo-negatives',
              'oracle': 'binary relevance ceiling only; not a serving method or event-recall ceiling',
              'rerank': 'disabled for every method; identical Top20 budget',
              'scoring_p95_ms': float(np.percentile(timings, 95)),
              'timing_scope': 'both rankers plus features on entire pool; not serving latency',
              'groups': {name: summarize(rows) for name, rows in (
                  ('all', records), ('warm', [r for r in records if r['warm']]),
                  ('cold', [r for r in records if not r['warm']]))}}
    (output / 'per_user.json').write_text(json.dumps(records), encoding='utf-8')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-users', type=int, default=256)
    args = parser.parse_args()
    backend = Backend(json.loads(args.config.read_text(encoding='utf-8')))
    try:
        report = evaluate(backend, args.out, args.max_users)
        print(json.dumps({'output': str(args.out), 'test_used': report['test_used'],
                          'candidate_sha256': report['candidate_sha256'],
                          'all': report['groups']['all']}, indent=2))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
