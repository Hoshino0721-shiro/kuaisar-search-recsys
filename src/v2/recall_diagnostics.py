"""Validation-only route attribution. No ranking models or test labels are scored."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from src.ctr_features import connection
from src.v2.recommend import Backend
from src.v2.train_rankers import sha


def fuse(routes, weights, limit, timestamp):
    ranks = {}
    for c in routes:
        if c.available_at_ms <= timestamp:
            key = (c.item_id, c.source)
            ranks[key] = min(ranks.get(key, c.rank), c.rank)
    scores = defaultdict(float)
    for (iid, source), rank in ranks.items():
        weight = weights.get(source, 1.)
        if weight > 0:
            scores[iid] += weight / (60 + rank)
    return sorted(scores, key=lambda iid: (-scores[iid], iid))[:limit]


def hits(items, positive):
    return sum(positive.get(i, 0) for i in set(items))


def evaluate(config, output, max_users):
    if output.exists():
        raise FileExistsError('Choose a new report path')
    if max_users < 0:
        raise ValueError('max-users must be nonnegative')
    started = perf_counter()
    backend = Backend(config)
    db = connection()
    positives = defaultdict(Counter)
    users = set()
    for u, i, y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
        users.add(u)
        if y:
            positives[u][i] += 1
    users = sorted(users)
    if max_users and len(users) > max_users:
        users = sorted(np.random.default_rng(2026).choice(users, max_users, replace=False).tolist())
    budget = backend.budget.recall
    # Equal-budget nonpersonalized comparators, extending beyond the current per-route limit.
    popular = sorted(np.flatnonzero(backend.state.ip > 0).tolist(),
                     key=lambda i: (-backend.state.ip[i], i))[:budget]
    recent = [r[0] for r in db.execute(
        "SELECT item_id,count(*) n FROM clean_events WHERE scene='rec' AND split='train' "
        "AND label=1 AND timestamp_ms>=? GROUP BY item_id ORDER BY n DESC,item_id LIMIT ?",
        (backend.snapshot - 86400000, budget))]
    db.close()
    catalog = set(backend.items.tolist())
    names = ('popular', 'recent', 'itemcf')
    cuts = sorted({20, 100, 400, backend.route_k, budget})
    groups = defaultdict(lambda: {'users': 0, 'positive_events': 0, 'catalog_positive_events': 0})
    measures = defaultdict(lambda: {'hits': Counter(), 'candidates': 0})
    exclusive = Counter()
    overlap = Counter()
    union_hits = fused_hits = union_size = fusion_lost = 0
    for index, uid in enumerate(users, 1):
        positive = positives[uid]
        group = 'has_train_clicks' if uid in backend.user_map else 'no_train_clicks'
        for g in ('all', group):
            groups[g]['users'] += 1
            groups[g]['positive_events'] += sum(positive.values())
            groups[g]['catalog_positive_events'] += hits(catalog.intersection(positive), positive)
        routes = backend.retrieve(uid)
        lists = {name: [c.item_id for c in routes if c.source == name] for name in names}
        pools = {name: set(lists[name]) for name in names}
        union = set().union(*pools.values())
        fused = fuse(routes, config.get('route_weights', {}), budget, backend.snapshot)
        variants = {**lists, 'fused': fused, 'popular_equal_budget': popular, 'recent_equal_budget': recent}
        for name in names:
            variants['without_' + name] = fuse([c for c in routes if c.source != name],
                                              config.get('route_weights', {}), budget, backend.snapshot)
            others = set().union(*(pools[n] for n in names if n != name))
            exclusive[name] += hits(pools[name] - others, positive)
        for a, b in combinations(names, 2):
            joined = pools[a] | pools[b]
            overlap[a + '__' + b] += len(pools[a] & pools[b]) / len(joined) if joined else 0.
        for name, ids in variants.items():
            for g in ('all', group):
                record = measures[g + '/' + name]
                record['candidates'] += len(ids)
                for k in cuts:
                    record['hits'][k] += hits(ids[:k], positive)
        uh = hits(union, positive)
        fh = hits(fused, positive)
        union_hits += uh
        fused_hits += fh
        union_size += len(union)
        fusion_lost += uh - fh
        if index % 1000 == 0:
            print(f'Diagnosed {index}/{len(users)} validation users', flush=True)
    for group in groups.values():
        total = group['positive_events']
        group['catalog_coverage_ceiling'] = group['catalog_positive_events'] / total if total else None
    metrics = {}
    for key, record in measures.items():
        group = groups[key.split('/')[0]]
        total = group['positive_events']
        reachable = group['catalog_positive_events']
        metrics[key] = {'mean_candidates': record['candidates'] / group['users'],
                        'at_k': {str(k): {'hit_events': record['hits'][k],
                                         'event_recall': record['hits'][k] / total if total else None,
                                         'catalog_conditional_recall': record['hits'][k] / reachable if reachable else None}
                                 for k in cuts}}
    total = groups['all']['positive_events']
    report = {'split': 'valid', 'test_used': False, 'users': len(users), 'seed': 2026,
              'scope': 'fixed train-cutoff lists; event-weighted observed-positive coverage, not request-level CTR',
              'config': config, 'groups': dict(groups), 'metrics': metrics,
              'union_before_budget': {'mean_candidates': union_size / len(users), 'hit_events': union_hits,
                                      'event_recall': union_hits / total if total else None},
              'fusion_budget_lost_hit_events': fusion_lost,
              'exclusive_hit_events_in_current_route_pools': dict(exclusive),
              'leave_one_out_fused_hit_delta': {
                  name: fused_hits - measures['all/without_' + name]['hits'][budget] for name in names},
              'mean_route_jaccard': {key: value / len(users) for key, value in overlap.items()},
              'notes': ['Route lists have at most route_k items; values beyond route_k plateau.',
                        'Equal-budget popularity comparators independently retrieve up to the fusion budget.',
                        'Exclusive hits concern pre-fusion pools; leave-one-out includes budget reallocation.',
                        'No-train-click users receive an empty ItemCF route, not a duplicated fallback.',
                        'Backend initializes ranking artifacts for compatibility but diagnostics never scores them.'],
              'source_sha256': sha(Path(__file__)), 'elapsed_seconds': perf_counter() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/cascade_v2.json'))
    parser.add_argument('--max-users', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.config.read_text(encoding='utf-8')), args.out, args.max_users)
    print(json.dumps({k: result[k] for k in ('users', 'groups', 'union_before_budget',
          'fusion_budget_lost_hit_events', 'exclusive_hit_events_in_current_route_pools',
          'leave_one_out_fused_hit_delta', 'elapsed_seconds')}, indent=2))
