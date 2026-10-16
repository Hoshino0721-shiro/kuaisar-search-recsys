"""Validate bounded ItemCF indexes; raw CF, fusion contribution, paired ablations.

No neural training and no test labels. All model selection remains development.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix
from src.itemcf import rank_sparse
from src.itemcf_index import ItemCFIndex, IndexConfig, read_snapshot, graph_matrices, digest, DATA


def recall_hits(items, positive):
    pool = set(items)
    return sum(n for i, n in positive.items() if i in pool)


def merged(items, base, weights, limit=2000):
    scores = base.copy()
    for rank, iid in enumerate(items, 1):
        scores[iid] = scores.get(iid, 0.) + weights.get('itemcf', 1.) / (60 + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))[:limit]


def clustered_interval(delta, positives, seed=2026):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(1000):
        take = rng.integers(0, len(delta), len(delta))
        denom = positives[take].sum()
        if denom:
            values.append(delta[take].sum() / denom)
    return np.percentile(values, [2.5, 97.5]).tolist()


def evaluate(cosine_path, iuf_path, data, output, benchmark_users=256, repeats=3, variant_config=None,
             legacy_report_path=Path('experiments/v2_recall_diagnostics_valid_all/metrics.json')):
    if output.exists():
        raise FileExistsError('Choose a new evaluation report directory')
    if benchmark_users < 0 or repeats < 1:
        raise ValueError('Invalid benchmark size/repetitions')
    import sqlite3
    started = perf_counter()
    expected = digest(data / 'manifest.json')
    if variant_config is None:
        paths = {'cosine': cosine_path, 'iuf': iuf_path}
        variants = {'cosine_flat': ('cosine', None, None), 'iuf_flat': ('iuf', None, None),
                    'iuf_decay_0.5d': ('iuf', .5, None), 'iuf_decay_1d': ('iuf', 1., None),
                    'iuf_decay_3d': ('iuf', 3., None)}
        comparison_pairs = [('iuf_flat', 'cosine_flat'), ('iuf_decay_1d', 'iuf_flat'),
                            ('iuf_decay_0.5d', 'iuf_flat'), ('iuf_decay_3d', 'iuf_flat')]
    else:
        paths = variant_config['indexes']
        variants = {name: (spec['index'], spec.get('half_life_days'), spec.get('neighbor_limit'))
                    for name, spec in variant_config['variants'].items()}
        comparison_pairs = variant_config.get('contrasts', [])
    indexes = {name: ItemCFIndex(path, expected) for name, path in paths.items()}
    reference = next(iter(indexes.values()))
    for index in indexes.values():
        if (index.metadata['cutoff_ms'] != reference.metadata['cutoff_ms']
                or not np.array_equal(index.items, reference.items)
                or not np.array_equal(index.users, reference.users)):
            raise ValueError('Ablations must share cutoff, catalog and users')
    db = sqlite3.connect((data / 'dataset.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    positive = defaultdict(Counter)
    users = set()
    for uid, iid, label in db.execute(
            "SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
        users.add(uid)
        if label:
            positive[uid][iid] += 1
    users = sorted(users)
    cutoff = reference.metadata['cutoff_ms']
    popular = [r[0] for r in db.execute(
        "SELECT item_id,count(*) n FROM clean_events WHERE scene='rec' AND split='train' AND label=1 "
        "AND timestamp_ms<? GROUP BY item_id ORDER BY n DESC,item_id LIMIT 1000", (cutoff,))]
    recent = [r[0] for r in db.execute(
        "SELECT item_id,count(*) n FROM clean_events WHERE scene='rec' AND split='train' AND label=1 "
        "AND timestamp_ms>=? AND timestamp_ms<? GROUP BY item_id ORDER BY n DESC,item_id LIMIT 1000",
        (cutoff - 86400000, cutoff))]
    db.close()
    weights = {'popular': 1., 'recent': 2., 'itemcf': 1.}
    base = defaultdict(float)
    for name, ids in (('popular', popular), ('recent', recent)):
        for rank, iid in enumerate(ids, 1):
            base[iid] += weights[name] / (60 + rank)
    other_pool = set(popular) | set(recent)
    catalog = set(reference.items.tolist())
    group_totals = defaultdict(Counter)
    records = defaultdict(lambda: {'hits': Counter(), 'fusion_hits': Counter(), 'covered_users': 0,
                                   'candidate_sum': 0, 'exclusive_hits': 0, 'ms': [], 'edges': []})
    per_user = {name: np.zeros((len(users), 2), dtype=np.int32) for name in variants}
    positive_counts = np.asarray([sum(positive[u].values()) for u in users], dtype=np.int64)
    for num, uid in enumerate(users):
        group = 'has_train_clicks' if reference.has_user(uid) else 'no_train_clicks'
        p = positive[uid]
        for g in ('all', group):
            group_totals[g]['users'] += 1
            group_totals[g]['positive_events'] += sum(p.values())
            group_totals[g]['catalog_positive_events'] += sum(n for i, n in p.items() if i in catalog)
        for name, (which, half, width) in variants.items():
            tick = perf_counter()
            result = indexes[which].query(uid, 1000, half_life_days=half, neighbor_limit=width)
            elapsed = (perf_counter() - tick) * 1000
            items = result['items']
            fused = merged(items, base, weights)
            item_hits = {k: recall_hits(items[:k], p) for k in (100, 1000)}
            fused_hits = {k: recall_hits(fused[:k], p) for k in (100, 1000, 2000)}
            per_user[name][num] = [item_hits[100], item_hits[1000]]
            pool = set(items)
            exclusive = sum(n for i, n in p.items() if i in pool and i not in other_pool)
            for g in ('all', group):
                rec = records[g + '/' + name]
                rec['hits'].update(item_hits)
                rec['fusion_hits'].update(fused_hits)
                rec['covered_users'] += bool(items)
                rec['candidate_sum'] += len(items)
                rec['exclusive_hits'] += exclusive
                rec['ms'].append(elapsed)
                rec['edges'].append(result['edges_visited'])
        if (num + 1) % 1000 == 0:
            print(f'ItemCF ablations: {num + 1}/{len(users)} validation users', flush=True)
    metrics = {}
    for key, rec in records.items():
        g = group_totals[key.split('/')[0]]
        metrics[key] = {'at_k': {str(k): {'hit_events': rec['hits'][k],
                                        'event_recall': rec['hits'][k] / g['positive_events'] if g['positive_events'] else None,
                                        'catalog_conditional_recall': rec['hits'][k] / g['catalog_positive_events'] if g['catalog_positive_events'] else None}
                                for k in (100, 1000)},
                        'fusion_at_k': {str(k): {'hit_events': rec['fusion_hits'][k],
                                                'event_recall': rec['fusion_hits'][k] / g['positive_events'] if g['positive_events'] else None}
                                        for k in (100, 1000, 2000)},
                        'mean_candidates': rec['candidate_sum'] / g['users'],
                        'user_coverage': rec['covered_users'] / g['users'],
                        'exclusive_hit_events_vs_hot_routes': rec['exclusive_hits'],
                        'query_ms_single_pass': {str(p): float(np.percentile(rec['ms'], p)) for p in (50, 95)},
                        'max_edges_visited': max(rec['edges'])}
    contrasts = {}
    for a, b in comparison_pairs:
        contrasts[a + '_minus_' + b] = {
            str(k): {'event_recall_delta': float((per_user[a][:, j] - per_user[b][:, j]).sum() / positive_counts.sum()),
                     'user_bootstrap_95_interval': clustered_interval(per_user[a][:, j] - per_user[b][:, j], positive_counts)}
            for j, k in enumerate((100, 1000))}
    selected = max(variants, key=lambda name: (metrics['all/' + name]['at_k']['1000']['hit_events'],
                                              metrics['all/' + name]['at_k']['100']['hit_events']))
    benchmarks = {}
    if benchmark_users:
        # Same users and Top1000, warmed, repeated, alternating algorithm order.
        exact_cfg = IndexConfig(max_user_items=2147483647, user_iuf=False,
                                query_half_life_days=None)
        exact_snapshot = read_snapshot(data, exact_cfg)
        normalized, _ = graph_matrices(exact_snapshot, exact_cfg)
        transposed = normalized.T.tocsr()
        user_map = {int(u): j for j, u in enumerate(exact_snapshot.users)}
        eligible = [u for u in users if u in user_map]
        chosen = np.random.default_rng(2026).choice(eligible, min(benchmark_users, len(eligible)), replace=False).tolist()
        def old_query(uid):
            pos = user_map[uid]
            a, b = exact_snapshot.seed_offsets[pos:pos + 2]
            cols = exact_snapshot.seed_columns[a:b]
            h = csr_matrix((np.ones(len(cols)), ([0] * len(cols), cols)), shape=(1, len(exact_snapshot.items)))
            return rank_sparse(((h @ transposed) @ normalized).tocsr(), h, exact_snapshot.items, 1000)
        timings = defaultdict(list)
        for uid in chosen[:16]:
            old_query(uid)
            for which, half, width in variants.values():
                indexes[which].query(uid, 1000, half_life_days=half, neighbor_limit=width)
        query_names = ['legacy_exact'] + list(variants)
        for repeat in range(repeats):
            for uid in chosen:
                for name in (query_names if repeat % 2 == 0 else query_names[::-1]):
                    tick = perf_counter()
                    if name == 'legacy_exact':
                        old_query(uid)
                    else:
                        which, half, width = variants[name]
                        indexes[which].query(uid, 1000, half_life_days=half, neighbor_limit=width)
                    timings[name].append((perf_counter() - tick) * 1000)
        benchmarks = {'users': len(chosen), 'repeats': repeats, 'k': 1000,
                      'group': 'fixed seeded validation users with train clicks',
                      'scope': 'single-process CPU CF-only query incl history lookup/ranking; no DB/network/concurrency; warmup excluded',
                      'methods': {name: {'p50_ms': float(np.percentile(values, 50)),
                                         'p95_ms': float(np.percentile(values, 95))}
                                  for name, values in timings.items()}}
    legacy_path = legacy_report_path
    legacy = json.loads(legacy_path.read_text(encoding='utf-8')) if legacy_path and legacy_path.exists() else None
    if legacy and (legacy['users'] != len(users) or legacy['groups']['all']['positive_events'] != int(positive_counts.sum())):
        raise ValueError('Legacy report does not match validation population')
    report = {'split': 'valid', 'test_used': False, 'users': len(users), 'seed': 2026,
              'scope': 'fixed train-cutoff lists; positive-event coverage; no popularity fallback in CF route',
              'fusion_config': {'route_k': 1000, 'recall_budget': 2000, 'rrf_constant': 60, 'route_weights': weights},
              'groups': {k: dict(v) for k, v in group_totals.items()}, 'metrics': metrics,
              'indexes': {k: {'path': str(v.path), 'metadata': v.metadata, 'load_seconds': v.load_seconds}
                          for k, v in indexes.items()},
              'variants': {k: {'index': w, 'query_half_life_days': h, 'neighbor_limit': width}
                           for k, (w, h, width) in variants.items()},
              'development_selected': selected, 'selection_metric': 'full-validation raw CF event Recall@1000; Recall@100 tie-break',
              'paired_contrasts': contrasts, 'benchmark': benchmarks,
              'legacy_report': str(legacy_path) if legacy else None,
              'legacy_report_sha256': digest(legacy_path) if legacy else None,
              'legacy_raw_itemcf': legacy['metrics']['all/itemcf'] if legacy else None,
              'legacy_fusion': legacy['metrics']['all/fused'] if legacy else None,
              'manifest_sha256': expected, 'source_sha256': digest(Path(__file__)),
              'notes': ['Single development window; not an independent test or online uplift.',
                        'Use explicit per-index configs to identify changed factors; old exact comparison changes graph pruning.',
                        'Bootstrap resamples users 1000 times; exploratory intervals, no multiple-comparison correction.',
                        'Offline build and startup hash verification are excluded from query latency.'],
              'elapsed_seconds': perf_counter() - started}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'paired_hits.npz', users=np.asarray(users), positive_counts=positive_counts,
                        **per_user)
    for index in indexes.values():
        index.close()
    print(json.dumps({'selected': selected, 'raw_cf': {n: metrics['all/' + n]['at_k'] for n in variants},
                      'benchmark': benchmarks}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cosine', type=Path)
    parser.add_argument('--iuf', type=Path)
    parser.add_argument('--variants', type=Path)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--benchmark-users', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not args.variants and (args.cosine is None or args.iuf is None):
        parser.error('Provide --variants or both --cosine and --iuf')
    specs = json.loads(args.variants.read_text(encoding='utf-8')) if args.variants else None
    evaluate(args.cosine, args.iuf, args.data, args.out, args.benchmark_users, args.repeats, specs)
