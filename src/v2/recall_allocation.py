"""Recall allocation development experiments, no ranker loading or training.

All policies are declared before evaluation. A seeded user partition selects a
policy on one part and checks it on the other. Both belong to the repeatedly
used validation day: the check is NOT a new independent temporal holdout.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import numpy as np
from src.itemcf_index import DATA, ItemCFIndex, digest
from src.itemcf_evaluate import clustered_interval
from src.v2.recall_fusion import route_limits, select_recall, validate_policy


def rank_map(lists, limits):
    return {(int(i), source): rank for source, ids in lists.items()
            for rank, i in enumerate(ids[:limits[source]], 1)}


def load_population(data, cutoff, limit):
    db = sqlite3.connect((data / 'dataset.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    try:
        users, positives = set(), defaultdict(Counter)
        for u, i, y in db.execute(
                "SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
            users.add(u)
            if y:
                positives[u][i] += 1
        sql = ("SELECT item_id,count(*) n FROM clean_events WHERE scene='rec' AND split='train' "
               "AND label=1 AND timestamp_ms<? {extra} GROUP BY item_id ORDER BY n DESC,item_id LIMIT ?")
        popular = [r[0] for r in db.execute(sql.format(extra=''), (cutoff, limit))]
        recent = [r[0] for r in db.execute(sql.format(extra='AND timestamp_ms>=?'),
                                          (cutoff, cutoff - 86400000, limit))]
        return sorted(users), positives, popular, recent
    finally:
        db.close()


def evaluate(config, data, output):
    if output.exists():
        raise FileExistsError('Refusing to overwrite an allocation experiment')
    variants = config['variants']
    if not variants or 'recent_only' not in variants or 'legacy_rrf' not in variants:
        raise ValueError('Require recent_only and legacy_rrf controls')
    budget = config['budget']
    limits = {name: route_limits({'route_limits': spec['route_limits']}) for name, spec in variants.items()}
    for spec in variants.values():
        validate_policy(spec.get('fusion'), budget)
    fraction = config.get('selection_user_fraction', .6)
    if not 0 < fraction < 1:
        raise ValueError('Require a nonempty selection and check user partition')
    start = perf_counter()
    manifest_sha = digest(data / 'manifest.json')
    index = ItemCFIndex(config['itemcf_index'], expected_manifest_sha256=manifest_sha)
    try:
        cutoff = index.metadata['cutoff_ms']
        manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
        if cutoff != manifest['cutoffs_ms'][0]:
            raise ValueError('ItemCF cutoff differs from training cutoff')
        hot_limit = max(budget, *(v[s] for v in limits.values() for s in ('popular', 'recent')))
        users, positive, popular, recent = load_population(data, cutoff, hot_limit)
        if len(users) < 2:
            raise ValueError('Need at least two validation users')
        pos_counts = np.asarray([sum(positive[u].values()) for u in users], dtype=np.int64)
        if not pos_counts.sum():
            raise ValueError('Validation population has no positive events')
        rng = np.random.default_rng(config.get('seed', 2026))
        selection = np.zeros(len(users), dtype=bool)
        selection[rng.permutation(len(users))[:max(1, min(len(users)-1, int(len(users)*fraction)))]] = True
        warm = np.asarray([index.has_user(u) for u in users])
        masks = {'all': np.ones(len(users), dtype=bool), 'selection': selection, 'check': ~selection,
                 'has_train_clicks': warm, 'no_train_clicks': ~warm}
        catalog = set(index.items.tolist())
        reachable = np.asarray([sum(n for i, n in positive[u].items() if i in catalog) for u in users])
        hit_vectors = {name: np.zeros(len(users), dtype=np.int32) for name in variants}
        gains = {name: np.zeros(len(users), dtype=np.int32) for name in variants}
        losses = {name: np.zeros(len(users), dtype=np.int32) for name in variants}
        sizes = {name: np.zeros(len(users), dtype=np.int32) for name in variants}
        fusion_ms = defaultdict(list)
        query_ms = []
        max_cf = max(v['itemcf'] for v in limits.values())
        # Per-user hit vectors support totals, partition, and contrast auditing.
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        for row, uid in enumerate(users):
            tick = perf_counter()
            cf = index.query(uid, max_cf, neighbor_limit=config.get('itemcf_neighbor_limit'))['items'] if max_cf else []
            query_ms.append((perf_counter()-tick)*1000)
            lists = {'popular': popular, 'recent': recent, 'itemcf': cf}
            p = positive[uid]
            reference = set(recent[:budget])
            for name, spec in variants.items():
                tick = perf_counter()
                ranks = rank_map(lists, limits[name])
                selected = select_recall(ranks, budget, spec.get('weights'), spec.get('fusion'))
                fusion_ms[name].append((perf_counter()-tick)*1000)
                pool = set(selected)
                hit_vectors[name][row] = sum(p.get(i, 0) for i in pool)
                gains[name][row] = sum(p.get(i, 0) for i in pool-reference)
                losses[name][row] = sum(p.get(i, 0) for i in reference-pool)
                sizes[name][row] = len(pool)
            if (row+1) % 1000 == 0:
                print(f'Allocation validation {row+1}/{len(users)}', flush=True)
        metrics = {}
        for name in variants:
            metrics[name] = {}
            for group, mask in masks.items():
                denom = int(pos_counts[mask].sum())
                cat_denom = int(reachable[mask].sum())
                h = int(hit_vectors[name][mask].sum())
                metrics[name][group] = {
                    'users': int(mask.sum()), 'positive_events': denom, 'hit_events': h,
                    'event_recall': h/denom if denom else None,
                    'catalog_conditional_recall': h/cat_denom if cat_denom else None,
                    'mean_candidates': float(sizes[name][mask].mean()),
                    'gained_events_vs_recent': int(gains[name][mask].sum()),
                    'lost_events_vs_recent': int(losses[name][mask].sum())}
            metrics[name]['fusion_p95_ms'] = float(np.percentile(fusion_ms[name], 95))
        # Candidate budgets can be short, but never exceed budget or duplicate IDs.
        if any(np.any(n > budget) for n in sizes.values()):
            raise AssertionError('Recall budget exceeded')
        selected = max(variants, key=lambda name: (metrics[name]['selection']['hit_events'], name == 'recent_only'))
        contrasts = {}
        for other in ('recent_only', 'legacy_rrf'):
            contrasts[other] = {}
            for group in ('all', 'selection', 'check'):
                mask = masks[group]
                delta = hit_vectors[selected][mask] - hit_vectors[other][mask]
                denominator = int(pos_counts[mask].sum())
                contrasts[other][group] = {
                    'event_recall_delta': float(delta.sum()/denominator) if denominator else None,
                    'exploratory_user_bootstrap_95_interval': (
                        clustered_interval(delta, pos_counts[mask]) if denominator else None)}
        report = {
            'split': 'valid', 'test_used': False, 'scope': 'fixed training cutoff; user-period observed-positive event coverage',
            'config': config, 'manifest_sha256': manifest_sha, 'index_version': index.metadata['version'],
            'source_sha256': digest(Path(__file__)),
            'fusion_source_sha256': digest(Path(__file__).with_name('recall_fusion.py')),
            'selected_on_selection_users': selected,
            'selection_rule': 'maximum Recall@budget on seeded selection users; prefer recent_only on exact ties',
            'catalog_positive_events': int(reachable.sum()), 'metrics': metrics, 'selected_contrasts': contrasts,
            'cf_query_p95_ms': float(np.percentile(query_ms,95)), 'elapsed_seconds': perf_counter()-start,
            'notes': ['All variants declared before this run; no test labels or neural training.',
                      'Check users are an internal consistency check on a previously inspected validation day, not independent temporal evidence.',
                      'Bootstrap after model selection is exploratory; no multiple-comparison correction.',
                      'Timing includes Python allocation overhead, single process CPU; no network/concurrency.',
                      'All variants share the same item catalog and cutoff; no new items introduced.']}
        np.savez_compressed(output/'paired_hits.npz', users=np.asarray(users), positive_counts=pos_counts,
                            selection=selection, warm=warm, **hit_vectors)
        (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'selected': selected, 'metrics': {n:m['all'] for n,m in metrics.items()},
                          'contrasts':contrasts}, indent=2),flush=True)
        return report
    finally:
        index.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/recall_allocation_experiments.json'))
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    evaluate(json.loads(args.config.read_text(encoding='utf-8')),args.data,args.out)
