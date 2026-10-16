"""CPU offline cascade with cached popularity and exact sparse ItemCF.

No online update or deployment is implied. Run from the repository root.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from scipy.sparse import csr_matrix
from src.ctr_features import CACHE, DATA, FeatureState, connection
from src.itemcf import rank_sparse
from src.v2.cascade import Budget, Candidate, Cascade, observed_positive_metrics
from src.v2.train_rankers import Ranker, sha
from src.v2.features import coarse_features
from src.itemcf_index import ItemCFIndex


class Backend:
    def __init__(self, config):
        tick = perf_counter()
        torch.set_num_threads(4)
        self.config = config
        self.budget = Budget(**config['budget'])
        self.state = FeatureState.load(CACHE / 'state.npz')
        self.snapshot = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))['cutoffs_ms'][0]
        state_hash = sha(CACHE / 'state.npz')
        self.coarse = Ranker(config['coarse_model'])
        self.fine = Ranker(config['fine_model'])
        if self.coarse.metadata['kind'] != 'coarse_lr' or self.fine.metadata['kind'] != 'fine_dcn':
            raise ValueError('Expected independent coarse LR and fine DCN artifacts')
        for model in (self.coarse, self.fine):
            if model.metadata['feature_state_sha256'] != state_hash:
                raise ValueError('Ranker and inference feature snapshots differ')
        self.route_k = config['route_k']
        if not isinstance(self.route_k, int) or self.route_k < 1:
            raise ValueError('route_k must be positive')
        db = connection()
        counts = self.state.ip
        self.popular = sorted(np.flatnonzero(counts > 0).tolist(),
                              key=lambda i: (-counts[i], i))[:self.route_k]
        self.recent = [r[0] for r in db.execute(
            "SELECT item_id,count(*) n FROM clean_events WHERE scene='rec' AND split='train' "
            "AND label=1 AND timestamp_ms>=? GROUP BY item_id ORDER BY n DESC,item_id LIMIT ?",
            (self.snapshot - 86400000, self.route_k))]
        self.itemcf_index = None
        if config.get('itemcf_index'):
            db.close()
            self.itemcf_index = ItemCFIndex(config['itemcf_index'],
                                            expected_manifest_sha256=sha(DATA / 'manifest.json'))
            if self.itemcf_index.metadata['cutoff_ms'] != self.snapshot:
                self.itemcf_index.close()
                raise ValueError('ItemCF and ranking feature cutoffs differ')
            self.items = self.itemcf_index.items
            self.user_map = {int(u): j for j, u in enumerate(self.itemcf_index.users)}
            self.startup_ms = (perf_counter() - tick) * 1000
            return
        # Build once per backend, not once for each user request.
        pairs = np.asarray(db.execute(
            "SELECT user_id,item_id FROM clean_events WHERE scene='rec' AND split='train' "
            "AND label=1 ORDER BY timestamp_ms DESC,item_id,id").fetchall(), dtype=np.int64)
        db.close()
        if not len(pairs):
            raise ValueError('No training positives')
        users, rows = np.unique(pairs[:, 0], return_inverse=True)
        self.items, columns = np.unique(pairs[:, 1], return_inverse=True)
        self.user_map = {int(u): j for j, u in enumerate(users)}
        self.item_map = {int(i): j for j, i in enumerate(self.items)}
        self.seeds = defaultdict(dict)
        for u, i in pairs:
            history = self.seeds[int(u)]
            if len(history) < 30:
                history[int(i)] = None
        binary = csr_matrix((np.ones(len(rows)), (rows, columns)),
                            shape=(len(users), len(self.items)))
        binary.data[:] = 1.0
        degree = np.asarray(binary.sum(axis=0)).ravel()
        self.normalized = binary.multiply(1 / np.sqrt(degree)).tocsr()
        self.transposed = self.normalized.T.tocsr()
        self.startup_ms = (perf_counter() - tick) * 1000

    def retrieve(self, uid, timestamp_ms=None):
        lists = {'popular': self.popular, 'recent': self.recent}
        if self.itemcf_index is not None:
            options = {}
            if 'itemcf_query_half_life_days' in self.config:
                options['half_life_days'] = self.config['itemcf_query_half_life_days']
            if 'itemcf_neighbor_limit' in self.config:
                options['neighbor_limit'] = self.config['itemcf_neighbor_limit']
            lists['itemcf'] = self.itemcf_index.query(
                uid, self.route_k, timestamp_ms, **options)['items']
        elif uid in self.user_map:
            indices = [self.item_map[i] for i in self.seeds[uid]]
            h = csr_matrix((np.ones(len(indices)), ([0] * len(indices), indices)),
                           shape=(1, len(self.items)))
            result = ((h @ self.transposed) @ self.normalized).tocsr()
            lists['itemcf'] = rank_sparse(result, h, self.items, self.route_k)
        else:
            lists['itemcf'] = []
        return [Candidate(i, source, j, self.snapshot)
                for source, items in lists.items() for j, i in enumerate(items, 1)]

    def run(self, uid, timestamp_ms=None, positives=None, budget=None):
        if not 0 <= uid < len(self.state.un):
            raise ValueError('User ID outside anonymized user table')
        timestamp_ms = self.snapshot if timestamp_ms is None else timestamp_ms
        if timestamp_ms < self.snapshot:
            raise ValueError('Request predates this training feature snapshot')
        tick = perf_counter()
        routes = self.retrieve(uid, timestamp_ms)
        authors = {}
        active_budget = budget or self.budget
        if active_budget.max_per_author is not None:
            db = connection()
            unique = sorted({c.item_id for c in routes})
            for lo in range(0, len(unique), 500):
                group = unique[lo:lo + 500]
                sql = 'SELECT item_id,author_id FROM items WHERE item_id IN (' + ','.join('?' for _ in group) + ')'
                for iid, author in db.execute(sql, group):
                    if author not in (None, '', 'NULL', 'null', 'unknown'):
                        authors[iid] = author
            db.close()
        retrieval_ms = (perf_counter() - tick) * 1000
        def features(ids, ts):
            return self.state.features(np.full(len(ids), uid), np.asarray(ids),
                                       np.full(len(ids), ts, dtype=np.int64))
        def coarse(user, ids, ts):
            cheap = coarse_features(self.state, np.full(len(ids), uid), np.asarray(ids))
            return self.coarse.predict_selected(cheap)
        def fine(user, ids, ts):
            return self.fine.predict(features(ids, ts))
        categories = {c.item_id: int(self.state.categories[c.item_id]) for c in routes}
        cascade = Cascade(coarse, fine, active_budget, categories=categories, authors=authors)
        # Repeat interactions remain valid labels, so history exclusion is disabled.
        result = cascade.run(uid, timestamp_ms, routes,
                             route_weights=self.config.get('route_weights'), positives=positives)
        result.update({'retrieval_elapsed_ms': retrieval_ms,
                       'request_elapsed_ms': retrieval_ms + result['total_elapsed_ms'],
                       'feature_snapshot_ms': self.snapshot, 'mode': 'offline_v2_cascade',
                       'historical_catalog_only': True})
        if self.itemcf_index is not None:
            result['itemcf_index_version'] = self.itemcf_index.metadata['version']
        return result

    def close(self):
        if self.itemcf_index is not None:
            self.itemcf_index.close()


def evaluate_valid(backend, output, max_users, compare_without_coarse=False):
    if output.exists():
        raise FileExistsError('Choose a new evaluation output path')
    if max_users < 0:
        raise ValueError('max-users must be nonnegative; 0 means all users')
    if backend.budget.output != 20:
        raise ValueError('Validation report compares Top20; set output budget to 20')
    db = connection()
    positives = defaultdict(Counter)
    users = set()
    for u, i, label in db.execute(
            "SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
        users.add(u)
        if label:
            positives[u][i] += 1
    db.close()
    users = sorted(users)
    if max_users and len(users) > max_users:
        users = sorted(np.random.default_rng(2026).choice(users, max_users, replace=False).tolist())
    if not users:
        raise ValueError('No validation users')
    warmup = backend.run(users[0])['request_elapsed_ms']
    if compare_without_coarse:
        backend.run(users[0], budget=replace(backend.budget, coarse_enabled=False))
    aggregate = {s: {'hits': 0, 'positives': 0, 'ms': [], 'counts': [], 'recall_at_20_hits': 0,
                     'ndcg': [], 'diversity': []}
                 for s in ('recall', 'coarse', 'fine', 'rerank')}
    baselines = {name: {'hits': 0, 'positives': 0, 'ndcg': [], 'ms': [], 'diversity': []}
                 for name in ('popular', 'recent', 'without_rerank')}
    if compare_without_coarse:
        baselines['without_coarse'] = {'hits': 0, 'positives': 0, 'ndcg': [], 'ms': [], 'diversity': []}
    request_ms = []
    def diversity(items):
        cats = backend.state.categories[np.asarray(items[:20], dtype=np.int64)]
        return len(set(cats[cats > 0].tolist())) / max(1, len(cats))
    for j, u in enumerate(users):
        result = backend.run(u, positives=positives[u])
        request_ms.append(result['request_elapsed_ms'])
        for stage in result['stages']:
            record = aggregate[stage['stage']]
            record['hits'] += stage['hit_events']
            record['positives'] += stage['positive_events']
            record['ms'].append(stage['elapsed_ms'])
            record['counts'].append(stage['output_count'])
            record['diversity'].append(diversity(stage['items']))
            if stage['user_period_ndcg_at_20'] is not None:
                record['ndcg'].append(stage['user_period_ndcg_at_20'])
            pool = set(stage['items'][:20])
            record['recall_at_20_hits'] += sum(n for i, n in positives[u].items() if i in pool)
        for name, baseline in baselines.items():
            if name == 'without_coarse':
                alternative = backend.run(u, budget=replace(backend.budget, coarse_enabled=False))
                items = alternative['items']
                baseline['ms'].append(alternative['request_elapsed_ms'])
            elif name == 'without_rerank':
                alternative = backend.run(u, budget=replace(backend.budget, category_penalty=0., max_per_author=None))
                items = alternative['items']
                baseline['ms'].append(alternative['request_elapsed_ms'])
            else:
                items = (backend.popular if name == 'popular' else backend.recent)[:20]
            baseline['diversity'].append(diversity(items))
            quality = observed_positive_metrics(items, positives[u])
            baseline['hits'] += quality['hit_events']
            baseline['positives'] += quality['positive_events']
            if quality['user_period_ndcg_at_20'] is not None:
                baseline['ndcg'].append(quality['user_period_ndcg_at_20'])
        if (j + 1) % 100 == 0:
            print(f'Evaluated {j + 1}/{len(users)} users', flush=True)
    summary = {}
    previous = None
    for name, record in aggregate.items():
        summary[name] = {'positive_events': record['positives'], 'hit_events_at_stage_budget': record['hits'],
                         'observed_event_recall_at_stage_budget': record['hits'] / record['positives'] if record['positives'] else None,
                         'positive_retention_vs_previous': record['hits'] / previous if previous else None,
                         'observed_event_recall_at_20': record['recall_at_20_hits'] / record['positives'] if record['positives'] else None,
                         'mean_candidates': float(np.mean(record['counts'])),
                         'user_period_ndcg_at_20_positive_users': float(np.mean(record['ndcg'])) if record['ndcg'] else None,
                         'category_diversity_at_20': float(np.mean(record['diversity'])),
                         'p50_ms': float(np.percentile(record['ms'], 50)),
                         'p95_ms': float(np.percentile(record['ms'], 95))}
        previous = record['hits']
    comparison = {name: {'observed_event_recall_at_20': row['hits'] / row['positives'] if row['positives'] else None,
                         'user_period_ndcg_at_20_positive_users': float(np.mean(row['ndcg'])) if row['ndcg'] else None,
                         'category_diversity_at_20': float(np.mean(row['diversity'])),
                         'request_p95_ms': float(np.percentile(row['ms'], 95)) if row['ms'] else None}
                  for name, row in baselines.items()}
    report = {'split': 'valid', 'users': len(users), 'seed': 2026, 'baselines': comparison,
              'scope': 'fixed train-cutoff lists; observed-positive events, not request-level outcomes',
              'max_users': max_users, 'test_used': False, 'config': backend.config,
              'coarse_sha256': sha(backend.config['coarse_model']),
              'fine_sha256': sha(backend.config['fine_model']), 'stages': summary,
              'request_p50_ms': float(np.percentile(request_ms, 50)),
              'request_p95_ms': float(np.percentile(request_ms, 95)),
              'warmup_request_ms_excluded': warmup,
              'startup_ms': backend.startup_ms,
              'timing_scope': 'single-process CPU, includes retrieval and feature construction; no network or concurrency'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/cascade_v2.json'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--user-id', type=int)
    group.add_argument('--evaluate-valid', action='store_true')
    parser.add_argument('--max-users', type=int, default=256)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--compare-without-coarse', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    backend = Backend(config)
    if args.evaluate_valid:
        if args.out is None:
            parser.error('--evaluate-valid requires --out')
        report = evaluate_valid(backend, args.out, args.max_users, args.compare_without_coarse)
    else:
        report = backend.run(args.user_id)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
