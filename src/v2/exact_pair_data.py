"""Vectorized exact-user-timestamp pairing with item dedup/conflict exclusion."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.ctr_features import CACHE, connection
from src.v2.pairwise_data import training_times
from src.v2.ranking_convergence import write_json
from src.v2.train_rankers import sha


def build_exact_pairs(users, items, times, labels, seed=2026):
    users, items, times, labels = [np.asarray(a) for a in (users, items, times, labels)]
    if not (len(users) == len(items) == len(times) == len(labels)) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid exposure arrays')
    if not len(labels):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {'rows': 0, 'pairs': 0}
    order = np.lexsort((np.arange(len(labels)), items, times, users))
    su, st, si = users[order], times[order], items[order]
    cuts = np.r_[0, np.flatnonzero((np.diff(su) != 0) | (np.diff(st) != 0) | (np.diff(si) != 0))+1]
    lengths = np.diff(np.r_[cuts, len(order)])
    minimum = np.minimum.reduceat(labels[order], cuts)
    maximum = np.maximum.reduceat(labels[order], cuts)
    consistent = minimum == maximum
    rows = order[cuts[consistent]]
    if not len(rows):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {
            'rows': len(labels), 'pairs': 0, 'contradictory_item_rows_excluded': int(lengths.sum())}
    group_starts = np.r_[0, np.flatnonzero((np.diff(users[rows]) != 0) | (np.diff(times[rows]) != 0))+1]
    groups = np.repeat(np.arange(len(group_starts)), np.diff(np.r_[group_starts, len(rows)]))
    is_positive = labels[rows] == 1
    p_count = np.bincount(groups, weights=is_positive.astype(np.int64)).astype(np.int64)
    n_count = np.bincount(groups)-p_count
    mixed = (p_count > 0) & (n_count > 0)
    positive = rows[is_positive & mixed[groups]]
    p_groups = groups[is_positive & mixed[groups]]
    all_negatives = rows[~is_positive]
    negative_offsets = np.cumsum(n_count)-n_count
    draw = np.random.default_rng(seed).integers(n_count[p_groups])
    negative = all_negatives[negative_offsets[p_groups]+draw]
    for a, b in ((users[positive], users[negative]), (times[positive], times[negative])):
        np.testing.assert_array_equal(a, b)
    if np.any(items[positive] == items[negative]) or not (np.all(labels[positive] == 1) and np.all(labels[negative] == 0)):
        raise ValueError('Invalid observed pairs')
    return positive, negative, {
        'rows': len(labels), 'pairs': len(positive), 'usable_groups': int(mixed.sum()),
        'paired_users': len(np.unique(users[positive])),
        'unique_endpoint_rows': len(np.unique(np.r_[positive, negative])),
        'contradictory_item_rows_excluded': int(lengths[~consistent].sum()),
        'consistent_duplicate_rows_collapsed': int((lengths[consistent]-1).sum()),
        'max_timestamp_gap_ms': 0, 'one_negative_per_positive': True,
        'negative_reuse_allowed': True, 'not_verified_request_ids': True}


def verify(path):
    meta = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('status') != 'ready' or meta['test_used']:
        raise ValueError('Completed non-test pair cache required')
    for name, digest in meta['numeric_files'].items():
        if Path(name).name != name or sha(CACHE/name) != digest:
            raise ValueError('Numeric input changed: '+name)
    for name, digest in meta['files'].items():
        if Path(name).name != name or sha(path/name) != digest:
            raise ValueError('Pair cache changed: '+name)
    return meta


def prepare(output, seed=2026):
    output.mkdir(parents=True, exist_ok=False)
    numeric_names = ['state.npz']+[f'{s}_{k}.npy' for s in ('train', 'valid')
                                  for k in ('users', 'items', 'y')]+['valid_times.npy']
    numeric_hashes = {name: sha(CACHE/name) for name in numeric_names}
    times = training_times()  # Full SQL row/order identity check, not an assumed sort.
    np.save(output/'train_times.npy', times)
    reports = {}
    for split in ('train', 'valid'):
        u, i, y = [np.load(CACHE/f'{split}_{key}.npy') for key in ('users', 'items', 'y')]
        t = times if split == 'train' else np.load(CACHE/'valid_times.npy')
        if split == 'valid':
            db = connection()
            try:
                rows = np.asarray(db.execute("SELECT user_id,item_id,label,timestamp_ms FROM clean_events WHERE scene='rec' AND split='valid' ORDER BY id").fetchall(), dtype=np.int64)
                for actual, expected in zip(rows.T, (u, i, y, t)):
                    np.testing.assert_array_equal(actual, expected)
            finally:
                db.close()
        p, n, report = build_exact_pairs(u, i, t, y, seed)
        if not len(p):
            raise ValueError('No usable observed exact-timestamp pairs')
        np.save(output/f'{split}_positive.npy', p)
        np.save(output/f'{split}_negative.npy', n)
        reports[split] = report
        print(split+': '+json.dumps(report), flush=True)
    if any(sha(CACHE/name) != digest for name, digest in numeric_hashes.items()):
        raise ValueError('Input changed during preparation')
    write_json(output/'metadata.json', {'status': 'ready', 'seed': seed, 'test_used': False,
               'source_sha256': sha(__file__), 'all_sql_rows_aligned': True,
               'numeric_files': numeric_hashes, 'splits': reports,
               'files': {p.name: sha(p) for p in output.glob('*.npy')}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    prepare(args.out, args.seed)
