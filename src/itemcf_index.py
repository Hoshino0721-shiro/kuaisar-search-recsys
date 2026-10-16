"""Bounded offline ItemCF index and NumPy-only request path.

Build uses recommendation TRAIN clicks only. Published directories are immutable;
querying needs no SQLite connection, torch model, or user-item matrix product.
"""
import argparse
from array import array
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import shutil
import sqlite3
import tempfile
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix

DAY_MS = 86_400_000
DATA = Path('data/processed/kuaisar_small_v1')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class IndexConfig:
    neighbors: int = 100
    max_user_items: int = 200
    seed_items: int = 30
    window_days: float | None = None
    user_iuf: bool = True
    popularity_alpha: float = 0.5
    graph_half_life_days: float | None = None
    query_half_life_days: float | None = 1.0
    min_cooccurrence: int = 1
    shrinkage: float = 0.0
    block_items: int = 256

    def __post_init__(self):
        for name in ('neighbors', 'max_user_items', 'seed_items', 'min_cooccurrence', 'block_items'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.seed_items > self.max_user_items:
            raise ValueError('seed_items cannot exceed max_user_items')
        for name in ('window_days', 'graph_half_life_days', 'query_half_life_days'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be positive or null')
        if not isinstance(self.user_iuf, bool):
            raise ValueError('user_iuf must be boolean')
        if isinstance(self.popularity_alpha, bool) or not np.isfinite(self.popularity_alpha) or not 0 <= self.popularity_alpha <= 1:
            raise ValueError('popularity_alpha must be between zero and one')
        if not np.isfinite(self.shrinkage) or self.shrinkage < 0:
            raise ValueError('shrinkage must be finite and nonnegative')


@dataclass
class Snapshot:
    items: np.ndarray
    users: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    times: np.ndarray
    user_degree: np.ndarray
    seed_offsets: np.ndarray
    seed_columns: np.ndarray
    seed_times: np.ndarray
    cutoff_ms: int
    manifest_sha256: str
    history_config: dict


def make_snapshot(rows, items, cutoff_ms, config, manifest_sha256):
    """Rows must be (user,item,time), ordered user ASC,time DESC,item ASC.

    Dedup uses latest click, not event multiplicity. Only strictly prior events
    enter either graph or stored query histories. Full window degree is measured
    before the per-user graph cap, so truncation does not erase user activity.
    """
    items = np.asarray(items, dtype=np.int64)
    if not len(items) or np.any(items < 0) or np.any(np.diff(items) <= 0):
        raise ValueError('Catalog must contain sorted unique nonnegative IDs')
    if len(items) >= np.iinfo(np.int32).max:
        raise ValueError('Catalog exceeds int32 index capacity')
    item_map = {int(i): j for j, i in enumerate(items)}
    users, degree, offsets = [], [], [0]
    rr, cc, tt, seeds, seed_times = array('i'), array('i'), array('q'), array('i'), array('q')
    current, seen, full_count, retained, seed_count = None, set(), 0, 0, 0
    lower = cutoff_ms - config.window_days * DAY_MS if config.window_days else -np.inf
    previous_key = None
    for uid, iid, timestamp in rows:
        uid, iid, timestamp = int(uid), int(iid), int(timestamp)
        if uid < 0:
            raise ValueError('Negative user ID')
        if timestamp >= cutoff_ms:
            continue
        key = (uid, -timestamp, iid)
        if previous_key is not None and key < previous_key:
            raise ValueError('Input histories are not chronologically ordered per user')
        previous_key = key
        if uid != current:
            if current is not None:
                degree.append(full_count)
                offsets.append(len(seeds))
            current, seen, full_count, retained, seed_count = uid, set(), 0, 0, 0
            users.append(uid)
        if iid in seen:
            continue
        seen.add(iid)
        if iid not in item_map:
            raise ValueError('Click outside declared training catalog')
        col = item_map[iid]
        if seed_count < config.seed_items:
            seeds.append(col)
            seed_times.append(timestamp)
            seed_count += 1
        if timestamp >= lower:
            full_count += 1
            if retained < config.max_user_items:
                rr.append(len(users) - 1)
                cc.append(col)
                tt.append(timestamp)
                retained += 1
    if current is not None:
        degree.append(full_count)
        offsets.append(len(seeds))
    if not len(rr):
        raise ValueError('No graph interactions before cutoff within configured window')
    return Snapshot(items, np.asarray(users, dtype=np.int64), np.asarray(rr, dtype=np.int32),
                    np.asarray(cc, dtype=np.int32), np.asarray(tt, dtype=np.int64),
                    np.asarray(degree, dtype=np.int32), np.asarray(offsets, dtype=np.int64),
                    np.asarray(seeds, dtype=np.int32), np.asarray(seed_times, dtype=np.int64),
                    int(cutoff_ms), manifest_sha256,
                    {k: getattr(config, k) for k in ('max_user_items', 'seed_items', 'window_days')})


def read_snapshot(data, config):
    data = Path(data)
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'ready':
        raise ValueError('Dataset manifest is not ready')
    cutoff = manifest['cutoffs_ms'][0]
    db = sqlite3.connect((data / 'dataset.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    try:
        db.execute('PRAGMA cache_size=-65536')
        items = np.fromiter((r[0] for r in db.execute(
            "SELECT DISTINCT item_id FROM clean_events WHERE scene='rec' AND split='train' "
            "AND label=1 AND timestamp_ms<? ORDER BY item_id", (cutoff,))), dtype=np.int64)
        cursor = db.execute(
            "SELECT user_id,item_id,MAX(timestamp_ms) t FROM clean_events "
            "WHERE scene='rec' AND split='train' AND label=1 AND timestamp_ms<? "
            "GROUP BY user_id,item_id ORDER BY user_id,t DESC,item_id", (cutoff,))
        snap = make_snapshot(cursor, items, cutoff, config, digest(data / 'manifest.json'))
    finally:
        db.close()
    print(f'Snapshot: {len(snap.users)} users, {len(snap.items)} items, '
          f'{len(snap.rows)} capped edges / {int(snap.user_degree.sum())} window edges', flush=True)
    return snap


def graph_matrices(snapshot, config):
    s = snapshot
    age = (s.cutoff_ms - s.times) / DAY_MS
    affinity = np.exp2(-age / config.graph_half_life_days) if config.graph_half_life_days else np.ones(len(age))
    degree = np.bincount(s.columns, weights=affinity, minlength=len(s.items))
    iuf = 1 / np.log(2 + s.user_degree.astype(float)) if config.user_iuf else np.ones(len(s.users))
    values = np.sqrt(affinity * iuf[s.rows]) / np.maximum(degree[s.columns], np.finfo(float).tiny) ** config.popularity_alpha
    shape = (len(s.users), len(s.items))
    normalized = csr_matrix((values, (s.rows, s.columns)), shape=shape)
    binary = csr_matrix((np.ones(len(s.rows)), (s.rows, s.columns)), shape=shape)
    return normalized, binary


def top_indices(indices, scores, k, items):
    valid = np.isfinite(scores) & (scores > 0)
    indices, scores = indices[valid], scores[valid]
    if len(scores) > k:
        threshold = np.partition(scores, -k)[-k]
        keep = scores >= threshold
        indices, scores = indices[keep], scores[keep]
    order = np.lexsort((items[indices], -scores))[:k]
    return indices[order], scores[order]


def build(snapshot, config, output):
    """Block sparse multiplication; never materializes the full item-item graph."""
    if snapshot.history_config != {k: getattr(config, k) for k in snapshot.history_config}:
        raise ValueError('Snapshot history construction differs from index config')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Published indexes are immutable; choose a new path')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.building-', dir=output.parent))
    started = perf_counter()
    s = snapshot
    n = len(s.items)
    normalized, binary = graph_matrices(s, config)
    left = normalized.tocsc()
    needs_support = config.shrinkage > 0 or config.min_cooccurrence > 1
    binary_left = binary.tocsc() if needs_support else None
    ids = np.lib.format.open_memmap(stage / 'neighbors.npy', mode='w+', dtype=np.int32,
                                    shape=(n, config.neighbors))
    weights = np.lib.format.open_memmap(stage / 'similarities.npy', mode='w+', dtype=np.float32,
                                        shape=(n, config.neighbors))
    edge_count, empty_rows = 0, 0
    try:
        for lo in range(0, n, config.block_items):
            hi = min(n, lo + config.block_items)
            block = (left[:, lo:hi].T @ normalized).tocsr()
            block.sort_indices()
            support = (binary_left[:, lo:hi].T @ binary).tocsr() if needs_support else None
            if support is not None:
                support.sort_indices()
            ids[lo:hi] = -1
            weights[lo:hi] = 0
            for local, source in enumerate(range(lo, hi)):
                a, b = block.indptr[local:local + 2]
                cols, vals = block.indices[a:b], block.data[a:b].copy()
                keep = cols != source
                cols, vals = cols[keep], vals[keep]
                if support is not None and len(cols):
                    sa, sb = support.indptr[local:local + 2]
                    sc = support.indices[sa:sb]
                    positions = np.searchsorted(sc, cols)
                    counts = support.data[sa:sb][positions]
                    keep = counts >= config.min_cooccurrence
                    cols, vals, counts = cols[keep], vals[keep], counts[keep]
                    if config.shrinkage:
                        vals *= counts / (counts + config.shrinkage)
                cols, vals = top_indices(cols, vals, config.neighbors, s.items)
                ids[source, :len(cols)] = cols
                weights[source, :len(cols)] = vals
                edge_count += len(cols)
                empty_rows += not len(cols)
            if lo == 0 or hi == n or hi % (config.block_items * 256) == 0:
                print(f'Index {output.name}: {hi}/{n} item rows, {edge_count} edges', flush=True)
        ids.flush()
        weights.flush()
    finally:
        ids._mmap.close()
        weights._mmap.close()
    for name, values in {'items': s.items, 'users': s.users, 'seed_offsets': s.seed_offsets,
                         'seed_columns': s.seed_columns, 'seed_times': s.seed_times}.items():
        np.save(stage / (name + '.npy'), values, allow_pickle=False)
    files = {p.name: digest(p) for p in sorted(stage.glob('*.npy'))}
    metadata = {'schema_version': 1, 'status': 'ready', 'cutoff_ms': s.cutoff_ms,
                'created_at': datetime.now(timezone.utc).isoformat(), 'config': asdict(config),
                'manifest_sha256': s.manifest_sha256, 'source_sha256': digest(Path(__file__)),
                'catalog_size': n, 'users': len(s.users), 'retained_train_edges': len(s.rows),
                'window_train_edges_before_cap': int(s.user_degree.sum()),
                'graph_edges': edge_count, 'zero_neighbor_items': empty_rows,
                'array_bytes': sum(p.stat().st_size for p in stage.glob('*.npy')),
                'build_seconds': perf_counter() - started, 'files': files,
                'semantics': 'latest distinct rec-train clicks; per-user cap; weighted cosine; '
                             'zero diagonal; row-wise TopK is directed; no embedded popularity fallback'}
    metadata['version'] = hashlib.sha256(json.dumps(
        {'files': files, 'config': asdict(config), 'cutoff_ms': s.cutoff_ms,
         'manifest': s.manifest_sha256}, sort_keys=True).encode()).hexdigest()[:20]
    (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if output.exists():
        raise FileExistsError('Output appeared during build; staged index preserved')
    os.rename(stage, output)
    print(f'Published {output}: {edge_count} edges, {metadata["array_bytes"] / 2**20:.1f} MiB', flush=True)
    return metadata


class ItemCFIndex:
    def __init__(self, path, expected_manifest_sha256=None, verify=True):
        started = perf_counter()
        self.path = Path(path)
        self.metadata = json.loads((self.path / 'metadata.json').read_text(encoding='utf-8'))
        m = self.metadata
        if m.get('status') != 'ready' or m.get('schema_version') != 1:
            raise ValueError('Unpublished or unsupported ItemCF index')
        if expected_manifest_sha256 and m['manifest_sha256'] != expected_manifest_sha256:
            raise ValueError('ItemCF index and dataset snapshot differ')
        required = {n + '.npy' for n in ('items', 'users', 'seed_offsets', 'seed_columns',
                                        'seed_times', 'neighbors', 'similarities')}
        if set(m['files']) != required:
            raise ValueError('Unexpected index file manifest')
        version = hashlib.sha256(json.dumps(
            {'files': m['files'], 'config': m['config'], 'cutoff_ms': m['cutoff_ms'],
             'manifest': m['manifest_sha256']}, sort_keys=True).encode()).hexdigest()[:20]
        if version != m['version']:
            raise ValueError('Index metadata version mismatch')
        if verify:
            for name, expected in m['files'].items():
                if digest(self.path / name) != expected:
                    raise ValueError(f'ItemCF file checksum mismatch: {name}')
        self.config = IndexConfig(**m['config'])
        self._arrays = []
        try:
            for name in ('items', 'users', 'seed_offsets', 'seed_columns', 'seed_times',
                         'neighbors', 'similarities'):
                value = np.load(self.path / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                setattr(self, name, value)
                self._arrays.append(value)
            if (self.neighbors.shape != (len(self.items), self.config.neighbors)
                    or self.similarities.shape != self.neighbors.shape
                    or self.neighbors.dtype != np.int32 or self.similarities.dtype != np.float32
                    or len(self.seed_offsets) != len(self.users) + 1
                    or self.seed_offsets[-1] != len(self.seed_columns)
                    or len(self.seed_times) != len(self.seed_columns)
                    or len(self.items) != m['catalog_size']):
                raise ValueError('Index array shape/dtype mismatch')
        except Exception:
            self.close()
            raise
        self.load_seconds = perf_counter() - started

    def close(self):
        for value in getattr(self, '_arrays', []):
            value._mmap.close()
        self._arrays = []

    def has_user(self, uid):
        pos = np.searchsorted(self.users, uid)
        return pos < len(self.users) and self.users[pos] == uid

    def query(self, uid, k=1000, timestamp_ms=None, *, history=None, exclude_items=(),
              exclude_history=False, explain=False, half_life_days='default', neighbor_limit=None):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        timestamp = self.metadata['cutoff_ms'] if timestamp_ms is None else int(timestamp_ms)
        if timestamp < self.metadata['cutoff_ms']:
            raise ValueError('Request predates index snapshot')
        half = self.config.query_half_life_days if half_life_days == 'default' else half_life_days
        if half is not None and (isinstance(half, bool) or not np.isfinite(half) or half <= 0):
            raise ValueError('half-life must be positive or null')
        width = self.config.neighbors if neighbor_limit is None else neighbor_limit
        if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= self.config.neighbors:
            raise ValueError('Invalid neighbor limit')
        ignored = {'not_prior': 0, 'unknown_item': 0, 'duplicate': 0}
        if history is None:
            pos = np.searchsorted(self.users, uid)
            if pos == len(self.users) or self.users[pos] != uid:
                cols, times = np.array([], dtype=np.int32), np.array([], dtype=np.int64)
            else:
                a, b = self.seed_offsets[pos:pos + 2]
                cols, times = self.seed_columns[a:b], self.seed_times[a:b]
            seen_history = self.items[cols].tolist()
        else:
            cols, times, seen = [], [], set()
            for iid, ts in sorted(history, key=lambda x: (-x[1], x[0])):
                iid, ts = int(iid), int(ts)
                if ts >= timestamp:
                    ignored['not_prior'] += 1
                    continue
                pos = np.searchsorted(self.items, iid)
                if pos == len(self.items) or self.items[pos] != iid:
                    ignored['unknown_item'] += 1
                    continue
                if iid in seen:
                    ignored['duplicate'] += 1
                    continue
                seen.add(iid)
                if len(cols) < self.config.seed_items:
                    cols.append(pos)
                    times.append(ts)
            cols, times = np.asarray(cols, dtype=np.int32), np.asarray(times, dtype=np.int64)
            seen_history = seen
        result = {'user_id': int(uid), 'items': [], 'scores': [], 'seed_count': len(cols),
                  'edges_visited': 0, 'unique_candidates_before_filter': 0, 'shortfall': k,
                  'ignored_history': ignored, 'index_version': self.metadata['version'],
                  'snapshot_ms': self.metadata['cutoff_ms'], 'fallback_applied': False}
        if not len(cols):
            result['reason'] = 'no_usable_history'
            return result
        age = (timestamp - times.astype(np.float64)) / DAY_MS
        affinity = np.exp2(-age / half) if half is not None else np.ones(len(cols))
        neighbors = self.neighbors[cols, :width]
        values = self.similarities[cols, :width].astype(np.float64) * affinity[:, None]
        mask = (neighbors >= 0) & (values > 0)
        ids, raw = neighbors[mask], values[mask]
        result['edges_visited'] = int(mask.sum())
        unique, inverse = np.unique(ids, return_inverse=True)
        scores = np.bincount(inverse, weights=raw, minlength=len(unique))
        result['unique_candidates_before_filter'] = len(unique)
        excluded = set(int(i) for i in exclude_items)
        if exclude_history:
            excluded.update(seen_history)
        if excluded:
            keep = ~np.isin(self.items[unique], list(excluded))
            unique, scores = unique[keep], scores[keep]
        selected, scores = top_indices(unique, scores, k, self.items)
        result.update({'items': self.items[selected].tolist(), 'scores': scores.tolist(),
                       'shortfall': max(0, k - len(selected)),
                       'reason': 'ok' if len(selected) else 'no_eligible_neighbors'})
        if explain:
            seed_for_edge = np.broadcast_to(np.arange(len(cols))[:, None], mask.shape)[mask]
            explanations = {}
            for item in selected:
                take = np.flatnonzero(ids == item)
                order = np.lexsort((self.items[cols[seed_for_edge[take]]], -raw[take]))[:3]
                explanations[str(int(self.items[item]))] = [
                    {'seed_item_id': int(self.items[cols[seed_for_edge[p]]]),
                     'contribution': float(raw[p]), 'history_weight': float(affinity[seed_for_edge[p]])}
                    for p in take[order]]
            result['top_seed_contributions'] = explanations
        return result

    def query_many(self, user_ids, k=1000, timestamp_ms=None, **kwargs):
        """Lazy batch interface keeps result memory bounded; no implicit concurrency."""
        for uid in user_ids:
            yield self.query(uid, k, timestamp_ms, **kwargs)


def export_prefix(source, output, neighbors):
    """Publish a physically smaller prefix without recomputing cooccurrences."""
    if isinstance(neighbors, bool) or not isinstance(neighbors, int) or not 1 <= neighbors <= source.config.neighbors:
        raise ValueError('Invalid exported neighbor count')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new immutable index path')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.building-', dir=output.parent))
    started = perf_counter()
    count, empty = 0, 0
    for name, dtype, original in [('neighbors', np.int32, source.neighbors),
                                  ('similarities', np.float32, source.similarities)]:
        target = np.lib.format.open_memmap(stage / (name + '.npy'), mode='w+', dtype=dtype,
                                          shape=(len(source.items), neighbors))
        try:
            for lo in range(0, len(source.items), 4096):
                hi = min(lo + 4096, len(source.items))
                target[lo:hi] = original[lo:hi, :neighbors]
                if name == 'neighbors':
                    valid = target[lo:hi] >= 0
                    count += int(valid.sum())
                    empty += int((~valid.any(axis=1)).sum())
            target.flush()
        finally:
            target._mmap.close()
    for name in ('items', 'users', 'seed_offsets', 'seed_columns', 'seed_times'):
        shutil.copyfile(source.path / (name + '.npy'), stage / (name + '.npy'))
    config = asdict(source.config)
    config['neighbors'] = neighbors
    files = {p.name: digest(p) for p in sorted(stage.glob('*.npy'))}
    metadata = {**source.metadata, 'config': config, 'files': files, 'graph_edges': count,
                'zero_neighbor_items': empty, 'array_bytes': sum(p.stat().st_size for p in stage.glob('*.npy')),
                'derived_from_version': source.metadata['version'],
                'parent_graph_build_seconds': source.metadata['build_seconds'],
                'source_sha256': digest(Path(__file__)),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'build_seconds': perf_counter() - started, 'build_kind': 'verified_row_prefix_export'}
    metadata['version'] = hashlib.sha256(json.dumps(
        {'files': files, 'config': config, 'cutoff_ms': metadata['cutoff_ms'],
         'manifest': metadata['manifest_sha256']}, sort_keys=True).encode()).hexdigest()[:20]
    (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if output.exists():
        raise FileExistsError('Output appeared during export; staging preserved')
    os.rename(stage, output)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--config', type=Path, required=True)
    b.add_argument('--data', type=Path, default=DATA)
    b.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('query')
    q.add_argument('--index', type=Path, required=True)
    q.add_argument('--user-id', type=int, required=True)
    q.add_argument('--k', type=int, default=20)
    q.add_argument('--explain', action='store_true')
    e = sub.add_parser('export')
    e.add_argument('--index', type=Path, required=True)
    e.add_argument('--out', type=Path, required=True)
    e.add_argument('--neighbors', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        config = IndexConfig(**json.loads(args.config.read_text(encoding='utf-8')))
        build(read_snapshot(args.data, config), config, args.out)
    elif args.command == 'export':
        index = ItemCFIndex(args.index)
        try:
            print(json.dumps(export_prefix(index, args.out, args.neighbors), indent=2))
        finally:
            index.close()
    else:
        index = ItemCFIndex(args.index)
        try:
            print(json.dumps(index.query(args.user_id, args.k, explain=args.explain), indent=2))
        finally:
            index.close()


if __name__ == '__main__':
    main()
