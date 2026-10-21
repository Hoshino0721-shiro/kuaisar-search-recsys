"""Exposure-aligned, versioned DCN input cache; no test-label reads.

Not the positive-only tower cache. Histories retain repeated events and freeze
before each complete day. Attribute vocabularies use training exposures only.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import sqlite3

import numpy as np

from src.ctr_features import CACHE, DATA, FeatureState
from src.v2.train_rankers import sha

DAY = 86400000
OFFSET = 28800000
MISSING = {None, '', 'NULL', 'null', 'unknown', 'UNKNOWN', 'UNKNOW'}


def vocabulary(counts, minimum):
    return {value: j + 2 for j, value in enumerate(sorted(k for k, n in counts.items() if n >= minimum))}


def history_array(events, length):
    a = np.zeros(length, dtype=np.int32)
    values = list(events)[-length:]
    if values:
        a[-len(values):] = values
    return a


def prepare(output, sample_rows=1000000, sample_seed=2026, minimum=5, length=30,
            cache=CACHE, data=DATA):
    if output.exists():
        raise FileExistsError('Choose a new enhanced cache directory')
    if min(sample_rows, minimum, length) < 1:
        raise ValueError('Preparation budgets must be positive')
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    cutoff = manifest['cutoffs_ms'][0]
    if manifest['status'] != 'ready' or (cutoff + OFFSET) % DAY:
        raise ValueError('Ready data with Beijing midnight cutoff required')
    state = FeatureState.load(cache / 'state.npz')
    tu = np.load(cache / 'train_users.npy', mmap_mode='r')
    ti = np.load(cache / 'train_items.npy', mmap_mode='r')
    indices = np.sort(np.random.default_rng(sample_seed).choice(len(tu), min(sample_rows, len(tu)), replace=False))
    uf = np.bincount(tu, minlength=len(state.un))
    itf = np.bincount(ti, minlength=len(state.inum))
    fit_uf = np.bincount(tu[indices], minlength=len(uf))
    fit_itf = np.bincount(ti[indices], minlength=len(itf))
    user_vocab = vocabulary(dict(enumerate(fit_uf)), minimum)
    item_vocab = vocabulary(dict(enumerate(fit_itf)), minimum)
    user_lookup = np.ones(len(uf), dtype=np.int32)
    for raw, token in user_vocab.items():
        user_lookup[raw] = token
    item_fields = np.ones((len(itf), 3), dtype=np.int32)
    for raw, token in item_vocab.items():
        item_fields[raw, 0] = token
    db = sqlite3.connect((data / 'dataset.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    try:
        counts = {'author': Counter(), 'category': Counter()}
        for iid, author, category in db.execute('SELECT item_id,author_id,category_id FROM items'):
            n = int(fit_itf[iid]) if iid < len(fit_itf) else 0
            if n:
                if author not in MISSING:
                    counts['author'][str(author)] += n
                if category not in MISSING and str(category) != '0':
                    counts['category'][str(category)] += n
        av, cv = (vocabulary(counts[name], minimum) for name in ('author', 'category'))
        for iid, author, category in db.execute('SELECT item_id,author_id,category_id FROM items'):
            if 0 <= iid < len(item_fields):
                item_fields[iid, 1] = av.get(str(author), 1) if author not in MISSING else 1
                item_fields[iid, 2] = cv.get(str(category), 1) if category not in MISSING else 1
        contexts_rec, contexts_search, context_users = [], [], []
        train_context = np.empty(len(indices), dtype=np.int32)
        rec, search = defaultdict(lambda: deque(maxlen=length)), defaultdict(lambda: deque(maxlen=length))
        first = db.execute("SELECT min(timestamp_ms) FROM clean_events WHERE scene='rec' AND split='train'").fetchone()[0]
        day = ((first + OFFSET) // DAY) * DAY - OFFSET
        offset = 0
        day_reports = []
        while day < cutoff:
            end = min(day + DAY, cutoff)
            n = db.execute("SELECT count(*) FROM clean_events WHERE scene='rec' AND split='train' "
                           "AND timestamp_ms>=? AND timestamp_ms<?", (day, end)).fetchone()[0]
            lo, hi = np.searchsorted(indices, [offset, offset + n])
            users = np.asarray(tu[indices[lo:hi]])
            per_day = {}
            for uid in sorted(set(users.tolist())):
                per_day[uid] = len(context_users)
                context_users.append(uid)
                contexts_rec.append(history_array(rec[uid], length))
                contexts_search.append(history_array(search[uid], length))
            train_context[lo:hi] = [per_day[int(u)] for u in users]
            # Only after all target contexts are frozen may today's clicks enter memory.
            for uid, iid in db.execute(
                    "SELECT user_id,item_id FROM clean_events WHERE scene='rec' AND split='train' "
                    "AND label=1 AND timestamp_ms>=? AND timestamp_ms<? ORDER BY timestamp_ms,item_id,id", (day, end)):
                rec[uid].append(int(iid))
            search = defaultdict(lambda: deque(maxlen=length))
            for uid, iid in db.execute(
                    "SELECT user_id,item_id FROM clean_events WHERE scene='search' AND split='train' "
                    "AND label>0 AND timestamp_ms>=? AND timestamp_ms<? ORDER BY timestamp_ms,item_id,id", (day, end)):
                search[uid].append(int(iid))
            day_reports.append({'start_ms': day, 'end_ms': end, 'rows': n, 'sampled': int(hi-lo),
                                'context_count': len(per_day)})
            offset += n
            day = end
            print(f'Enhanced history day {day}: {hi}/{len(indices)} sample contexts', flush=True)
        if offset != len(tu):
            raise ValueError('Daily row counts differ from numeric training cache')
        query_rec = np.stack([history_array(rec[u], length) for u in range(len(uf))])
        query_search = np.stack([history_array(search[u], length) for u in range(len(uf))])
        context_rec = np.stack(contexts_rec)
        context_search = np.stack(contexts_search)
        # Histories are actual fitting inputs too. Do not restrict their IDs to
        # recommendation targets; this would erase almost all search histories.
        input_counts = fit_itf.copy()
        context_weights = np.bincount(train_context, minlength=len(context_users))
        for raw_history in (context_rec, context_search):
            for lo in range(0,len(raw_history),10000):
                block = raw_history[lo:lo+10000]
                counts = np.bincount(block.ravel(), weights=np.repeat(context_weights[lo:lo+len(block)],length),
                                     minlength=len(input_counts))
                if len(counts) > len(input_counts):
                    raise ValueError('History ID outside raw event vocabulary bounds')
                input_counts += counts.astype(np.int64)
        input_counts[0] = 0  # raw zero is PAD, never a learned item identity.
        item_vocab = vocabulary(dict(enumerate(input_counts)),minimum)
        item_fields[:,0] = 1
        for raw,token in item_vocab.items():
            item_fields[raw,0] = token
        def encode_history(raw):
            return np.where(raw==0,0,item_fields[raw,0]).astype(np.int32)
        context_rec,context_search,query_rec,query_search = map(
            encode_history,(context_rec,context_search,query_rec,query_search))
        def fields(users, items):
            return np.column_stack([user_lookup[users], item_fields[items]]).astype(np.int32)
        vu = np.load(cache / 'valid_users.npy')
        vi = np.load(cache / 'valid_items.npy')
        arrays = {'sample_indices': indices, 'user_lookup': user_lookup, 'item_fields': item_fields,
                  'item_train_frequency': itf.astype(np.int32),
                  'train_fields': fields(tu[indices], ti[indices]), 'valid_fields': fields(vu, vi),
                  'train_context': train_context, 'context_users': np.asarray(context_users, dtype=np.int32),
                  'context_rec': context_rec, 'context_search': context_search,
                  'query_rec': query_rec, 'query_search': query_search}
        output.mkdir(parents=True)
        for name, value in arrays.items():
            np.save(output / (name + '.npy'), value)
        (output / 'vocab.json').write_text(json.dumps({'user': list(user_vocab), 'item': list(item_vocab),
            'author': list(av), 'category': list(cv)}), encoding='utf-8')
        meta = {'status': 'ready', 'schema': 'enhanced_dcn_v1', 'cutoff_ms': cutoff,
                'minimum_frequency': minimum, 'history_length': length, 'sample_seed': sample_seed,
                'sample_rows': len(indices), 'days': day_reports,
                'cardinalities': {'user': len(user_vocab)+2, 'item': len(item_vocab)+2,
                                  'author': len(av)+2, 'category': len(cv)+2},
                'history': 'prior complete days; latest rec events with repeats; previous day search events with repeats',
                'ties': 'timestamp,item_id,id; deterministic, not a claim of feedback arrival order',
                'attributes': 'static author/category assumption; raw author 0 is not implicitly treated as missing',
                'vocab_scope': 'actual fixed fitting sample fields; shared item IDs additionally count their causal rec/search input histories, weighted by target rows; no valid/query/test fitting',
                'manifest_sha256': sha(data/'manifest.json'), 'feature_state_sha256': sha(cache/'state.npz'),
                'source_sha256': sha(__file__), 'test_used': False,
                'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
        (output / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return meta
    finally:
        db.close()


class EnhancedData:
    def __init__(self, path, verify=True):
        self.path = Path(path)
        self.meta = json.loads((self.path/'metadata.json').read_text(encoding='utf-8'))
        if self.meta.get('status') != 'ready' or self.meta.get('schema') != 'enhanced_dcn_v1':
            raise ValueError('Unsupported/incomplete enhanced cache')
        if verify:
            for name, digest in self.meta['files'].items():
                if Path(name).name != name or sha(self.path/name) != digest:
                    raise ValueError('Enhanced cache checksum mismatch: ' + name)
        for name in self.meta['files']:
            if name.endswith('.npy'):
                setattr(self, name[:-4], np.load(self.path/name, mmap_mode='r'))

    def fields(self, users, items):
        users, items = np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)
        if users.shape != items.shape or users.ndim != 1:
            raise ValueError('Expected aligned user/item IDs')
        if np.any(users < 0) or np.any(users >= len(self.user_lookup)) or np.any(items < 0) or np.any(items >= len(self.item_fields)):
            raise ValueError('IDs outside frozen raw table bounds')
        return np.column_stack([self.user_lookup[users], self.item_fields[items]]).astype(np.int64)

    def close(self):
        for value in self.__dict__.values():
            if isinstance(value, np.memmap) and value._mmap is not None:
                value._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.out), indent=2))
