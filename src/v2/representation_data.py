"""Full-exposure categorical/content inputs and strict prior-day histories.

New schema: author raw 0 is PAD, not an author identity. Raw anonymous word 0
is not assumed to be PAD: encoded words reserve 0/1 independently. All fitting
counts come from training target or causal input history, never validation.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from src.ctr_features import CACHE, DATA, connection
from src.v2.ranking_convergence import write_json
from src.v2.train_rankers import sha

DAY = 86400000
OFFSET = 28800000
MISSING = {None, '', '0', 'NULL', 'null', 'unknown', 'UNKNOWN', 'UNKNOW'}


def padded(values, length):
    a = np.zeros(length, np.int32)
    values = list(values)[-length:]
    if values:
        a[-len(values):] = values
    return a


def parse_caption(raw, length):
    if raw in ('', 'NULL', 'null', 'None'):
        return []
    value = json.loads(raw)
    if not isinstance(value, list) or any(type(x) is not int or x < 0 for x in value):
        raise ValueError('Expected nonnegative anonymous word ID list')
    return value[:length]


def vocab(counts, minimum, limit=None):
    keys = [key for key, count in counts.items() if count >= minimum]
    if limit is not None:
        keys = sorted(keys, key=lambda key: (-counts[key], key))[:limit]
    return {key: j+2 for j, key in enumerate(sorted(keys))}


def histories(users, times, cutoff, length):
    db = connection()
    rec, search = defaultdict(lambda: deque(maxlen=length)), defaultdict(lambda: deque(maxlen=length))
    context_users, context_times, cr, cs = [], [], [], []
    row_context = np.empty(len(users), np.int32)
    first = int(times.min())
    day = ((first+OFFSET)//DAY)*DAY-OFFSET
    cursor = 0
    try:
        while day < cutoff:
            stop = day+DAY
            n = int(np.sum((times >= day) & (times < stop)))
            uu = users[cursor:cursor+n]
            if n and not np.all((times[cursor:cursor+n] >= day) & (times[cursor:cursor+n] < stop)):
                raise ValueError('Training cache is not in day/id order')
            unique, inverse = np.unique(uu, return_inverse=True)
            start = len(context_users)
            for uid in unique:
                context_users.append(int(uid)); context_times.append(day)
                cr.append(padded(rec[int(uid)], length)); cs.append(padded(search[int(uid)], length))
            row_context[cursor:cursor+n] = start+inverse
            # All of this day's inputs are frozen before any same-day feedback.
            for uid, iid in db.execute("SELECT user_id,item_id FROM clean_events WHERE scene='rec' AND split='train' AND label=1 AND timestamp_ms>=? AND timestamp_ms<? ORDER BY timestamp_ms,item_id,id", (day, stop)):
                rec[uid].append(iid)
            search = defaultdict(lambda: deque(maxlen=length))
            for uid, iid in db.execute("SELECT user_id,item_id FROM clean_events WHERE scene='search' AND split='train' AND label=1 AND timestamp_ms>=? AND timestamp_ms<? AND result_type IN ('VIDEO','IMAGE_ATLAS') ORDER BY timestamp_ms,item_id,id", (day, stop)):
                search[uid].append(iid)
            cursor += n
            print(f'Representation histories day {len(set(context_times))}: {cursor}/{len(users)}', flush=True)
            day = stop
        if cursor != len(users):
            raise ValueError('Not all training rows have causal context')
        nusers = int(max(users.max(), np.load(CACHE/'valid_users.npy').max()))+1
        return {'train_context': row_context, 'context_users': np.asarray(context_users, np.int32),
                'context_times': np.asarray(context_times, np.int64),
                'context_rec': np.stack(cr), 'context_search': np.stack(cs),
                'query_rec': np.stack([padded(rec[u], length) for u in range(nusers)]),
                'query_search': np.stack([padded(search[u], length) for u in range(nusers)])}
    finally:
        db.close()


def prepare(config, output, times_path):
    if output.exists():
        raise FileExistsError('Preserve immutable representation cache')
    cfg = json.loads(config.read_text(encoding='utf-8'))
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    cutoff = manifest['cutoffs_ms'][0]
    if manifest['status'] != 'ready' or (cutoff+OFFSET) % DAY:
        raise ValueError('Ready midnight-aligned dataset required')
    from src.v2.exact_pair_data import verify
    verify(times_path.parent)  # Times were reconstructed with full SQL row checks.
    numeric_files = {f'{s}_{k}.npy': sha(CACHE/f'{s}_{k}.npy') for s in ('train', 'valid') for k in ('users', 'items', 'y')}
    times = np.load(times_path)
    users = np.load(CACHE/'train_users.npy')
    items = np.load(CACHE/'train_items.npy')
    state = np.load(CACHE/'state.npz')
    nitems = len(state['ip']); nusers = len(state['up'])
    arrays = histories(users, times, cutoff, cfg['history_length'])
    frequency = np.bincount(items, minlength=nitems).astype(np.int64)
    # A shared embedding may learn from actual causal history inputs too.
    input_counts = frequency.copy()
    weights = np.bincount(arrays['train_context'], minlength=len(arrays['context_users']))
    for key in ('context_rec', 'context_search'):
        for lo in range(0, len(weights), 4096):
            a = arrays[key][lo:lo+4096]
            input_counts += np.bincount(a.ravel(), weights=np.repeat(weights[lo:lo+len(a)], a.shape[1]), minlength=nitems).astype(np.int64)
    input_counts[0] = 0
    uv = vocab(dict(enumerate(np.bincount(users, minlength=nusers))), cfg['minimum_frequency'])
    iv = vocab(dict(enumerate(input_counts)), cfg['minimum_frequency'])
    user_lookup = np.ones(nusers, np.int32)
    for raw, token in uv.items(): user_lookup[raw] = token
    item_ids = np.ones(nitems, np.int32); item_ids[0] = 0
    for raw, token in iv.items(): item_ids[raw] = token
    db = connection()
    counts = {'author': Counter(), 'category': Counter()}
    try:
        for iid, author, category in db.execute('SELECT item_id,author_id,category_id FROM items'):
            weight = int(input_counts[iid])
            if weight:
                for name, raw in (('author', author), ('category', category)):
                    if raw not in MISSING: counts[name][str(raw)] += weight
        av, cv = [vocab(counts[k], cfg['minimum_frequency']) for k in ('author', 'category')]
        fields = np.ones((nitems, 3), np.int32); fields[:, 0] = item_ids; fields[0] = 0
        missing_author_rows = 0
        for iid, author, category in db.execute('SELECT item_id,author_id,category_id FROM items'):
            fields[iid, 1] = 0 if author in MISSING else av.get(str(author), 1)
            fields[iid, 2] = 0 if category in MISSING else cv.get(str(category), 1)
            if author in MISSING: missing_author_rows += int(frequency[iid])
    finally:
        db.close()
    output.mkdir(parents=True)
    raw_caption = np.lib.format.open_memmap(output/'caption_raw.npy', mode='w+', dtype=np.int32,
                                          shape=(nitems, cfg['caption_length']))
    raw_caption[:] = -1
    word_counts = Counter(); records = 0
    archive = Path('data/raw/KuaiSAR.zip')
    with zipfile.ZipFile(archive) as z:
        with io.TextIOWrapper(z.open('KuaiSAR_final/item_features.csv'), encoding='utf-8-sig', newline='') as stream:
            for record in csv.DictReader(stream):
                iid = int(record['item_id'])
                tokens = parse_caption(record['caption'], cfg['caption_length'])
                raw_caption[iid, :len(tokens)] = tokens
                weight = int(input_counts[iid])
                if weight:
                    for token in tokens: word_counts[token] += weight
                records += 1
    raw_caption.flush()
    wv = vocab(word_counts, cfg['minimum_frequency'], cfg['word_limit'])
    if not wv: raise ValueError('No fitted caption vocabulary')
    lookup = np.ones(max(max(wv), int(raw_caption.max()))+2, np.int32)
    for raw, token in wv.items(): lookup[raw] = token
    caption = np.lib.format.open_memmap(output/'caption.npy', mode='w+', dtype=np.int32, shape=raw_caption.shape)
    for lo in range(0, nitems, 100000):
        block = raw_caption[lo:lo+100000]
        caption[lo:lo+len(block)] = np.where(block < 0, 0, lookup[np.maximum(block, 0)])
    caption.flush()
    arrays.update(user_lookup=user_lookup, item_fields=fields, item_frequency=frequency,
                  item_input_frequency=input_counts)
    for name, array in arrays.items(): np.save(output/(name+'.npy'), array)
    write_json(output/'vocab.json', {name: list(value) for name, value in
               [('user', uv), ('item', iv), ('author', av), ('category', cv), ('word', wv)]})
    def coverage(raw):
        real = raw != 0
        ids = fields[raw, 0]
        word = caption[raw]
        return {'nonpad_history_items': int(real.sum()),
                'id_unk_fraction': float(np.mean(ids[real] == 1)) if real.any() else None,
                'items_with_known_word_fraction': float(np.mean((word[real] >= 2).any(axis=1))) if real.any() else None,
                'items_with_known_attribute_fraction': float(np.mean((fields[raw[real], 1:] >= 2).any(axis=1))) if real.any() else None}
    coverage_report = {name: coverage(arrays[name]) for name in ('context_rec', 'context_search', 'query_rec', 'query_search')}
    metadata = {'status': 'ready', 'schema': 'representation_v1', 'config': cfg,
                'train_rows': len(users), 'contexts': len(weights), 'caption_records': records,
                'cutoff_ms': cutoff, 'author_missing_train_rows': missing_author_rows,
                'cardinalities': {name: len(value)+2 for name, value in
                    [('user', uv), ('item', iv), ('author', av), ('category', cv), ('word', wv)]},
                'coverage': coverage_report, 'state_sha256': sha(CACHE/'state.npz'),
                'numeric_files': numeric_files, 'times_sha256': sha(times_path),
                'manifest_sha256': sha(DATA/'manifest.json'), 'archive_sha256': sha(archive),
                'source_sha256': sha(__file__), 'test_used': False,
                'history_rule': 'previous complete days; rec last10 repeated clicks; previous-day video/image search clicks',
                'vocab_rule': 'training exposure targets plus actual prior-day input histories only; no validation/query fitting',
                'attributes': 'static metadata assumption; raw author/category 0 -> PAD; unseen nonmissing -> UNK',
                'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/'metadata.json', metadata)
    raw_caption._mmap.close(); caption._mmap.close()
    print(json.dumps({'cardinalities': metadata['cardinalities'], 'coverage': coverage_report}, indent=2), flush=True)


def verify(path):
    meta = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('schema') != 'representation_v1' or meta.get('status') != 'ready' or meta['test_used']:
        raise ValueError('Ready non-test representation cache required')
    if meta['state_sha256'] != sha(CACHE/'state.npz'):
        raise ValueError('Representation state mismatch')
    for name, digest in meta['files'].items():
        if Path(name).name != name or sha(path/name) != digest: raise ValueError('Representation cache changed: '+name)
    for name, digest in meta['numeric_files'].items():
        if sha(CACHE/name) != digest: raise ValueError('Numeric cache changed: '+name)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/representation_v1.json'))
    parser.add_argument('--times', type=Path, default=Path('artifacts/exact_pair_finetune_20261022/pairs/train_times.npy'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.out, args.times)
