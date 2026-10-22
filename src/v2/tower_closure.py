"""Bounded, validation-selected final dual-tower study; no final-test access.

The validation selector is exact full-catalog retrieval on fixed users, NOT
in-batch accuracy. Non-clicked contrast items are never called exposed negatives.
Only full-epoch resume is supported; partial epochs must be replayed.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.itemcf_index import DATA, digest
from src.v2.ranking_convergence import write_json, save_checkpoint
from src.v2.tower_data import DAY, load_metadata
from src.v2.tower_model import FeatureTwoTower, in_batch_loss, model_source_digest
from src.v2.tower_retrieval import best


def load_training(path):
    meta = load_metadata(path)
    names = ('sample_contexts', 'targets', 'context_users', 'context_rec', 'context_search',
             'item_fields', 'query_user_tokens', 'query_rec', 'query_search', 'query_users', 'items')
    values = {name: torch.from_numpy(np.load(path/(name+'.npy')).astype(np.int64)) for name in names}
    return meta, values


def population(data, cutoff):
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if cutoff+DAY > manifest['cutoffs_ms'][1]:
        raise ValueError('Refuse any final test labels')
    positives, users = defaultdict(Counter), set()
    with closing(sqlite3.connect((data/'dataset.sqlite').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        for uid, iid, y in db.execute("SELECT user_id,item_id,label FROM clean_events "
                "WHERE scene='rec' AND timestamp_ms>=? AND timestamp_ms<?", (cutoff, cutoff+DAY)):
            users.add(uid)
            if y: positives[uid][iid] += 1
    return sorted(users), positives


def query_arrays(raw_users, tensors):
    stored = tensors['query_users'].numpy()
    positions = np.searchsorted(stored, raw_users)
    safe = np.minimum(positions, len(stored)-1)
    known = (positions < len(stored)) & (stored[safe] == raw_users)
    result = []
    for name in ('query_user_tokens', 'query_rec', 'query_search'):
        value = tensors[name][torch.from_numpy(safe)].clone()
        value[~torch.from_numpy(known)] = 1 if name == 'query_user_tokens' else 0
        result.append(value)
    return result


def query_signal(values, config):
    uid, rec, search = values
    return ((uid.ge(2) & config.get('use_user_id', True))
            | (rec.ne(0).any(1) & config.get('use_history', True))
            | (search.ne(0).any(1) & config.get('use_search', False))).numpy()


def encode_catalog(model, count, batch=4096):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model.items(torch.arange(lo+2, min(count, lo+batch)+2)).numpy()
                               for lo in range(0, count, batch)])


def retrieve(vectors, queries, item_ids, k=1000, batch=64, block=32768):
    """Use the same deterministic block/top-k algorithm as the served TowerIndex."""
    output = np.empty((len(queries), min(k, len(item_ids))), dtype=np.int64)
    with threadpool_limits(limits=4):
        for lo in range(0, len(queries), batch):
            q = queries[lo:lo+batch]
            ids = [np.empty(0, np.int64) for _ in q]
            scores = [np.empty(0, np.float32) for _ in q]
            for start in range(0, len(item_ids), block):
                values = q@vectors[start:start+block].T
                for row in range(len(q)):
                    ids[row], scores[row] = best(np.r_[ids[row], item_ids[start:start+block]],
                                                np.r_[scores[row], values[row]], k)
            output[lo:lo+len(q)] = np.asarray(ids)
    return output


def selector(model, tensors, users, positives, cfg):
    tick = perf_counter()
    inputs = query_arrays(np.asarray(users), tensors)
    signal = query_signal(inputs, cfg)
    model.eval()
    with torch.inference_mode():
        queries = model.users(*inputs).numpy()
        vectors = encode_catalog(model, len(tensors['items']), cfg['export_batch_size'])
    ids = retrieve(vectors, queries, tensors['items'].numpy(), cfg['validation_k'])
    hits = sum(sum(positives[u].get(int(i), 0) for i in ids[j]) for j, u in enumerate(users) if signal[j])
    counts = sum(sum(positives[u].values()) for u in users)
    if not counts: raise ValueError('Selector needs logged positive events')
    return {'event_recall': hits/counts, 'hit_events': hits, 'positive_events': counts,
            'users': len(users), 'usable_users': int(signal.sum()), 'seconds': perf_counter()-tick}


def batch_loss(model, arrays, indices, cfg):
    context = arrays['sample_contexts'][indices]
    target = arrays['targets'][indices]
    rec, search = arrays['context_rec'][context], arrays['context_search'][context]
    unique = torch.unique(target, sorted=True)
    if len(unique) < 2: return None
    return in_batch_loss(model.users(arrays['context_users'][context], rec, search), model.items(unique),
                         target, rec, search, cfg['temperature'], None, cfg['mask_history_negatives'])


def probe_loss(model, arrays, indices, cfg):
    model.eval()
    total = rows = 0
    with torch.inference_mode():
        for lo in range(0, len(indices), cfg['batch_size']):
            ix = indices[lo:lo+cfg['batch_size']]
            loss = batch_loss(model, arrays, ix, cfg)
            if loss is not None: total += loss.item()*len(ix); rows += len(ix)
    return total/rows


def build_model(meta, cfg, arrays):
    return FeatureTwoTower(meta['cardinalities'], cfg, arrays['item_fields'])


def optimizers_for(model, cfg):
    sparse = model.sparse_parameters()
    ids = {id(p) for p in sparse}
    return [torch.optim.Adam([p for p in model.parameters() if id(p) not in ids], lr=cfg['dense_lr']),
            torch.optim.SparseAdam(sparse, lr=cfg['sparse_lr'])]


def fit(prepared, cfg, output, data=DATA, resume=False, stop_after=None):
    if cfg['device'] != 'cpu': raise ValueError('This bounded closure entry explicitly targets local CPU')
    if cfg.get('logq_correction'): raise ValueError('logQ is outside the final experiment scope')
    torch.set_num_threads(cfg['threads'])
    meta, arrays = load_training(prepared)
    users, positive = population(data, meta['cutoff_ms'])
    rng = np.random.default_rng(cfg['validation_seed'])
    selected = np.sort(rng.choice(users, min(len(users), cfg['validation_users']), replace=False)).tolist()
    protocol = {'config': cfg, 'prepared_sha256': digest(prepared/'metadata.json'),
                'manifest_sha256': digest(data/'manifest.json'), 'selector_users': selected,
                'source_sha256': digest(Path(__file__)), 'model_source_sha256': model_source_digest(),
                'test_used': False, 'default_changed': False}
    torch.manual_seed(cfg['seed'])
    model = build_model(meta, cfg, arrays)
    optimizers = optimizers_for(model, cfg)
    order_rng = torch.Generator().manual_seed(cfg['seed']+1000)
    probe = torch.from_numpy(np.random.default_rng(2026).choice(len(arrays['targets']),
                            min(cfg['probe_events'], len(arrays['targets'])), replace=False))
    history, best_state, best_epoch = [], None, 0
    best_score = plateau = -1.
    bad = lr_bad = reductions = 0
    elapsed = 0.
    if resume:
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if saved['protocol'] != protocol: raise ValueError('Resume protocol mismatch')
        model.load_state_dict(saved['state_dict'])
        for opt, state in zip(optimizers, saved['optimizers']): opt.load_state_dict(state)
        torch.set_rng_state(saved['torch_rng']); order_rng.set_state(saved['order_rng'])
        history, best_state, best_epoch = saved['history'], saved['best_state'], saved['best_epoch']
        best_score, plateau = saved['best_score'], saved['plateau']
        bad, lr_bad, reductions = saved['bad'], saved['lr_bad'], saved['reductions']
        elapsed = saved['elapsed_seconds']
        if saved['finished']: return materialize(saved, meta, output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'protocol.json', protocol)
    started = perf_counter()
    for epoch in range(len(history)+1, cfg['max_epochs']+1):
        model.train()
        order = torch.randperm(len(arrays['targets']), generator=order_rng)
        total = rows = skipped = steps = 0
        lrs = [o.param_groups[0]['lr'] for o in optimizers]
        tick = perf_counter()
        for lo in range(0, len(order), cfg['batch_size']):
            ix = order[lo:lo+cfg['batch_size']]
            loss = batch_loss(model, arrays, ix, cfg)
            if loss is None: skipped += len(ix); continue
            if not torch.isfinite(loss): raise ValueError('Nonfinite tower loss')
            for opt in optimizers: opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers: opt.step()
            steps += 1
            total += loss.item()*len(ix); rows += len(ix)
            if lo and lo//cfg['batch_size'] % 1000 == 0:
                print(f'{output.name} epoch={epoch} rows={lo}/{len(order)} elapsed={perf_counter()-tick:.1f}s', flush=True)
        if not rows: raise ValueError('No usable training batches')
        train_seconds = perf_counter()-tick
        probe_value = probe_loss(model, arrays, probe, cfg)
        valid = selector(model, arrays, selected, positive, cfg)
        score = valid['event_recall']
        history.append({'epoch': epoch, 'train_contrast_loss': total/rows, 'probe_contrast_loss': probe_value,
                        'trained_events': rows, 'skipped_events': skipped, 'optimizer_steps': steps, 'learning_rates': lrs,
                        'validation': valid, 'train_seconds': train_seconds,
                        'order_sha256': hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if score > plateau+cfg['min_delta']:
            plateau = score; bad = lr_bad = 0
        else: bad += 1; lr_bad += 1
        if lr_bad >= cfg['lr_patience']:
            for opt in optimizers: opt.param_groups[0]['lr'] *= cfg['lr_factor']
            reductions += 1; lr_bad = 0
        plateau_stop = epoch >= cfg['min_epochs'] and bad >= cfg['early_patience'] and reductions >= 2
        finished = plateau_stop or epoch == cfg['max_epochs']
        saved = {'protocol': protocol, 'state_dict': model.state_dict(), 'best_state': best_state,
                 'optimizers': [o.state_dict() for o in optimizers], 'torch_rng': torch.get_rng_state(),
                 'order_rng': order_rng.get_state(), 'history': history, 'best_score': best_score,
                 'best_epoch': best_epoch, 'plateau': plateau, 'bad': bad, 'lr_bad': lr_bad,
                 'reductions': reductions, 'finished': finished,
                 'reason': 'validation_plateau' if plateau_stop else
                    ('max_epochs_not_proof_of_convergence' if finished else 'running'),
                 'elapsed_seconds': elapsed+perf_counter()-started}
        save_checkpoint(output/'last.pt', saved)
        result = materialize(saved, meta, output)
        print(f'{output.name} epoch={epoch} loss={total/rows:.5f} probe={probe_value:.5f} '
              f'validR={score:.6f} best={best_epoch} elapsed={saved["elapsed_seconds"]:.1f}s', flush=True)
        if finished or (stop_after is not None and epoch >= stop_after): break
    return result


def materialize(saved, meta, output):
    checkpoint = {'state_dict': saved['best_state'], 'config': saved['protocol']['config'],
                  'data_metadata_sha256': saved['protocol']['prepared_sha256'],
                  'cardinalities': meta['cardinalities'], 'model_source_sha256': model_source_digest(),
                  'selected_epoch': saved['best_epoch'], 'torch_version': str(torch.__version__)}
    save_checkpoint(output/'model.pt', checkpoint)
    result = {'status': 'complete' if saved['finished'] else 'running', 'protocol': saved['protocol'],
              'history': saved['history'], 'selected_epoch': saved['best_epoch'],
              'selected_validation': saved['history'][saved['best_epoch']-1]['validation'],
              'early_stopping_reason': saved['reason'], 'elapsed_seconds': saved['elapsed_seconds'],
              'checkpoint_sha256': digest(output/'model.pt'), 'test_used': False,
              'optimizer_steps': sum(r['optimizer_steps'] for r in saved['history'])}
    write_json(output/'metrics.json', result)
    return result


def benchmark(prepared, cfg, steps=20):
    torch.set_num_threads(cfg['threads'])
    meta, arrays = load_training(prepared)
    reports = {}
    for history in (False, True):
        arm = dict(cfg, use_history=history)
        torch.manual_seed(2026)
        model = build_model(meta, arm, arrays)
        opts = optimizers_for(model, arm)
        order = torch.randperm(len(arrays['targets']))
        started = perf_counter()
        for j in range(steps+2):
            ix = order[j*cfg['batch_size']:(j+1)*cfg['batch_size']]
            loss = batch_loss(model, arrays, ix, arm)
            for opt in opts: opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in opts: opt.step()
            if j == 1: started = perf_counter()
        seconds = (perf_counter()-started)/steps
        reports['history' if history else 'attributes'] = {'seconds_per_step': seconds,
            'estimated_training_seconds_per_epoch': seconds*((len(order)+cfg['batch_size']-1)//cfg['batch_size']),
            'scope': 'measured optimizer steps, excluding validation/checkpoint/export/full evaluation'}
        del model, opts
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--config', type=Path, default=Path('configs/tower_closure.json'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--arm', choices=['attributes', 'history'], default='history')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config.update(seed=args.seed, use_history=args.arm == 'history')
    if args.benchmark:
        print(json.dumps(benchmark(args.prepared, config), indent=2), flush=True)
    else:
        if args.out is None: parser.error('--out required')
        print(json.dumps(fit(args.prepared, config, args.out, args.data, args.resume), indent=2))
