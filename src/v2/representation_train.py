"""Full BCE representation ablation with epoch validation and exact resume."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from src.ctr_features import CACHE
from src.v2.full_exposure_control import verify_prepared
from src.v2.ranking_convergence import ExposureDiagnostics, frozen_assets, save_checkpoint, write_json
from src.v2.representation_data import verify
from src.v2.representation_model import RepresentationDCN, RepresentationScorer, dense_and_sparse
from src.v2.route_attribution import normalize
from src.v2.temporal_route_train import verify_cache
from src.v2.train_rankers import sha


def load_data(path):
    meta = verify(path)
    return SimpleNamespace(path=path, meta=meta, **{name: torch.from_numpy(np.load(path/(name+'.npy')))
        for name in ('user_lookup', 'item_fields', 'caption', 'context_rec', 'context_search',
                     'query_rec', 'query_search', 'train_context')})


def validation(model, x, users, items, data, batch=8192):
    model.eval()
    with torch.inference_mode():
        logits = np.concatenate([model(x[lo:lo+batch], data.user_lookup[users[lo:lo+batch]],
            data.item_fields[items[lo:lo+batch]], data.caption[items[lo:lo+batch]]).numpy()
            for lo in range(0, len(items), batch)])
    return logits


def fit(content, seed, cfg, data, x, y, users, items, xv, uv, iv, diagnostics,
        mean, std, destination, context, resume=False, stop_after=None):
    protocol = {'config': cfg, 'content': content, 'seed': seed, 'context': context,
                'source_sha256': sha(__file__), 'train_rows': len(y), 'test_used': False}
    torch.manual_seed(seed)
    model = RepresentationDCN(data.meta['cardinalities'], content, cfg['embedding_dim'])
    initial = hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    dense, sparse = dense_and_sparse(model)
    optimizers = [torch.optim.Adam(dense, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay']),
                  torch.optim.SparseAdam(sparse, lr=cfg['learning_rate'])]
    order_rng = torch.Generator().manual_seed(seed+1000)
    ty = torch.as_tensor(y, dtype=torch.float32)
    probe = np.sort(np.random.default_rng(2026).choice(len(y), min(65536, len(y)), replace=False))
    history = []
    best = plateau = float('inf')
    best_epoch = bad = lr_bad = reductions = 0
    elapsed = 0.
    if resume:
        saved = torch.load(destination/'last.pt', map_location='cpu', weights_only=True)
        if saved['protocol'] != protocol: raise ValueError('Resume protocol mismatch')
        model.load_state_dict(saved['state_dict'])
        for opt, state in zip(optimizers, saved['optimizers']): opt.load_state_dict(state)
        torch.set_rng_state(saved['torch_rng']); order_rng.set_state(saved['order_rng'])
        history, best = saved['history'], saved['best']
        best_state = saved['best_state']
        plateau, best_epoch = saved['plateau'], saved['best_epoch']
        bad, lr_bad, reductions = saved['bad'], saved['lr_bad'], saved['reductions']
        elapsed = saved['elapsed_seconds']
        materialize(saved, destination)
        if saved['finished']: return json.loads((destination/'metrics.json').read_text())
    else:
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination/'protocol.json', protocol)
    start = perf_counter()
    for epoch in range(len(history)+1, cfg['max_epochs']+1):
        model.train(); order = torch.randperm(len(y), generator=order_rng)
        total = 0.; lr = optimizers[0].param_groups[0]['lr']
        for lo in range(0, len(y), cfg['batch_size']):
            ix = order[lo:lo+cfg['batch_size']]
            ids = items[ix]
            logits = model(x[ix], data.user_lookup[users[ix]], data.item_fields[ids], data.caption[ids])
            loss = F.binary_cross_entropy_with_logits(logits, ty[ix])
            if not torch.isfinite(loss): raise ValueError('Nonfinite representation BCE')
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for optimizer in optimizers: optimizer.step()
            total += loss.item()*len(ix)
        logits = validation(model, xv, uv, iv, data)
        p = torch.sigmoid(torch.from_numpy(logits)).numpy()
        valid = diagnostics.metrics(p)
        probe_logits = validation(model, x[probe], users[probe], items[probe], data).astype(np.float64)
        probe_loss = float(np.mean(np.logaddexp(0., probe_logits)-np.asarray(y)[probe]*probe_logits))
        score = valid['logloss']
        history.append({'epoch': epoch, 'learning_rate': lr, 'train_logloss': total/len(y),
                        'train_probe_logloss': probe_loss, 'valid': valid,
                        'order_sha256': hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        if score < best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if score < plateau-cfg['min_delta']:
            plateau = score; bad = lr_bad = 0
        else: bad += 1; lr_bad += 1
        if lr_bad >= cfg['lr_patience']:
            for optimizer in optimizers: optimizer.param_groups[0]['lr'] *= cfg['lr_factor']
            reductions += 1; lr_bad = 0
        plateau_stop = epoch >= cfg['min_epochs'] and bad >= cfg['early_patience'] and reductions >= 2
        finished = plateau_stop or epoch == cfg['max_epochs']
        metadata = {'kind': 'representation_dcn_v2', 'content': content, 'seed': seed,
                    'dimension': cfg['embedding_dim'], 'selected_epoch': best_epoch,
                    'initial_parameters_sha256': initial, 'test_used': False}
        saved = {'protocol': protocol, 'metadata': metadata, 'state_dict': model.state_dict(),
                 'best_state': best_state, 'mean': mean.tolist(), 'std': std.tolist(),
                 'optimizers': [o.state_dict() for o in optimizers], 'torch_rng': torch.get_rng_state(),
                 'order_rng': order_rng.get_state(), 'history': history, 'best': best, 'plateau': plateau,
                 'best_epoch': best_epoch, 'bad': bad, 'lr_bad': lr_bad, 'reductions': reductions,
                 'finished': finished, 'reason': 'validation_plateau' if plateau_stop else
                    ('max_epochs_not_proof_of_convergence' if finished else 'running'),
                 'elapsed_seconds': elapsed+perf_counter()-start}
        save_checkpoint(destination/'last.pt', saved)
        result = materialize(saved, destination)
        print(f'{destination.name} epoch {epoch}: trainLL={total/len(y):.6f} validLL={score:.6f} '
              f'AUC={valid["auc"]:.6f} GAUC={valid["gauc"]:.6f} best={best_epoch} '
              f'elapsed={saved["elapsed_seconds"]:.1f}s', flush=True)
        if finished or (stop_after is not None and epoch >= stop_after): break
    return result


def materialize(saved, destination):
    save_checkpoint(destination/'model.pt', {'state_dict': saved['best_state'], 'metadata': saved['metadata'],
                    'mean': saved['mean'], 'std': saved['std']})
    result = {'metadata': saved['metadata'], 'protocol': saved['protocol'], 'history': saved['history'],
              'selected_epoch': saved['best_epoch'], 'epochs_completed': len(saved['history']),
              'selected_validation': saved['history'][saved['best_epoch']-1]['valid'],
              'checkpoint_sha256': sha(destination/'model.pt'), 'elapsed_seconds': saved['elapsed_seconds'],
              'optimizer_steps': len(saved['history'])*((saved['protocol']['train_rows']+saved['protocol']['config']['batch_size']-1)//saved['protocol']['config']['batch_size']),
              'early_stopping_reason': saved['reason'], 'status': 'complete' if saved['finished'] else 'running',
              'test_used': False}
    write_json(destination/'metrics.json', result)
    return result


def run(cache, reference, prepared, config, output, seeds=None, resume=False, benchmark=False):
    cfg = json.loads(config.read_text(encoding='utf-8'))
    if seeds: cfg['seeds'] = seeds
    torch.set_num_threads(cfg['threads'])
    data = load_data(cache)
    if any(cfg[key] != data.meta['config'][key] for key in
           ('minimum_frequency', 'caption_length', 'history_length', 'word_limit')):
        raise ValueError('Representation feature preparation differs from training config')
    verify_cache(reference); verify_prepared(prepared, reference, cfg)
    protocol = {'config': cfg, 'cache_metadata_sha256': sha(cache/'metadata.json'),
                'reference_sha256': sha(reference/'metadata.json'), 'prepared_sha256': sha(prepared/'metadata.json'),
                'source_sha256': sha(__file__), 'model_source_sha256': sha(Path(__file__).with_name('representation_model.py')),
                'state_sha256': sha(CACHE/'state.npz'), 'test_used': False, 'default_changed': False}
    x = torch.from_numpy(normalize(np.load(prepared/'full_X.npy', mmap_mode='r'),
        mean := np.load(prepared/'full_mean.npy'), std := np.load(prepared/'full_std.npy'), list(range(28,34))))
    xv = torch.from_numpy(normalize(np.load(reference/'valid_X.npy', mmap_mode='r'), mean, std, list(range(28,34))))
    y = np.load(CACHE/'train_y.npy')
    u, i, uv, iv = [torch.from_numpy(np.load(CACHE/f'{split}_{name}.npy'))
                     for split, name in [('train','users'),('train','items'),('valid','users'),('valid','items')]]
    diag = ExposureDiagnostics(np.load(CACHE/'valid_y.npy'), uv.numpy(), np.load(CACHE/'valid_times.npy'), iv.numpy())
    if benchmark:
        torch.manual_seed(2026)
        m = RepresentationDCN(data.meta['cardinalities'], True, cfg['embedding_dim'])
        dense, sparse = dense_and_sparse(m)
        opts = [torch.optim.Adam(dense, lr=.0005), torch.optim.SparseAdam(sparse, lr=.0005)]
        target = torch.from_numpy(y.astype(np.float32)); rng = torch.Generator().manual_seed(2)
        start = perf_counter()
        for _ in range(20):
            ix = torch.randint(len(y), (cfg['batch_size'],), generator=rng)
            for opt in opts: opt.zero_grad(set_to_none=True)
            p = m(x[ix], data.user_lookup[u[ix]], data.item_fields[i[ix]], data.caption[i[ix]])
            F.binary_cross_entropy_with_logits(p, target[ix]).backward()
            for opt in opts: opt.step()
        seconds = (perf_counter()-start)/20
        print(json.dumps({'seconds_per_step_20_step_probe': seconds,
                          'approximate_train_seconds_per_epoch': seconds*((len(y)+cfg['batch_size']-1)//cfg['batch_size']),
                          'includes_optimizer_initialization': True}), flush=True)
        return
    frozen_assets()
    if resume:
        if json.loads((output/'protocol.json').read_text()) != protocol: raise ValueError('Study protocol changed')
    else:
        output.mkdir(parents=True, exist_ok=False); write_json(output/'protocol.json', protocol)
    records = {}
    for content in (False, True):
        for seed in cfg['seeds']:
            name = ('content' if content else 'id_attrs')+f'_s{seed}'
            dest = output/name
            records[name] = fit(content, seed, cfg, data, x, y, u, i, xv, uv, iv, diag, mean, std,
                dest, protocol, resume=resume and (dest/'last.pt').exists())
            write_json(output/'progress.json', records)
    if sha(__file__) != protocol['source_sha256']: raise ValueError('Trainer changed during run')
    frozen_assets()
    write_json(output/'metrics.json', {'protocol': protocol, 'runs': records, 'status': 'complete',
               'test_used': False, 'default_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    base = Path('artifacts/full_exposure_local_20261022')
    parser.add_argument('--cache', type=Path, default=Path('artifacts/representation_20261022/cache'))
    parser.add_argument('--reference', type=Path, default=base/'reference')
    parser.add_argument('--prepared', type=Path, default=base/'full_cache')
    parser.add_argument('--config', type=Path, default=Path('configs/representation_v1.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    run(args.cache, args.reference, args.prepared, args.config, args.out, args.seeds, args.resume, args.benchmark)
