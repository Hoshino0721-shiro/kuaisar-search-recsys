"""Frozen learned content representation + mean/DIN-inspired residual BCE."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from src.ctr_features import CACHE
from src.v2.ranking_convergence import ExposureDiagnostics, frozen_assets, save_checkpoint, write_json
from src.v2.representation_model import InterestResidual, RepresentationScorer
from src.v2.representation_train import load_data
from src.v2.full_exposure_control import verify_prepared
from src.v2.temporal_route_train import verify_cache
from src.v2.route_attribution import normalize
from src.v2.train_rankers import sha


def freeze_items(scorer, data, output):
    context = {'checkpoint_sha256': sha(scorer.checkpoint), 'data_sha256': sha(data.path/'metadata.json')}
    if output.exists():
        meta = json.loads((output/'metadata.json').read_text())
        if meta['context'] != context or sha(output/'items.npy') != meta['items_sha256']:
            raise ValueError('Frozen item representations changed')
    else:
        output.mkdir(parents=True)
        a = np.lib.format.open_memmap(output/'items.npy', mode='w+', dtype=np.float32,
            shape=(len(data.item_fields), scorer.model.dimension))
        with torch.inference_mode():
            for lo in range(0, len(a), 8192):
                a[lo:lo+8192] = scorer.model.item_vector(data.item_fields[lo:lo+8192], data.caption[lo:lo+8192]).numpy()
        a[0] = 0.; a.flush(); a._mmap.close()
        write_json(output/'metadata.json', {'context': context, 'items_sha256': sha(output/'items.npy'),
                   'test_used': False, 'source_sha256': sha(__file__)})
    return torch.from_numpy(np.load(output/'items.npy'))


class FrozenContent:
    def __init__(self, checkpoint, data, vectors):
        self.scorer = RepresentationScorer(checkpoint, data)
        self.scorer.model.requires_grad_(False)
        self.data = data
        self.vectors = vectors

    def logits(self, numeric, users, items):
        with torch.inference_mode():
            return self.scorer.model.core(torch.cat([numeric,
                self.scorer.model.embed('user',self.data.user_lookup[users]), self.vectors[items]], dim=1))


def probabilities(model, x, base, candidates, histories, mask, contexts):
    model.eval(); pieces = []
    with torch.inference_mode():
        for lo in range(0, len(base), 8192):
            sl = slice(lo, lo+8192); c = contexts[sl]
            pieces.append(model(x[sl], base[sl], candidates[sl], histories[c], mask[c]).numpy())
    return torch.sigmoid(torch.from_numpy(np.concatenate(pieces))).numpy()


def fit(attention, seed, cfg, x, y, candidates, histories, mask, contexts, base,
        xv, cv, hv, mv, uv, bv, diag, output, context, resume=False, stop_after=None):
    protocol = {'config': cfg, 'attention': attention, 'seed': seed, 'context': context,
                'source_sha256': sha(__file__), 'test_used': False}
    torch.manual_seed(seed)
    model = InterestResidual(cfg['embedding_dim'], attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(seed+2000)
    ty = torch.from_numpy(np.asarray(y, np.float32))
    if resume:
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if saved['protocol'] != protocol: raise ValueError('Interest resume protocol mismatch')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator']); torch.set_rng_state(saved['rng'])
        if saved['finished']: return materialize(saved, output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        p = probabilities(model, xv, bv, cv, hv, mv, uv)
        np.testing.assert_array_max_ulp(p, torch.sigmoid(bv).numpy(), maxulp=2)
        valid = diag.metrics(p)
        saved = {'protocol': protocol, 'history': [{'epoch': 0, 'valid': valid, 'learning_rate': 0.}],
                 'best_epoch': 0, 'best': valid['logloss'], 'plateau': valid['logloss'],
                 'best_state': copy.deepcopy(model.state_dict()), 'bad': 0, 'lr_bad': 0, 'reductions': 0,
                 'finished': False, 'reason': 'running', 'elapsed_seconds': 0.}
    started = perf_counter(); elapsed = saved['elapsed_seconds']
    probe = np.sort(np.random.default_rng(2026).choice(len(y), min(65536, len(y)), replace=False))
    for epoch in range(len(saved['history']), cfg['max_epochs']+1):
        model.train(); order = torch.randperm(len(y), generator=generator); total = 0.
        lr = optimizer.param_groups[0]['lr']
        for lo in range(0, len(y), cfg['batch_size']):
            ix = order[lo:lo+cfg['batch_size']]; c = contexts[ix]
            logits = model(x[ix], base[ix], candidates[ix], histories[c], mask[c])
            loss = F.binary_cross_entropy_with_logits(logits, ty[ix])
            if not torch.isfinite(loss): raise ValueError('Nonfinite interest loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += loss.item()*len(ix)
        p = probabilities(model, xv, bv, cv, hv, mv, uv); valid = diag.metrics(p)
        probe_p = np.clip(probabilities(model, x[probe], base[probe], candidates[probe], histories, mask, contexts[probe]).astype(np.float64), 1e-7, 1-1e-7)
        probe_ll = -np.mean(y[probe]*np.log(probe_p)+(1-y[probe])*np.log1p(-probe_p))
        saved['history'].append({'epoch': epoch, 'learning_rate': lr, 'train_logloss': total/len(y),
            'train_probe_logloss': float(probe_ll), 'valid': valid,
            'order_sha256': hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        if valid['logloss'] < saved['best']:
            saved['best'] = valid['logloss']; saved['best_epoch'] = epoch
            saved['best_state'] = copy.deepcopy(model.state_dict())
        if valid['logloss'] < saved['plateau']-cfg['min_delta']:
            saved['plateau'] = valid['logloss']; saved['bad'] = saved['lr_bad'] = 0
        else: saved['bad'] += 1; saved['lr_bad'] += 1
        if saved['lr_bad'] >= cfg['lr_patience']:
            optimizer.param_groups[0]['lr'] *= cfg['lr_factor']; saved['reductions'] += 1; saved['lr_bad'] = 0
        plateau_stop = epoch >= cfg['min_epochs'] and saved['bad'] >= cfg['early_patience'] and saved['reductions'] >= 2
        saved['finished'] = plateau_stop or epoch == cfg['max_epochs']
        saved['reason'] = 'validation_plateau' if plateau_stop else ('max_epochs_not_proof_of_convergence' if saved['finished'] else 'running')
        saved.update(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                     generator=generator.get_state(), rng=torch.get_rng_state(), elapsed_seconds=elapsed+perf_counter()-started)
        save_checkpoint(output/'last.pt', saved); result = materialize(saved, output)
        print(f'{output.name} epoch {epoch}: AUC={valid["auc"]:.6f} LL={valid["logloss"]:.6f} '
              f'GAUC={valid["gauc"]:.6f} best={saved["best_epoch"]}', flush=True)
        if saved['finished'] or (stop_after is not None and epoch >= stop_after): break
    return result


def materialize(saved, output):
    meta = {'kind': 'frozen_content_interest_v1', 'attention': saved['protocol']['attention'],
            'dimension': saved['protocol']['config']['embedding_dim'], 'seed': saved['protocol']['seed'],
            'selected_epoch': saved['best_epoch'], 'test_used': False}
    save_checkpoint(output/'model.pt', {'state_dict': saved['best_state'], 'metadata': meta})
    result = {'metadata': meta, 'protocol': saved['protocol'], 'history': saved['history'],
              'selected_epoch': saved['best_epoch'], 'selected_validation': saved['history'][saved['best_epoch']]['valid'],
              'epochs_completed': len(saved['history'])-1, 'early_stopping_reason': saved['reason'],
              'checkpoint_sha256': sha(output/'model.pt'), 'elapsed_seconds': saved['elapsed_seconds'],
              'test_used': False, 'status': 'complete' if saved['finished'] else 'running'}
    write_json(output/'metrics.json', result)
    return result


def run(cache, content_models, prepared, reference, config, output, resume=False):
    cfg = json.loads(config.read_text()); torch.set_num_threads(cfg['threads'])
    data = load_data(cache)
    source = json.loads((content_models/'metrics.json').read_text())
    if source['status'] != 'complete' or source['test_used']: raise ValueError('Complete content training required')
    verify_cache(reference); verify_prepared(prepared,reference,cfg)
    if (source['protocol']['reference_sha256']!=sha(reference/'metadata.json') or
            source['protocol']['prepared_sha256']!=sha(prepared/'metadata.json') or
            source['protocol']['cache_metadata_sha256']!=sha(cache/'metadata.json') or
            source['protocol']['model_source_sha256']!=sha(Path(__file__).with_name('representation_model.py'))):
        raise ValueError('Frozen content/data provenance changed')
    protocol = {'config': cfg, 'content_report_sha256': sha(content_models/'metrics.json'),
                'cache_sha256': sha(cache/'metadata.json'), 'source_sha256': sha(__file__),
                'model_sha256': sha(Path(__file__).with_name('representation_model.py')), 'test_used': False,
                'default_changed': False}
    if resume:
        if json.loads((output/'protocol.json').read_text()) != protocol: raise ValueError('Interest study mismatch')
    else: output.mkdir(parents=True, exist_ok=False); write_json(output/'protocol.json', protocol)
    frozen_assets()
    mean, std = [np.load(prepared/f'full_{k}.npy') for k in ('mean','std')]
    x = torch.from_numpy(normalize(np.load(prepared/'full_X.npy', mmap_mode='r'), mean, std, list(range(28,34))))
    xv = torch.from_numpy(normalize(np.load(reference/'valid_X.npy', mmap_mode='r'), mean, std, list(range(28,34))))
    y = np.load(CACHE/'train_y.npy')
    u,i,uv,iv = [torch.from_numpy(np.load(CACHE/f'{split}_{key}.npy')).long()
        for split,key in [('train','users'),('train','items'),('valid','users'),('valid','items')]]
    diag = ExposureDiagnostics(np.load(CACHE/'valid_y.npy'), uv.numpy(), np.load(CACHE/'valid_times.npy'), iv.numpy())
    records = {}
    for seed in cfg['seeds']:
        checkpoint = content_models/f'content_s{seed}'/'model.pt'
        if sha(checkpoint) != source['runs'][f'content_s{seed}']['checkpoint_sha256']: raise ValueError('Content checkpoint changed')
        scorer = RepresentationScorer(checkpoint, data); scorer.checkpoint = checkpoint
        vectors = freeze_items(scorer, data, output/'frozen'/f's{seed}')
        frozen = FrozenContent(checkpoint, data, vectors)
        def base_logits(xx, uu, ii):
            return torch.cat([frozen.logits(xx[lo:lo+8192], uu[lo:lo+8192], ii[lo:lo+8192])
                              for lo in range(0, len(ii), 8192)])
        b, bv = base_logits(x,u,i), base_logits(xv,uv,iv)
        candidate, cv = vectors[i], vectors[iv]
        cr, qr = data.context_rec.long(), data.query_rec.long()
        h, hv = vectors[cr], vectors[qr]; mask, mv = cr.ne(0), qr.ne(0)
        initial = diag.metrics(torch.sigmoid(bv).numpy())
        # Snapshot reconstruction may change float32 grouping, not the model.
        np.testing.assert_allclose(initial['logloss'], source['runs'][f'content_s{seed}']['selected_validation']['logloss'], atol=1e-8, rtol=0)
        context = {**protocol, 'frozen_sha256': sha(output/'frozen'/f's{seed}'/'metadata.json'),
                   'content_checkpoint_sha256': sha(checkpoint)}
        for attention in (False, True):
            name = ('din' if attention else 'mean')+f'_s{seed}'; dest = output/name
            records[name] = fit(attention, seed, cfg, x,y,candidate,h,mask,data.train_context.long(),b,
                xv,cv,hv,mv,uv,bv,diag,dest,context,resume=resume and (dest/'last.pt').exists())
            write_json(output/'progress.json', records)
        del vectors, frozen, scorer, candidate,cv,h,hv,b,bv
    if sha(__file__) != protocol['source_sha256']: raise ValueError('Interest source changed during run')
    frozen_assets()
    write_json(output/'metrics.json', {'protocol': protocol, 'runs': records, 'status': 'complete',
               'test_used': False, 'default_changed': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); base = Path('artifacts/representation_20261022')
    p.add_argument('--cache', type=Path, default=base/'cache')
    p.add_argument('--content-models', type=Path, default=base/'models_safe')
    p.add_argument('--prepared', type=Path, default=Path('artifacts/full_exposure_local_20261022/full_cache'))
    p.add_argument('--reference', type=Path, default=Path('artifacts/full_exposure_local_20261022/reference'))
    p.add_argument('--config', type=Path, default=Path('configs/representation_v1.json'))
    p.add_argument('--out', type=Path, required=True); p.add_argument('--resume', action='store_true')
    a=p.parse_args();run(a.cache,a.content_models,a.prepared,a.reference,a.config,a.out,a.resume)
