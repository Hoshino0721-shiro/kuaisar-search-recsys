"""Epoch-wise, validation-selected full-exposure BCE with exact epoch resume.

No change to the old fixed-five-epoch experiment or serving default. The first
five epochs retain the old initialization/order/LR. Scheduler starts after five.
Candidate coverage and same-timestamp NDCG are diagnostics, never selectors.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.ctr_features import CACHE
from src.v2.full_exposure_control import verify_prepared, evaluate_one
from src.v2.models import FineDCN
from src.v2.route_attribution import Scorer, normalize, metric_arrays, summary, load_labels
from src.v2.temporal_route_train import verify_cache
from src.v2.train_rankers import probability_metrics, sha


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                         encoding='utf-8', newline='\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


class ExposureDiagnostics:
    """Tie-aware row-weighted user AUC; binary exposure NDCG within timestamp.

    Timestamp groups are distribution batches, not verified request IDs.
    Repeated exposure rows are retained, not interpreted as unique candidates.
    """
    def __init__(self, y, users, times, items):
        self.y = np.asarray(y, dtype=np.int8)
        self.users = np.asarray(users)
        self.items = np.asarray(items)
        if not len(y) or any(len(a) != len(y) for a in (users, times, items)):
            raise ValueError('Invalid validation alignment')
        if not np.isin(y, [0, 1]).all():
            raise ValueError('Observed binary exposure labels required')
        _, self.user_groups = np.unique(users, return_inverse=True)
        user_n = np.bincount(self.user_groups)
        user_pos = np.bincount(self.user_groups, weights=self.y)
        self.user_pos = user_pos
        self.user_neg = user_n-user_pos
        self.user_n = user_n
        self.eligible = (user_pos > 0) & (self.user_neg > 0)
        order = np.lexsort((times, users))
        starts = np.r_[0, np.flatnonzero((np.diff(np.asarray(users)[order]) != 0)
                                         | (np.diff(np.asarray(times)[order]) != 0))+1]
        group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(y)]))
        self.time_groups = np.empty(len(y), dtype=np.int64)
        self.time_groups[order] = group
        self.time_n = np.bincount(self.time_groups)
        self.time_pos = np.bincount(self.time_groups, weights=self.y).astype(np.int64)
        self.time_eligible = (self.time_pos > 0) & (self.time_pos < self.time_n)
        self.discount = 1/np.log2(np.arange(20)+2)
        ideal = np.r_[0., np.cumsum(self.discount)]
        self.idcg = ideal[np.minimum(self.time_pos, 20)]

    def metrics(self, p):
        p = np.asarray(p)
        result = probability_metrics(self.y, p)
        order = np.lexsort((p, self.user_groups))
        u = self.user_groups[order]
        values = p[order]
        starts = np.r_[0, np.flatnonzero((np.diff(u) != 0) | (np.diff(values) != 0))+1]
        n = np.diff(np.r_[starts, len(p)])
        pos = np.add.reduceat(self.y[order].astype(np.int64), starts)
        neg = n-pos
        below = np.cumsum(neg)-neg
        tie_u = u[starts]
        user_starts = np.r_[0, np.flatnonzero(np.diff(tie_u))+1]
        baseline = np.repeat(below[user_starts], np.diff(np.r_[user_starts, len(starts)]))
        wins = np.bincount(tie_u, weights=pos*(below-baseline+0.5*neg),
                           minlength=len(self.user_n))
        eligible = self.eligible
        auc = wins[eligible]/(self.user_pos[eligible]*self.user_neg[eligible])
        result.update(gauc=float(np.average(auc, weights=self.user_n[eligible])) if eligible.any() else None,
                      gauc_groups=int(eligible.sum()), gauc_effective_rows=int(self.user_n[eligible].sum()),
                      brier=float(np.mean((p.astype(np.float64)-self.y)**2)))
        order = np.lexsort((np.arange(len(p)), self.items, -p, self.time_groups))
        groups = self.time_groups[order]
        starts = np.r_[0, np.flatnonzero(np.diff(groups))+1]
        ranks = np.arange(len(p))-np.repeat(starts, np.diff(np.r_[starts, len(p)]))
        keep = ranks < 20
        dcg = np.bincount(groups[keep], weights=self.y[order[keep]]*self.discount[ranks[keep]],
                          minlength=len(self.time_n))
        mixed = self.time_eligible
        result['same_timestamp'] = {
            'mixed_groups': int(mixed.sum()), 'mixed_rows': int(self.time_n[mixed].sum()),
            'ndcg_at20': float(np.mean(dcg[mixed]/self.idcg[mixed])) if mixed.any() else None,
            'scope': 'macro mixed user-distribution-timestamp exposure-row groups; not verified requests'}
        return result


def predict(model, x):
    model.eval()
    pieces = []
    with torch.inference_mode():
        for lo in range(0, len(x), 65536):
            pieces.append(torch.sigmoid(model(x[lo:lo+65536])).numpy())
    return np.concatenate(pieces)


def fit(x, y, xv, diagnostics, mean, std, columns, seed, cfg, destination,
        context, diagnostic=None, resume=False, stop_after=None):
    """Resume only from last completed epoch, including optimizer/LR/RNG state."""
    destination = Path(destination)
    if min(cfg['max_epochs'], cfg['batch_size'], cfg['early_patience'], cfg['lr_patience']) < 1:
        raise ValueError('Positive training budgets required')
    if not (5 <= cfg['min_epochs'] <= cfg['max_epochs']) or cfg['min_delta'] < 0:
        raise ValueError('Invalid stopping rule')
    if not (0 < cfg['lr_factor'] < 1 and 0 < cfg['min_lr'] <= cfg['learning_rate']):
        raise ValueError('Invalid LR schedule')
    if len(x) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid observed training labels')
    protocol = {'config': cfg, 'seed': seed, 'columns': columns, 'context': context,
                'train_rows': len(y), 'valid_rows': len(diagnostics.y),
                'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest(),
                'trainer_source_sha256': sha(__file__),
                'model_source_sha256': sha(Path(__file__).with_name('models.py'))}
    tx = torch.from_numpy(normalize(x, mean, std, columns))
    vx = torch.from_numpy(normalize(xv, mean, std, columns))
    ty = torch.from_numpy(np.asarray(y, dtype=np.float32))
    if not torch.isfinite(tx).all() or not torch.isfinite(vx).all():
        raise ValueError('Nonfinite normalized features')
    probe = np.sort(np.random.default_rng(2026).choice(len(y), min(65536, len(y)), replace=False))
    probe_x = tx[probe]
    probe_y = np.asarray(y)[probe]
    protocol['train_probe_indices_sha256'] = hashlib.sha256(probe.tobytes()).hexdigest()
    torch.manual_seed(seed)
    model = FineDCN(tx.shape[1])
    initial = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history = []
    best = float('inf')
    plateau_best = float('inf')
    bad = lr_bad = reductions = 0
    elapsed = 0.
    selected_epoch = 0
    if resume:
        saved = torch.load(destination/'last.pt', map_location='cpu', weights_only=True)
        if saved['protocol'] != protocol:
            raise ValueError('Resume protocol/data/source mismatch')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        torch.set_rng_state(saved['rng_state'])
        history = saved['history']
        best, plateau_best = saved['best'], saved['plateau_best']
        bad, lr_bad, reductions = saved['bad'], saved['lr_bad'], saved['reductions']
        selected_epoch, elapsed = saved['selected_epoch'], saved['elapsed_seconds']
        if not (destination/'model.pt').is_file():
            raise ValueError('Selected checkpoint missing')
        if sha(destination/'model.pt') != saved['best_checkpoint_sha256']:
            raise ValueError('Selected checkpoint differs from last committed epoch')
        if saved['finished']:
            result = make_result(saved)
            write_json(destination/'metrics.json', result)
            return result
    else:
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination/'protocol.json', protocol)
    started = perf_counter()
    finished = False
    reason = 'interrupted_at_epoch_boundary'
    for epoch in range(len(history)+1, cfg['max_epochs']+1):
        model.train()
        order = torch.randperm(len(tx))
        order_sha = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        total = 0.
        lr = optimizer.param_groups[0]['lr']
        for lo in range(0, len(tx), cfg['batch_size']):
            ix = order[lo:lo+cfg['batch_size']]
            loss = nn.functional.binary_cross_entropy_with_logits(model(tx[ix]), ty[ix])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(ix)
        p = predict(model, vx)
        valid = diagnostics.metrics(p)
        score = valid['logloss']
        probe_p = np.clip(predict(model, probe_x).astype(np.float64), 1e-7, 1-1e-7)
        record = {'epoch': epoch, 'learning_rate': lr, 'train_online_logloss': total/len(tx),
                  'train_probe_logloss': float(-np.mean(probe_y*np.log(probe_p)+(1-probe_y)*np.log1p(-probe_p))),
                  'train_probe_rows': len(probe),
                  'valid': valid, 'order_sha256': order_sha}
        if diagnostic is not None:
            record['candidate_diagnostic'] = diagnostic(model)
        history.append(record)
        if score < best:
            best = score
            selected_epoch = epoch
        if score < plateau_best-cfg['min_delta']:
            plateau_best = score
            bad = lr_bad = 0
        else:
            bad += 1
            lr_bad += 1
        # Epochs 1..5 exactly reproduce the fixed-five-epoch comparator.
        if epoch >= 5 and lr_bad >= cfg['lr_patience'] and lr > cfg['min_lr']:
            optimizer.param_groups[0]['lr'] = max(cfg['min_lr'], lr*cfg['lr_factor'])
            reductions += 1
            lr_bad = 0
        if epoch >= cfg['min_epochs'] and bad >= cfg['early_patience'] and reductions >= 2:
            finished = True
            reason = 'validation_plateau_after_lr_reductions'
        elif epoch == cfg['max_epochs']:
            finished = True
            reason = 'max_epochs_reached_not_proof_of_convergence'
        meta = {'kind': 'convergence_dcn_v1', 'columns': columns, 'seed': seed,
                'feature_state_sha256': context.get('feature_state_sha256'),
                'selected_epoch': selected_epoch, 'checkpoint_epoch': epoch,
                'checkpoint_rule': cfg['selection'], 'initial_parameters_sha256': initial,
                'test_used': False}
        saved = {'state_dict': model.state_dict(), 'mean': mean.tolist(), 'std': std.tolist(),
                 'metadata': meta, 'protocol': protocol, 'optimizer_state_dict': optimizer.state_dict(),
                 'rng_state': torch.get_rng_state(), 'history': history, 'best': best,
                 'plateau_best': plateau_best, 'bad': bad, 'lr_bad': lr_bad, 'reductions': reductions,
                 'selected_epoch': selected_epoch, 'elapsed_seconds': elapsed+perf_counter()-started,
                 'finished': finished, 'early_stopping_reason': reason}
        if selected_epoch == epoch:
            save_checkpoint(destination/'model.pt', saved)
        if epoch == 5:
            save_checkpoint(destination/'epoch5.pt', saved)
        saved['best_checkpoint_sha256'] = sha(destination/'model.pt')
        save_checkpoint(destination/'last.pt', saved)
        result = make_result(saved)
        write_json(destination/'metrics.json', result)
        print(f'{destination.name} epoch {epoch}: lr={lr:.7f} trainLL={total/len(tx):.7f} '
              f'validLL={score:.7f} AUC={valid["auc"]:.7f} GAUC={valid["gauc"]} best={selected_epoch}', flush=True)
        if finished or (stop_after is not None and epoch >= stop_after):
            break
    return result


def make_result(saved):
    selected = saved['selected_epoch']
    return {'protocol': saved['protocol'], 'history': saved['history'], 'selected_epoch': selected,
            'epochs_completed': len(saved['history']), 'early_stopping_reason': saved['early_stopping_reason'],
            'lr_reductions': saved['reductions'], 'selected_validation': saved['history'][selected-1]['valid'],
            'checkpoint_sha256': saved['best_checkpoint_sha256'],
            'status': 'complete' if saved['finished'] else 'running', 'test_used': False,
            'elapsed_seconds': saved['elapsed_seconds']}


def frozen_assets():
    freeze = json.loads(Path('experiments/final_test/freeze.json').read_text(encoding='utf-8'))
    for name, expected in freeze['asset_hashes'].items():
        if sha(name) != expected:
            raise ValueError('Original frozen asset changed: '+name)
    return len(freeze['asset_hashes'])


def run(reference, prepared, previous, config, output, resume=False):
    cfg = json.loads(config.read_text(encoding='utf-8'))
    torch.set_num_threads(cfg['threads'])
    frozen_count = frozen_assets()
    meta = verify_cache(reference)
    ready = verify_prepared(prepared, reference, cfg)
    old = json.loads((previous/'metrics.json').read_text(encoding='utf-8'))
    if old['status'] != 'complete' or old['protocol']['feature_state_sha256'] != ready['feature_state_sha256']:
        raise ValueError('Fixed-five local comparator missing or wrong state')
    protocol = {'config': cfg, 'feature_state_sha256': ready['feature_state_sha256'],
                'reference_metadata_sha256': sha(reference/'metadata.json'),
                'prepared_metadata_sha256': sha(prepared/'metadata.json'),
                'previous_metrics_sha256': sha(previous/'metrics.json'),
                'source_sha256': sha(__file__), 'candidate_sha256': meta['candidate_sha256'],
                'test_used': False, 'default_changed': False}
    if resume:
        if json.loads((output/'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise ValueError('Study resume protocol mismatch')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'protocol.json', protocol)
    x = np.load(prepared/'full_X.npy', mmap_mode='r')
    y = np.load(CACHE/'train_y.npy')
    xv = np.load(reference/'valid_X.npy', mmap_mode='r')
    yv = np.load(CACHE/'valid_y.npy')
    users = np.load(reference/'pool_users.npy')
    items = np.load(reference/'pool_items.npy', mmap_mode='r')
    pool = np.load(reference/'pool_X.npy', mmap_mode='r')
    labels, positive, _, _ = load_labels(users, items)
    diag = ExposureDiagnostics(yv, np.load(CACHE/'valid_users.npy'),
                                np.load(CACHE/'valid_times.npy'), np.load(CACHE/'valid_items.npy'))
    mean = np.load(prepared/'full_mean.npy')
    std = np.load(prepared/'full_std.npy')
    chosen = np.sort(np.random.default_rng(cfg['diagnostic_seed']).choice(
        len(users), min(cfg['diagnostic_users'], len(users)), replace=False))
    np.save(output/'diagnostic_user_positions.npy', chosen)
    np.save(output/'pool_item_seen_sample.npy', np.load(previous/'pool_item_seen_sample.npy'))
    records = {}
    for arm, columns in cfg['arms'].items():
        px = torch.from_numpy(normalize(pool[chosen].reshape(-1, 34), mean, std, columns))
        def candidate_diagnostic(model):
            predictions = predict(model, px).reshape(len(chosen), items.shape[1])
            h, d = metric_arrays(items[chosen], predictions, [labels[j] for j in chosen])
            return summary(h, d, int(positive[chosen].sum()))
        for seed in cfg['seeds']:
            name = f'full_{arm}_s{seed}'
            dest = output/'models'/name
            train = fit(x, y, xv, diag, mean, std, columns, seed, cfg, dest, protocol,
                        candidate_diagnostic, resume=resume and (dest/'last.pt').exists())
            baseline = old['runs'][name]['training']
            checkpoint5 = torch.load(dest/'epoch5.pt', map_location='cpu', weights_only=True)
            if checkpoint5['metadata']['initial_parameters_sha256'] != baseline['metadata']['initial_parameters_sha256']:
                raise ValueError('Initialization differs from fixed-five comparator')
            for actual, expected in zip(train['history'][:5], baseline['history']):
                if abs(actual['train_online_logloss']-expected['train_logloss']) > 1e-8:
                    raise ValueError('First-five trajectory failed exact reproduction')
            np.testing.assert_allclose(train['history'][4]['valid']['logloss'], baseline['exposure']['logloss'], atol=1e-8, rtol=0)
            a = Scorer(dest/'epoch5.pt').model.state_dict()
            b = Scorer(previous/'models'/name/'model.pt').model.state_dict()
            if not all(torch.equal(a[k], b[k]) for k in a):
                raise ValueError('Epoch-five weights differ')
            scorer = Scorer(dest/'model.pt')
            p = scorer.predict(xv)
            valid = diag.metrics(p)
            np.testing.assert_allclose(valid['logloss'], train['selected_validation']['logloss'], atol=1e-10)
            np.save(dest/'valid_predictions.npy', p)
            record, _ = evaluate_one(name, scorer, train, reference, output, labels, positive, meta['config'])
            record['selected_validation'] = valid
            record['epoch5_weights_bitwise_match'] = True
            record['fixed5_comparator'] = {k: old['runs'][name][k] for k in ('direct', 'cascade')}
            records[name] = record
            write_json(output/'progress.json', records)
    if frozen_assets() != frozen_count or sha(__file__) != protocol['source_sha256']:
        raise ValueError('Frozen assets or trainer changed during run')
    aggregate = {}
    for arm in cfg['arms']:
        names = [f'full_{arm}_s{s}' for s in cfg['seeds']]
        aggregate[arm] = {'selected_epochs': [records[n]['training']['selected_epoch'] for n in names],
                          'stopped_epochs': [records[n]['training']['epochs_completed'] for n in names],
                          'validation': {k: float(np.mean([records[n]['selected_validation'][k] for n in names]))
                                         for k in ('auc', 'logloss', 'gauc', 'brier')},
                          'direct_hits': float(np.mean([records[n]['direct']['hit_events'] for n in names])),
                          'direct_ndcg': float(np.mean([records[n]['direct']['ndcg_at20'] for n in names])),
                          'cascade_hits': float(np.mean([records[n]['cascade']['hit_events'] for n in names]))}
    write_json(output/'metrics.json', {'protocol': protocol, 'runs': records, 'aggregate': aggregate,
               'status': 'complete', 'frozen_assets_verified': frozen_count, 'test_used': False,
               'default_changed': False, 'baselines': old['baselines']})
    for array in (x, xv, items, pool):
        array._mmap.close()
    print('Convergence study complete; default unchanged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    root = Path('artifacts/full_exposure_local_20261022')
    parser.add_argument('--reference', type=Path, default=root/'reference')
    parser.add_argument('--prepared', type=Path, default=root/'full_cache')
    parser.add_argument('--previous', type=Path, default=root/'study')
    parser.add_argument('--config', type=Path, default=Path('configs/ranking_convergence.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.reference, args.prepared, args.previous, args.config, args.out, args.resume)
