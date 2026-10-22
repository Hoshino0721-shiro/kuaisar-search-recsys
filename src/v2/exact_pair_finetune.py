"""Matched three-arm warm-start fine-tuning on full BCE and exact-time pairs.

For each seed, all arms share epoch orders, pair draws, optimizer steps and LR.
The BCE control drives LR reductions. Stop only when all arms plateau. Best
validation pair loss is selected within exposure guardrails, including epoch0.
"""
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
from src.v2.exact_pair_data import verify as verify_pairs
from src.v2.full_exposure_control import verify_prepared, evaluate_one
from src.v2.models import FineDCN
from src.v2.ranking_convergence import ExposureDiagnostics, frozen_assets, save_checkpoint, write_json
from src.v2.route_attribution import Scorer, normalize, metric_arrays, summary, load_labels
from src.v2.temporal_route_train import verify_cache
from src.v2.train_rankers import sha


def objective_loss(logits, labels, positive, negative, objective, weight):
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    pair = F.softplus(negative-positive).mean()
    endpoint = (F.softplus(-positive).mean()+F.softplus(negative).mean())/2
    if objective not in ('bce', 'endpoint_aux', 'pair_aux'):
        raise ValueError('Unknown fine-tuning objective')
    auxiliary = endpoint if objective == 'endpoint_aux' else pair
    return bce+(0 if objective == 'bce' else weight)*auxiliary, bce, auxiliary


def predict_logits(model, x):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(x[lo:lo+65536]).numpy() for lo in range(0, len(x), 65536)])


def pair_metrics(logits, positive, negative):
    difference = logits[positive].astype(np.float64)-logits[negative].astype(np.float64)
    return {'pairs': len(difference), 'logloss': float(np.logaddexp(0., -difference).mean()),
            'accuracy_half_ties': float(np.mean((difference > 0)+.5*(difference == 0)))}


def eligible(valid, warm_valid, cfg):
    return (valid['logloss'] <= warm_valid['logloss']+cfg['logloss_guard']
            and valid['auc'] >= warm_valid['auc']-cfg['auc_guard'])


def reports(saved, destination):
    records = {}
    for objective in saved['protocol']['config']['objectives']:
        selected = saved['selected_epochs'][objective]
        metadata = {'kind': 'exact_pair_finetune_v1', 'columns': list(range(28, 34)),
                    'seed': saved['protocol']['seed'], 'objective': objective,
                    'selected_epoch': selected, 'warm_checkpoint_sha256': saved['protocol']['warm_sha256'],
                    'test_used': False, 'checkpoint_rule': saved['protocol']['config']['selection']}
        dest = destination/objective
        dest.mkdir(exist_ok=True)
        save_checkpoint(dest/'model.pt', {'state_dict': saved['best_states'][objective],
                        'mean': saved['mean'], 'std': saved['std'], 'metadata': metadata})
        record = {'metadata': metadata, 'history': saved['histories'][objective],
                  'selected_epoch': selected, 'epochs_completed': saved['epoch'],
                  'selected_validation': saved['histories'][objective][selected]['valid'],
                  'selected_pair': saved['histories'][objective][selected]['valid_pair'],
                  'optimizer_steps': saved['epoch']*saved['steps_per_epoch'],
                  'early_stopping_reason': saved['reason'], 'lr_reductions': saved['reductions'],
                  'checkpoint_sha256': sha(dest/'model.pt'),
                  'status': 'complete' if saved['finished'] else 'running', 'test_used': False}
        write_json(dest/'metrics.json', record)
        records[objective] = record
    write_json(destination/'metrics.json', {'protocol': saved['protocol'], 'runs': records,
               'epoch': saved['epoch'], 'status': 'complete' if saved['finished'] else 'running',
               'test_used': False})
    return records


def fit_seed(tx, y, vx, diagnostics, train_pairs, valid_pairs, warm_path, cfg, seed,
             destination, context, candidate_diagnostic=None, resume=False, stop_after=None):
    if set(cfg['objectives']) != {'bce', 'endpoint_aux', 'pair_aux'} or len(cfg['objectives']) != 3:
        raise ValueError('Three matched objectives required')
    if not 1 <= cfg['min_epochs'] <= cfg['max_epochs'] or cfg['aux_weight'] < 0:
        raise ValueError('Invalid budget or auxiliary weight')
    if min(cfg['batch_size'], cfg['pair_batch_size'], cfg['lr_patience'], cfg['early_patience']) < 1:
        raise ValueError('Invalid batch/stopping rule')
    protocol = {'config': cfg, 'seed': seed, 'context': context, 'warm_sha256': sha(warm_path),
                'source_sha256': sha(__file__), 'train_rows': len(y), 'test_used': False}
    torch.manual_seed(seed)
    warm = torch.load(warm_path, map_location='cpu', weights_only=True)
    if warm['metadata']['columns'] != list(range(28, 34)):
        raise ValueError('Expected R warm-start model')
    models = {key: FineDCN(tx.shape[1]) for key in cfg['objectives']}
    optimizers = {}
    for key, model in models.items():
        model.load_state_dict(warm['state_dict'])
        optimizers[key] = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    order_rng = torch.Generator().manual_seed(seed+1000)
    pair_rng = torch.Generator().manual_seed(seed+10000)
    tp, tn = [torch.as_tensor(a, dtype=torch.long) for a in train_pairs]
    vp, vn = valid_pairs
    if not len(tp) or len(tp) != len(tn) or not len(vp):
        raise ValueError('Observed train and validation pairs required')
    ty = torch.as_tensor(y, dtype=torch.float32)
    probe = np.sort(np.random.default_rng(2026).choice(len(y), min(65536, len(y)), replace=False))
    probe_x, probe_y = tx[probe], np.asarray(y)[probe]
    steps = (len(y)+cfg['batch_size']-1)//cfg['batch_size']
    def validation(model):
        logits = predict_logits(model, vx)
        p = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return diagnostics.metrics(p), pair_metrics(logits, vp, vn)
    initial_valid, initial_pair = validation(models['bce'])
    if resume:
        saved = torch.load(destination/'last.pt', map_location='cpu', weights_only=True)
        if saved['protocol'] != protocol:
            raise ValueError('Resume protocol/input/source mismatch')
        for key in models:
            models[key].load_state_dict(saved['states'][key])
            optimizers[key].load_state_dict(saved['optimizers'][key])
        order_rng.set_state(saved['order_rng'])
        pair_rng.set_state(saved['pair_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if saved['finished']:
            return reports(saved, destination)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination/'protocol.json', protocol)
        histories = {}
        for key in models:
            record = {'epoch': 0, 'valid': initial_valid, 'valid_pair': initial_pair,
                      'guardrail_pass': True, 'learning_rate': 0.}
            if candidate_diagnostic is not None:
                record['candidate_diagnostic'] = candidate_diagnostic(models[key])
            histories[key] = [record]
        saved = {'protocol': protocol, 'epoch': 0, 'histories': histories,
                 'selected_epochs': {k: 0 for k in models},
                 'best_states': {k: copy.deepcopy(m.state_dict()) for k, m in models.items()},
                 'best_pair': {k: initial_pair['logloss'] for k in models},
                 'plateau': {k: initial_pair['logloss'] for k in models},
                 'bad': {k: 0 for k in models}, 'lr_bad': 0, 'reductions': 0,
                 'mean': warm['mean'], 'std': warm['std'], 'steps_per_epoch': steps,
                 'finished': False, 'reason': 'running', 'elapsed_seconds': 0.}
    start = perf_counter()
    previous_elapsed = saved['elapsed_seconds']
    for epoch in range(saved['epoch']+1, cfg['max_epochs']+1):
        for model in models.values():
            model.train()
        order = torch.randperm(len(y), generator=order_rng)
        draws = torch.randint(len(tp), (steps, cfg['pair_batch_size']), generator=pair_rng)
        order_sha = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        draw_sha = hashlib.sha256(draws.numpy().tobytes()).hexdigest()
        total = {k: np.zeros(3) for k in models}
        lr = optimizers['bce'].param_groups[0]['lr']
        for step, lo in enumerate(range(0, len(y), cfg['batch_size'])):
            ix = order[lo:lo+cfg['batch_size']]
            p, n = tp[draws[step]], tn[draws[step]]
            values = tx[torch.cat([ix, p, n])]
            for key, model in models.items():
                logits = model(values)
                full, pos, neg = torch.split(logits, [len(ix), len(p), len(n)])
                loss, bce, aux = objective_loss(full, ty[ix], pos, neg, key, cfg['aux_weight'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite fine-tuning loss')
                optimizers[key].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[key].step()
                total[key] += np.array([loss.item(), bce.item(), aux.item()])*len(ix)
        for key, model in models.items():
            valid, pair = validation(model)
            allowed = eligible(valid, initial_valid, cfg)
            probe_logits = predict_logits(model, probe_x)
            probe_loss = float(np.mean(np.logaddexp(0., probe_logits.astype(np.float64))-probe_y*probe_logits))
            record = {'epoch': epoch, 'learning_rate': lr, 'order_sha256': order_sha, 'pair_draw_sha256': draw_sha,
                      'train_objective': float(total[key][0]/len(y)), 'train_bce': float(total[key][1]/len(y)),
                      'train_auxiliary': float(total[key][2]/len(y)), 'train_probe_logloss': probe_loss,
                      'valid': valid, 'valid_pair': pair, 'guardrail_pass': bool(allowed)}
            if candidate_diagnostic is not None:
                record['candidate_diagnostic'] = candidate_diagnostic(model)
            saved['histories'][key].append(record)
            if allowed and pair['logloss'] < saved['best_pair'][key]:
                saved['best_pair'][key] = pair['logloss']
                saved['selected_epochs'][key] = epoch
                saved['best_states'][key] = copy.deepcopy(model.state_dict())
            improved = allowed and pair['logloss'] < saved['plateau'][key]-cfg['min_delta']
            if improved:
                saved['plateau'][key] = pair['logloss']
                saved['bad'][key] = 0
            else:
                saved['bad'][key] += 1
            if key == 'bce':
                saved['lr_bad'] = 0 if improved else saved['lr_bad']+1
            print(f's{seed} {key} epoch {epoch}: LR={lr:.7f} pairLL={pair["logloss"]:.7f} '
                  f'LL={valid["logloss"]:.7f} GAUC={valid["gauc"]} guard={allowed} best={saved["selected_epochs"][key]}', flush=True)
        if saved['lr_bad'] >= cfg['lr_patience'] and lr > cfg['min_lr']:
            for optimizer in optimizers.values():
                optimizer.param_groups[0]['lr'] = max(cfg['min_lr'], lr*cfg['lr_factor'])
            saved['reductions'] += 1
            saved['lr_bad'] = 0
        saved['epoch'] = epoch
        if epoch >= cfg['min_epochs'] and min(saved['bad'].values()) >= cfg['early_patience'] and saved['reductions'] >= 2:
            saved['finished'] = True
            saved['reason'] = 'all_arms_validation_plateau_after_shared_lr_reductions'
        elif epoch == cfg['max_epochs']:
            saved['finished'] = True
            saved['reason'] = 'max_epochs_reached_not_proof_of_convergence'
        saved.update(states={k: m.state_dict() for k, m in models.items()},
                     optimizers={k: o.state_dict() for k, o in optimizers.items()},
                     order_rng=order_rng.get_state(), pair_rng=pair_rng.get_state(), torch_rng=torch.get_rng_state(),
                     elapsed_seconds=previous_elapsed+perf_counter()-start)
        # One atomic group commit; all selected inference files are recoverable from it.
        save_checkpoint(destination/'last.pt', saved)
        result = reports(saved, destination)
        if saved['finished'] or (stop_after is not None and epoch >= stop_after):
            break
    return result


def run(reference, prepared, pairs, warm, previous, config, output, resume=False):
    cfg = json.loads(config.read_text(encoding='utf-8'))
    torch.set_num_threads(cfg['threads'])
    meta = verify_cache(reference)
    verify_prepared(prepared, reference, cfg)
    pair_meta = verify_pairs(pairs)
    old = json.loads((warm/'metrics.json').read_text(encoding='utf-8'))
    if pair_meta['seed'] != cfg['pair_seed'] or old['status'] != 'complete' or old['test_used']:
        raise ValueError('Warm-start or pair protocol mismatch')
    if old['protocol']['feature_state_sha256'] != sha(CACHE/'state.npz'):
        raise ValueError('Warm-start feature state mismatch')
    protocol = {'config': cfg, 'state_sha256': sha(CACHE/'state.npz'),
                'pairs_metadata_sha256': sha(pairs/'metadata.json'), 'reference_sha256': sha(reference/'metadata.json'),
                'prepared_sha256': sha(prepared/'metadata.json'), 'warm_report_sha256': sha(warm/'metrics.json'),
                'source_sha256': sha(__file__), 'candidate_sha256': meta['candidate_sha256'],
                'dependencies': {name: sha(Path(__file__).with_name(name)) for name in
                    ('exact_pair_data.py', 'ranking_convergence.py', 'models.py', 'route_attribution.py', 'full_exposure_control.py')},
                'test_used': False, 'default_changed': False}
    count = frozen_assets()
    if resume:
        if json.loads((output/'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise ValueError('Study resume protocol mismatch')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'protocol.json', protocol)
    x = np.load(prepared/'full_X.npy', mmap_mode='r')
    xv = np.load(reference/'valid_X.npy', mmap_mode='r')
    y = np.load(CACHE/'train_y.npy')
    mean, std = [np.load(prepared/f'full_{name}.npy') for name in ('mean', 'std')]
    tx = torch.from_numpy(normalize(x, mean, std, list(range(28, 34))))
    vx = torch.from_numpy(normalize(xv, mean, std, list(range(28, 34))))
    diag = ExposureDiagnostics(np.load(CACHE/'valid_y.npy'), np.load(CACHE/'valid_users.npy'),
                               np.load(CACHE/'valid_times.npy'), np.load(CACHE/'valid_items.npy'))
    train_pairs = [np.load(pairs/f'train_{k}.npy') for k in ('positive', 'negative')]
    valid_pairs = [np.load(pairs/f'valid_{k}.npy') for k in ('positive', 'negative')]
    users = np.load(reference/'pool_users.npy')
    items = np.load(reference/'pool_items.npy', mmap_mode='r')
    pool = np.load(reference/'pool_X.npy', mmap_mode='r')
    labels, positive, _, _ = load_labels(users, items)
    chosen = np.load(warm/'diagnostic_user_positions.npy')[:cfg['diagnostic_users']]
    np.save(output/'diagnostic_user_positions.npy', chosen)
    px = torch.from_numpy(normalize(pool[chosen].reshape(-1, 34), mean, std, list(range(28, 34))))
    np.save(output/'pool_item_seen_sample.npy', np.load(previous/'pool_item_seen_sample.npy'))
    def candidate_diagnostic(model):
        p = torch.sigmoid(torch.from_numpy(predict_logits(model, px))).numpy().reshape(len(chosen), items.shape[1])
        return summary(*metric_arrays(items[chosen], p, [labels[j] for j in chosen]), int(positive[chosen].sum()))
    records = {}
    for seed in cfg['seeds']:
        warm_path = warm/'models'/f'full_R_s{seed}'/'model.pt'
        if sha(warm_path) != old['runs'][f'full_R_s{seed}']['training']['checkpoint_sha256']:
            raise ValueError('Warm checkpoint changed')
        checkpoint = torch.load(warm_path, map_location='cpu', weights_only=True)
        np.testing.assert_array_equal(checkpoint['mean'], mean)
        np.testing.assert_array_equal(checkpoint['std'], std)
        dest = output/'models'/f's{seed}'
        fits = fit_seed(tx, y, vx, diag, train_pairs, valid_pairs, warm_path, cfg, seed, dest,
                        protocol, candidate_diagnostic, resume=resume and (dest/'last.pt').exists())
        for objective, fit in fits.items():
            name = f'{objective}_s{seed}'
            scorer = Scorer(dest/objective/'model.pt')
            prediction = scorer.predict(xv)
            np.save(dest/objective/'valid_predictions.npy', prediction)
            record, _ = evaluate_one(name, scorer, fit, reference, output, labels, positive, meta['config'])
            records[name] = record
            write_json(output/'progress.json', records)
    aggregates = {}
    for objective in cfg['objectives']:
        group = [records[f'{objective}_s{s}'] for s in cfg['seeds']]
        aggregates[objective] = {'selected_epochs': [r['training']['selected_epoch'] for r in group],
            'validation': {k: float(np.mean([r['training']['selected_validation'][k] for r in group]))
                           for k in ('auc', 'logloss', 'gauc', 'brier')},
            'pair_logloss': float(np.mean([r['training']['selected_pair']['logloss'] for r in group])),
            'direct_hits': float(np.mean([r['direct']['hit_events'] for r in group])),
            'direct_ndcg': float(np.mean([r['direct']['ndcg_at20'] for r in group])),
            'cascade_hits': float(np.mean([r['cascade']['hit_events'] for r in group]))}
    if frozen_assets() != count or sha(__file__) != protocol['source_sha256']:
        raise ValueError('Frozen assets or training source changed')
    write_json(output/'metrics.json', {'protocol': protocol, 'runs': records, 'aggregate': aggregates,
               'warm_aggregate': old['aggregate']['R'], 'status': 'complete', 'test_used': False,
               'default_changed': False, 'frozen_assets_verified': count})
    for array in (x, xv, items, pool):
        array._mmap.close()
    print('Exact-timestamp fine-tuning complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    base = Path('artifacts/full_exposure_local_20261022')
    parser.add_argument('--reference', type=Path, default=base/'reference')
    parser.add_argument('--prepared', type=Path, default=base/'full_cache')
    parser.add_argument('--pairs', type=Path, default=Path('artifacts/exact_pair_finetune_20261022/pairs'))
    parser.add_argument('--warm', type=Path, default=Path('artifacts/ranking_convergence_20261022'))
    parser.add_argument('--previous', type=Path, default=base/'study')
    parser.add_argument('--config', type=Path, default=Path('configs/exact_pair_finetune.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.reference, args.prepared, args.pairs, args.warm, args.previous, args.config, args.out, args.resume)
