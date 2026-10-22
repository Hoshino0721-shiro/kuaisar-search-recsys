"""Independent exact-pair SQL, matched-training, metrics and executor audit."""
import argparse
from collections import defaultdict
import json
from math import log2
from pathlib import Path

import numpy as np
import torch

from src.ctr_features import CACHE, FeatureState, connection
from src.ranking import metrics as exposure_metrics
from src.v2.cascade import Budget, Candidate, Cascade
from src.v2.exact_pair_data import verify as verify_pairs
from src.v2.ranking_convergence import write_json, frozen_assets
from src.v2.route_attribution import Scorer, metric_arrays, load_labels
from src.v2.train_rankers import sha


def audit_pairs(pairs):
    meta = verify_pairs(pairs)
    reports = {}
    db = connection()
    rng = np.random.default_rng(2026)
    limits = []
    try:
        for split in ('train', 'valid'):
            u, i, y = [np.load(CACHE/f'{split}_{k}.npy') for k in ('users', 'items', 'y')]
            t = np.load(pairs/'train_times.npy') if split == 'train' else np.load(CACHE/'valid_times.npy')
            p, n = [np.load(pairs/f'{split}_{k}.npy') for k in ('positive', 'negative')]
            if not len(p) or len(p) != len(n) or min(p.min(), n.min()) < 0 or max(p.max(), n.max()) >= len(y):
                raise ValueError('Bad pair indices')
            np.testing.assert_array_equal(u[p], u[n])
            np.testing.assert_array_equal(t[p], t[n])
            if not (np.all(y[p] == 1) and np.all(y[n] == 0) and np.all(i[p] != i[n])):
                raise ValueError('Observed-pair invariants failed')
            if len(np.unique(p)) != len(p):
                raise ValueError('More than one pair per positive endpoint')
            pu, pt = u[p], t[p]
            sampled = set((int(pu[j]), int(pt[j])) for j in rng.choice(len(p), min(64, len(p)), replace=False))
            for uid, timestamp in sampled:
                rows = db.execute("SELECT item_id,label FROM clean_events WHERE scene='rec' AND split=? AND user_id=? AND timestamp_ms=? ORDER BY id", (split, uid, timestamp)).fetchall()
                by_item = defaultdict(list)
                for iid, label in rows:
                    by_item[int(iid)].append(int(label))
                pos = {iid for iid, labels in by_item.items() if min(labels) == max(labels) == 1}
                neg = {iid for iid, labels in by_item.items() if min(labels) == max(labels) == 0}
                selected = np.flatnonzero((pu == uid) & (pt == timestamp))
                if set(i[p[selected]]) != pos or not set(i[n[selected]]).issubset(neg):
                    raise ValueError('Independent SQL context membership differs')
                for index in np.r_[p[selected], n[selected]]:
                    same = np.flatnonzero((u == uid) & (t == timestamp) & (i == i[index]))
                    if index != same[0]:
                        raise ValueError('Endpoint is not earliest consistent exposure row')
            reports[split] = {'pairs': len(p), 'all_pair_invariants_checked': True,
                              'sql_contexts_checked': len(sampled)}
            limits.append((int(t.min()), int(t.max())))
    finally:
        db.close()
    if limits[0][1] >= limits[1][0]:
        raise ValueError('Training/validation times overlap')
    return meta, reports


def audit(reference, pairs, study, output, pairs_only=False):
    if output.exists():
        raise FileExistsError('Preserve existing audit')
    pair_meta, pair_checks = audit_pairs(pairs)
    if pairs_only:
        write_json(output, {'status': 'passed', 'pairs': pair_checks, 'test_used': False,
                            'source_sha256': sha(__file__)})
        print('Exact-pair data audit passed', flush=True)
        return
    torch.set_num_threads(4)
    report = json.loads((study/'metrics.json').read_text(encoding='utf-8'))
    protocol = report['protocol']
    cfg = protocol['config']
    if report['status'] != 'complete' or report['test_used'] or report['default_changed']:
        raise ValueError('Complete non-test study required')
    if sha('src/v2/exact_pair_finetune.py') != protocol['source_sha256']:
        raise ValueError('Trainer changed')
    for name, expected in protocol['dependencies'].items():
        if sha(Path('src/v2')/name) != expected:
            raise ValueError('Training dependency changed: '+name)
    if sha(pairs/'metadata.json') != protocol['pairs_metadata_sha256'] or pair_meta['seed'] != cfg['pair_seed']:
        raise ValueError('Pair protocol changed')
    if sha(reference/'metadata.json') != protocol['reference_sha256']:
        raise ValueError('Reference metadata changed')
    if sha(reference/'candidates.jsonl') != protocol['candidate_sha256']:
        raise ValueError('Candidates changed')
    xv = np.load(reference/'valid_X.npy', mmap_mode='r')
    y, uv, vi, times = [np.load(CACHE/f'valid_{k}.npy') for k in ('y', 'users', 'items', 'times')]
    vp, vn = [np.load(pairs/f'valid_{k}.npy') for k in ('positive', 'negative')]
    pool_users = np.load(reference/'pool_users.npy')
    items = np.load(reference/'pool_items.npy', mmap_mode='r')
    pool = np.load(reference/'pool_X.npy', mmap_mode='r')
    labels, _, _, _ = load_labels(pool_users, items)
    coarse = np.load(reference/'coarse_positions.npy')
    meta = json.loads((reference/'metadata.json').read_text(encoding='utf-8'))
    categories = FeatureState.load(CACHE/'state.npz').categories
    order = np.lexsort((times, uv))
    cuts = np.r_[0, np.flatnonzero((np.diff(uv[order]) != 0) | (np.diff(times[order]) != 0))+1, len(y)]
    positive = np.add.reduceat(y[order].astype(np.int64), cuts[:-1])
    mixed = np.flatnonzero((positive > 0) & (positive < np.diff(cuts)))
    rng = np.random.default_rng(2026)
    checks = []
    for seed in cfg['seeds']:
        group = torch.load(study/'models'/f's{seed}'/'last.pt', map_location='cpu', weights_only=True)
        if not group['finished'] or group['epoch'] < cfg['min_epochs']:
            raise ValueError('Incomplete seed group')
        initial = group['histories']['bce'][0]['valid']
        lr = cfg['learning_rate']
        lr_bad = reductions = 0
        plateaus = {key: group['histories'][key][0]['valid_pair']['logloss'] for key in cfg['objectives']}
        bad = {key: 0 for key in cfg['objectives']}
        for epoch in range(1, group['epoch']+1):
            rows = [group['histories'][key][epoch] for key in cfg['objectives']]
            for field in ('order_sha256', 'pair_draw_sha256', 'learning_rate'):
                if len({row[field] for row in rows}) != 1:
                    raise ValueError('Unmatched training budget/order/LR')
            if rows[0]['learning_rate'] != lr:
                raise ValueError('Wrong common LR')
            for key, row in zip(cfg['objectives'], rows):
                allowed = (row['valid']['logloss'] <= initial['logloss']+cfg['logloss_guard']
                           and row['valid']['auc'] >= initial['auc']-cfg['auc_guard'])
                if row['guardrail_pass'] != allowed:
                    raise ValueError('Incorrect guardrail')
                improved = allowed and plateaus[key]-row['valid_pair']['logloss'] > cfg['min_delta']
                if improved:
                    plateaus[key] = row['valid_pair']['logloss']
                    bad[key] = 0
                else:
                    bad[key] += 1
                if key == 'bce':
                    lr_bad = 0 if improved else lr_bad+1
            if lr_bad >= cfg['lr_patience'] and lr > cfg['min_lr']:
                lr = max(cfg['min_lr'], lr*cfg['lr_factor'])
                lr_bad = 0
                reductions += 1
        if group['bad'] != bad or group['reductions'] != reductions or group['lr_bad'] != lr_bad:
            raise ValueError('Stopping/scheduler counter mismatch')
        if group['reason'] == 'all_arms_validation_plateau_after_shared_lr_reductions':
            if min(bad.values()) < cfg['early_patience'] or reductions < 2:
                raise ValueError('Premature synchronized stop')
        elif group['epoch'] != cfg['max_epochs']:
            raise ValueError('Unexpected budget stop')
        for key in cfg['objectives']:
            name = f'{key}_s{seed}'
            record = report['runs'][name]
            train = record['training']
            dest = study/'models'/f's{seed}'/key
            if sha(dest/'model.pt') != train['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed')
            history = group['histories'][key]
            selected = min((r for r in history if r['guardrail_pass']), key=lambda r: r['valid_pair']['logloss'])['epoch']
            if train['selected_epoch'] != selected or history != train['history']:
                raise ValueError('Selection/history mismatch')
            optimizer = group['optimizers'][key]
            steps = {int(v['step'].item()) for v in optimizer['state'].values()}
            if steps != {train['optimizer_steps']} or optimizer['param_groups'][0]['lr'] != lr:
                raise ValueError('Actual Adam update counts or LR differ')
            scorer = Scorer(dest/'model.pt')
            for parameter, value in scorer.model.state_dict().items():
                if not torch.equal(value, group['best_states'][key][parameter]):
                    raise ValueError('Group commit/selected model mismatch')
            p = scorer.predict(xv)
            np.testing.assert_array_equal(p, np.load(dest/'valid_predictions.npy'))
            # Training applies sigmoid once to all validation logits; serving
            # applies it per chunk. CPU vector tails can differ by two float32
            # ULPs, affecting exact ties. Verify both paths instead of relaxing
            # the metric comparison or changing recorded training results.
            from src.v2.route_attribution import normalize
            with torch.inference_mode():
                logits = np.concatenate([scorer.model(torch.from_numpy(normalize(xv[lo:lo+65536], scorer.mean, scorer.std, scorer.meta['columns']))).numpy()
                                         for lo in range(0, len(xv), 65536)])
                validation_p = torch.sigmoid(torch.from_numpy(logits)).numpy()
            max_sigmoid_difference = float(np.max(np.abs(p-validation_p)))
            np.testing.assert_array_max_ulp(p, validation_p, maxulp=2)
            actual = exposure_metrics(y, validation_p, uv)
            for metric in ('auc', 'logloss', 'gauc', 'brier'):
                np.testing.assert_allclose(actual[metric], train['selected_validation'][metric], atol=1e-12, rtol=0)
            # Pair loss is independently recomputed from raw selected-model logits.
            pair_ll = float(np.logaddexp(0., logits[vn].astype(np.float64)-logits[vp].astype(np.float64)).mean())
            np.testing.assert_allclose(pair_ll, train['selected_pair']['logloss'], atol=1e-12)
            ndcg = []
            for g in mixed:
                ix = order[cuts[g]:cuts[g+1]]
                ranked = sorted(ix.tolist(), key=lambda j: (-float(validation_p[j]), int(vi[j]), j))[:20]
                ideal = sum(1/log2(k+2) for k in range(min(20, int(positive[g]))))
                ndcg.append(sum(int(y[j])/log2(k+2) for k, j in enumerate(ranked))/ideal)
            np.testing.assert_allclose(np.mean(ndcg), train['selected_validation']['same_timestamp']['ndcg_at20'], atol=1e-12)
            scores = np.load(study/(name+'_scores.npy'), mmap_mode='r')
            arrays = np.load(study/(name+'_per_user.npz'))
            h, d = metric_arrays(items, scores, labels)
            np.testing.assert_array_equal(h, arrays['direct_hits'])
            np.testing.assert_allclose(d, arrays['direct_ndcg'], equal_nan=True)
            if int(h.sum()) != record['direct']['hit_events'] or int(arrays['cascade_hits'].sum()) != record['cascade']['hit_events']:
                raise ValueError('List summaries disagree')
            for j in rng.choice(len(pool_users), 8, replace=False):
                np.testing.assert_allclose(scorer.predict(pool[j]), scores[j], atol=1e-7, rtol=1e-6)
                ids = items[j].tolist()
                cs = np.full(len(ids), -100000.)
                cs[coarse[j]] = -np.arange(len(coarse[j]))
                cm, fm = dict(zip(ids, cs)), dict(zip(ids, scores[j]))
                executor = Cascade(lambda u, ii, t: [cm[i] for i in ii],
                    lambda u, ii, t: [fm[i] for i in ii], Budget(**meta['config']['budget']),
                    categories=dict(zip(ids, categories[items[j]].tolist())))
                served = executor.run(int(pool_users[j]), meta['cutoff_ms'],
                    [Candidate(i, 'frozen', k, meta['cutoff_ms']) for k, i in enumerate(ids, 1)], positives=labels[j])
                if served['stages'][-1]['hit_events'] != arrays['cascade_hits'][j]:
                    raise ValueError('Serving executor replay differs')
            checks.append({'name': name, 'selected_epoch': selected, 'matched_optimizer_steps': train['optimizer_steps'],
                           'all_valid_exposure_metrics': True, 'timestamp_groups_checked': len(mixed),
                           'full_user_direct_metrics': True, 'executor_replays': 8,
                           'max_chunk_vs_whole_sigmoid_probability_difference': max_sigmoid_difference,
                           'sigmoid_paths_within_two_float32_ulps': True})
            arrays.close()
            scores._mmap.close()
    if len(checks) != len(cfg['seeds'])*3:
        raise ValueError('Missing models')
    write_json(output, {'status': 'passed', 'pairs': pair_checks, 'models': checks,
               'frozen_assets_verified': frozen_assets(), 'source_sha256': sha(__file__),
               'test_used': False, 'default_changed': False})
    for array in (xv, items, pool):
        array._mmap.close()
    print('Exact-pair fine-tuning audit passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, default=Path('artifacts/full_exposure_local_20261022/reference'))
    parser.add_argument('--pairs', type=Path, default=Path('artifacts/exact_pair_finetune_20261022/pairs'))
    parser.add_argument('--study', type=Path, default=Path('artifacts/exact_pair_finetune_20261022/study'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pairs-only', action='store_true')
    args = parser.parse_args()
    audit(args.reference, args.pairs, args.study, args.out, args.pairs_only)
