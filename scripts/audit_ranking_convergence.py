"""Independent selected-model/trajectory/serving audit, validation only."""
import argparse
import json
from math import log2
from pathlib import Path

import numpy as np
import torch

from src.ctr_features import CACHE, FeatureState
from src.ranking import metrics as independent_metrics
from src.v2.cascade import Budget, Candidate, Cascade
from src.v2.full_exposure_control import verify_prepared
from src.v2.ranking_convergence import frozen_assets, write_json
from src.v2.route_attribution import Scorer, load_labels, metric_arrays
from src.v2.train_rankers import sha


def audit(reference, study, previous, prepared, output):
    if output.exists():
        raise FileExistsError('Preserve existing audit')
    report = json.loads((study/'metrics.json').read_text(encoding='utf-8'))
    cfg = report['protocol']['config']
    if report['status'] != 'complete' or report['test_used'] or report['default_changed']:
        raise ValueError('Completed non-test study required')
    if sha('src/v2/ranking_convergence.py') != report['protocol']['source_sha256']:
        raise ValueError('Trainer changed')
    if sha(CACHE/'state.npz') != report['protocol']['feature_state_sha256']:
        raise ValueError('Feature state changed')
    if sha(reference/'metadata.json') != report['protocol']['reference_metadata_sha256']:
        raise ValueError('Reference changed')
    if sha(reference/'candidates.jsonl') != report['protocol']['candidate_sha256']:
        raise ValueError('Candidate pool changed')
    if sha(previous/'metrics.json') != report['protocol']['previous_metrics_sha256']:
        raise ValueError('Comparator changed')
    if sha(prepared/'metadata.json') != report['protocol']['prepared_metadata_sha256']:
        raise ValueError('Prepared full cache metadata changed')
    verify_prepared(prepared, reference, cfg)
    torch.set_num_threads(4)
    items = np.load(reference/'pool_items.npy', mmap_mode='r')
    users = np.load(reference/'pool_users.npy')
    pool = np.load(reference/'pool_X.npy', mmap_mode='r')
    xv = np.load(reference/'valid_X.npy', mmap_mode='r')
    y = np.load(CACHE/'valid_y.npy')
    uv = np.load(CACHE/'valid_users.npy')
    times = np.load(CACHE/'valid_times.npy')
    vi = np.load(CACHE/'valid_items.npy')
    train_x = np.load(prepared/'full_X.npy', mmap_mode='r')
    train_y = np.load(CACHE/'train_y.npy')
    probe = np.sort(np.random.default_rng(2026).choice(len(train_y), min(65536, len(train_y)), replace=False))
    labels, _, _, _ = load_labels(users, items)
    coarse = np.load(reference/'coarse_positions.npy')
    reference_meta = json.loads((reference/'metadata.json').read_text(encoding='utf-8'))
    budget = reference_meta['config']['budget']
    categories = FeatureState.load(CACHE/'state.npz').categories
    rng = np.random.default_rng(2026)
    # Independent Python group indexing and ranking for timestamp metric checks.
    order = np.lexsort((times, uv))
    cuts = np.r_[0, np.flatnonzero((np.diff(uv[order]) != 0) | (np.diff(times[order]) != 0))+1, len(y)]
    pos = np.add.reduceat(y[order].astype(np.int64), cuts[:-1])
    sizes = np.diff(cuts)
    eligible = np.flatnonzero((pos > 0) & (pos < sizes))
    checks = []
    for name, record in report['runs'].items():
        dest = study/'models'/name
        train = record['training']
        history = train['history']
        selected = int(np.argmin([h['valid']['logloss'] for h in history]))+1
        if selected != train['selected_epoch'] or len(history) < cfg['min_epochs']:
            raise ValueError('Selection or minimum budget incorrect')
        if sha(dest/'model.pt') != train['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        last = torch.load(dest/'last.pt', map_location='cpu', weights_only=True)
        if last['history'] != history or not last['finished'] or last['selected_epoch'] != selected:
            raise ValueError('Last checkpoint/history disagreement')
        if not all(k in last for k in ('optimizer_state_dict', 'rng_state', 'best_checkpoint_sha256')):
            raise ValueError('Resume state incomplete')
        plateau = float('inf')
        bad = since_decay = decays = 0
        lr = cfg['learning_rate']
        for row in history:
            if row['learning_rate'] != lr:
                raise ValueError('Recorded LR does not match declared scheduler')
            if plateau-row['valid']['logloss'] > cfg['min_delta']:
                plateau = row['valid']['logloss']
                bad = since_decay = 0
            else:
                bad += 1
                since_decay += 1
            if row['epoch'] >= 5 and since_decay >= cfg['lr_patience'] and lr > cfg['min_lr']:
                lr = max(cfg['min_lr'], lr*cfg['lr_factor'])
                decays += 1
                since_decay = 0
        if (last['bad'], last['reductions'], last['lr_bad']) != (bad, decays, since_decay):
            raise ValueError('Saved stopping counters disagree with replay')
        if last['optimizer_state_dict']['param_groups'][0]['lr'] != lr:
            raise ValueError('Saved optimizer LR differs from scheduler')
        if sha(dest/'model.pt') != last['best_checkpoint_sha256']:
            raise ValueError('Best/last checkpoint commitment mismatch')
        if train['early_stopping_reason'] == 'validation_plateau_after_lr_reductions':
            if last['bad'] < cfg['early_patience'] or last['reductions'] < 2:
                raise ValueError('Premature early stop')
        elif len(history) != cfg['max_epochs']:
            raise ValueError('Incomplete epoch budget')
        a = Scorer(dest/'epoch5.pt').model.state_dict()
        b = Scorer(previous/'models'/name/'model.pt').model.state_dict()
        if not all(torch.equal(a[k], b[k]) for k in a):
            raise ValueError('Epoch five not bitwise identical')
        scorer = Scorer(dest/'model.pt')
        p = scorer.predict(xv)
        np.testing.assert_array_equal(p, np.load(dest/'valid_predictions.npy'))
        metrics = independent_metrics(y, p, uv)
        for key in ('auc', 'gauc', 'logloss', 'brier'):
            np.testing.assert_allclose(metrics[key], record['selected_validation'][key], atol=1e-12, rtol=0)
        probe_p = np.clip(scorer.predict(train_x[probe]).astype(np.float64), 1e-7, 1-1e-7)
        probe_ll = -np.mean(train_y[probe]*np.log(probe_p)+(1-train_y[probe])*np.log1p(-probe_p))
        np.testing.assert_allclose(probe_ll, history[selected-1]['train_probe_logloss'], atol=1e-12, rtol=0)
        # Macro NDCG across every mixed timestamp group, independent Python ordering.
        ndcg = []
        for g in eligible:
            ix = order[cuts[g]:cuts[g+1]]
            ranked = sorted(ix.tolist(), key=lambda j: (-float(p[j]), int(vi[j]), j))[:20]
            dcg = sum(int(y[j])/log2(k+2) for k, j in enumerate(ranked))
            ideal = sum(1/log2(k+2) for k in range(min(20, int(pos[g]))))
            ndcg.append(dcg/ideal)
        np.testing.assert_allclose(np.mean(ndcg), record['selected_validation']['same_timestamp']['ndcg_at20'], atol=1e-12)
        scores = np.load(study/(name+'_scores.npy'), mmap_mode='r')
        arrays = np.load(study/(name+'_per_user.npz'))
        hits, d = metric_arrays(items, scores, labels)
        np.testing.assert_array_equal(hits, arrays['direct_hits'])
        np.testing.assert_allclose(d, arrays['direct_ndcg'], equal_nan=True)
        if int(hits.sum()) != record['direct']['hit_events'] or int(arrays['cascade_hits'].sum()) != record['cascade']['hit_events']:
            raise ValueError('Full-user list summaries disagree')
        for j in rng.choice(len(users), min(8, len(users)), replace=False):
            np.testing.assert_allclose(scorer.predict(pool[j]), scores[j], atol=1e-7, rtol=1e-6)
            coarse_scores = np.full(len(items[j]), -100000., dtype=float)
            coarse_scores[coarse[j]] = -np.arange(len(coarse[j]))
            ids = items[j].tolist()
            cm = dict(zip(ids, coarse_scores))
            fm = dict(zip(ids, scores[j]))
            category_map = dict(zip(ids, categories[items[j]].tolist()))
            executor = Cascade(lambda u, ii, t: [cm[i] for i in ii],
                               lambda u, ii, t: [fm[i] for i in ii], Budget(**budget), categories=category_map)
            candidates = [Candidate(i, 'frozen', k, reference_meta['cutoff_ms'])
                          for k, i in enumerate(ids, 1)]
            served = executor.run(int(users[j]), reference_meta['cutoff_ms'], candidates, positives=labels[j])
            if served['stages'][-1]['hit_events'] != arrays['cascade_hits'][j]:
                raise ValueError('Serving executor replay differs')
        checks.append({'name': name, 'selected_epoch': selected, 'stopped_epoch': len(history),
                       'epoch5_bitwise_match': True, 'independent_exposure_metrics': True,
                       'mixed_timestamp_groups_checked': len(eligible), 'full_user_list_metrics': True,
                       'executor_replays': min(8, len(users))})
        scores._mmap.close()
        arrays.close()
    expected = len(cfg['arms'])*len(cfg['seeds'])
    if len(checks) != expected:
        raise ValueError('Missing models')
    write_json(output, {'status': 'passed', 'models': checks, 'model_count': len(checks),
               'original_frozen_assets_verified': frozen_assets(),
               'source_sha256': sha(__file__), 'test_used': False, 'default_changed': False})
    for array in (items, pool, xv, train_x):
        array._mmap.close()
    print('Convergence audit passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, default=Path('artifacts/full_exposure_local_20261022/reference'))
    parser.add_argument('--previous', type=Path, default=Path('artifacts/full_exposure_local_20261022/study'))
    parser.add_argument('--prepared', type=Path, default=Path('artifacts/full_exposure_local_20261022/full_cache'))
    parser.add_argument('--study', type=Path, default=Path('artifacts/ranking_convergence_20261022'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.reference, args.study, args.previous, args.prepared, args.out)
