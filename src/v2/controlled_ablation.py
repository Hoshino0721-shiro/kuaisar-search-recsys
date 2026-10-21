"""Predeclared diagnostic retraining; never deploys or opens test arrays.

Factorial: search availability x prior-seen-item exposure weighting. Same
sample, architecture, normalization, seeds and training budget for every cell.
Weighting is a covariate proxy, not propensity correction or true candidate matching.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.ctr_features import CACHE, FEATURES, FeatureState, connection
from src.ranking import metrics as exposure_metrics
from src.v2.models import FineDCN
from src.v2.cascade import observed_positive_metrics
from src.v2.ranking_diagnostics import rank_pool
from src.v2.ranking_root_cause import weighted_exposure_metrics
from src.v2.train_rankers import sha


def standardized(x, mean, std, masked_columns):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError('Expected finite 20-column causal inputs')
    values = (x - mean) / std
    values[:, masked_columns] = 0.
    return values


def exposure_weights(x, warm_weight):
    if not np.isfinite(warm_weight) or warm_weight <= 0:
        raise ValueError('warm_weight must be positive')
    return np.where(np.asarray(x)[:, 10] > 0, warm_weight, 1.).astype(np.float32)


class DiagnosticRanker:
    def __init__(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        self.metadata = saved['metadata']
        if self.metadata['kind'] != 'diagnostic_dcn_masked_v1':
            raise ValueError('Not a controlled diagnostic checkpoint')
        self.mean = np.asarray(saved['mean'], dtype=np.float32)
        self.std = np.asarray(saved['std'], dtype=np.float32)
        self.mask = self.metadata['masked_columns']
        self.model = FineDCN()
        self.model.load_state_dict(saved['state_dict'])
        self.model.eval()

    def predict(self, x):
        pieces = []
        with torch.inference_mode():
            for lo in range(0, len(x), 16384):
                z = standardized(x[lo:lo + 16384], self.mean, self.std, self.mask)
                pieces.append(torch.sigmoid(self.model(torch.from_numpy(z))).numpy())
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)


def train_one(x, y, indices, mean, std, config, seed, search, warm_weight, output):
    if output.exists():
        raise FileExistsError('Existing run is preserved')
    mask = [] if search else list(range(16, 20))
    selected = np.asarray(x[indices], dtype=np.float32)
    weights = exposure_weights(selected, warm_weight)
    weight_mean = float(weights.mean())
    tx = torch.from_numpy(standardized(selected, mean, std, mask))
    ty = torch.from_numpy(np.asarray(y[indices], dtype=np.float32))
    tw = torch.from_numpy(weights / weight_mean)
    torch.manual_seed(seed)
    model = FineDCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    history = []
    for epoch in range(config['epochs']):
        order = torch.randperm(len(tx))
        total = 0.
        for lo in range(0, len(tx), config['batch_size']):
            group = order[lo:lo + config['batch_size']]
            losses = nn.functional.binary_cross_entropy_with_logits(model(tx[group]), ty[group], reduction='none')
            loss = (losses * tw[group]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite diagnostic training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(group)
        history.append({'epoch': epoch + 1, 'weighted_train_logloss': total / len(tx)})
    metadata = {'kind': 'diagnostic_dcn_masked_v1', 'seed': seed, 'search_enabled': search,
                'columns': list(range(len(FEATURES))),
                'masked_columns': mask, 'warm_weight': warm_weight,
                'effective_weighted_warm_fraction': float(weights[selected[:, 10] > 0].sum() / weights.sum()),
                'sample_rows': len(indices), 'epochs': config['epochs'],
                'selection': config['checkpoint_rule'], 'test_used': False,
                'feature_state_sha256': sha(CACHE / 'state.npz'),
                'model_source_sha256': sha(Path(__file__).with_name('models.py'))}
    output.mkdir(parents=True)
    torch.save({'metadata': metadata, 'mean': mean.tolist(), 'std': std.tolist(),
                'state_dict': model.state_dict()}, output / 'model.pt')
    (output / 'training.json').write_text(json.dumps({'metadata': metadata, 'history': history}, indent=2), encoding='utf-8')
    return DiagnosticRanker(output / 'model.pt')


def run(config_path, output):
    if output.exists():
        raise FileExistsError('Choose a new experiment root')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['checkpoint_rule'] != 'fixed final epoch; no validation-based checkpoint selection':
        raise ValueError('Unsupported checkpoint protocol')
    pool = Path(config['candidate_archive'])
    archive = json.loads((pool / 'metrics.json').read_text(encoding='utf-8'))
    if archive['split'] != 'valid' or archive['test_used']:
        raise ValueError('Validation candidates required')
    if sha(pool / 'candidates.jsonl') != archive['candidate_sha256']:
        raise ValueError('Frozen pool changed')
    output.mkdir(parents=True)
    # Written before fitting/evaluation, and never updated in response to scores.
    (output / 'protocol.json').write_text(json.dumps({
        'config': config, 'source_sha256': sha(__file__), 'test_used': False,
        'candidate_sha256': archive['candidate_sha256'],
        'feature_cache_sha256': sha(CACHE / 'ready.json')}, indent=2), encoding='utf-8')
    torch.set_num_threads(4)
    x = np.load(CACHE / 'train_X.npy', mmap_mode='r')
    y = np.load(CACHE / 'train_y.npy', mmap_mode='r')
    indices = np.sort(np.random.default_rng(config['sample_seed']).choice(
        len(x), min(config['sample_rows'], len(x)), replace=False))
    np.save(output / 'sample_indices.npy', indices)
    selected = np.asarray(x[indices], dtype=np.float32)
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(selected.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    del selected
    xv = np.load(CACHE / 'valid_X.npy', mmap_mode='r')
    yv = np.load(CACHE / 'valid_y.npy')
    uv = np.load(CACHE / 'valid_users.npy')
    models, reports = {}, {}
    for weight in config['warm_weights']:
        for search in config['search_enabled']:
            for seed in config['model_seeds']:
                name = f'w{weight:g}_search{int(search)}_s{seed}'
                model = train_one(x, y, indices, mean, std, config, seed, search, weight, output / name)
                p = model.predict(xv)
                np.save(output / name / 'valid_predictions.npy', p)
                groups = {'all': np.ones(len(yv), dtype=bool),
                          'warm_item': xv[:, 10] > 0, 'has_search': xv[:, 19] > 0}
                reports[name] = {'metadata': model.metadata,
                                 'checkpoint_sha256': sha(output / name / 'model.pt'),
                                 'exposure': {g: exposure_metrics(yv[m], p[m], uv[m]) for g, m in groups.items()}}
                models[name] = model
                print(f'Trained {name}: AUC={reports[name]["exposure"]["all"]["auc"]:.6f}', flush=True)
    db = connection()
    positive, exposure = defaultdict(Counter), defaultdict(Counter)
    try:
        for u, i, label in db.execute(
                "SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
            exposure[u][i] += 1
            if label:
                positive[u][i] += 1
    finally:
        db.close()
    state = FeatureState.load(CACHE / 'state.npz')
    aggregate = {name: {'hits': 0, 'ndcg': [], 'matched_p': []} for name in models}
    matched_n, matched_y, per_user = [], [], []
    for line in (pool / 'candidates.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        uid = row['user_id']
        ids = row['items']
        features = state.features(np.full(len(ids), uid), np.asarray(ids, dtype=np.int64),
                                  np.full(len(ids), row['timestamp_ms']))
        n = np.array([exposure[uid][i] for i in ids])
        labels = np.array([positive[uid][i] for i in ids])
        known = n > 0
        matched_n.extend(n[known].tolist())
        matched_y.extend(labels[known].tolist())
        record = {'user_id': uid, 'methods': {}}
        for name, model in models.items():
            p = model.predict(features)
            order = rank_pool(ids, p)[:20]
            quality = observed_positive_metrics(order, positive[uid])
            aggregate[name]['hits'] += quality['hit_events']
            if quality['user_period_ndcg_at_20'] is not None:
                aggregate[name]['ndcg'].append(quality['user_period_ndcg_at_20'])
            aggregate[name]['matched_p'].extend(p[known].tolist())
            record['methods'][name] = quality
        per_user.append(record)
        if len(per_user) % 1000 == 0:
            print(f'Paired candidate evaluation {len(per_user)}/{len(archive["users"])}', flush=True)
    total_positive = sum(sum(positive[u].values()) for u in archive['users'])
    for name, values in aggregate.items():
        reports[name]['candidate_list'] = {
            'users': len(per_user), 'positive_events': total_positive,
            'hit_events': values['hits'], 'event_recall_at20': values['hits'] / total_positive,
            'ndcg_at20_positive_users': float(np.mean(values['ndcg'])),
            'matched_logged_exposures': weighted_exposure_metrics(values['matched_p'], matched_n, matched_y)}
    result = {'test_used': False, 'diagnosis_only': True, 'config': config,
              'candidate_sha256': archive['candidate_sha256'], 'runs': reports,
              'scope': 'three model seeds, fixed exposure sample; no statistical significance claim or deployment selection',
              'warm_weighting': 'prior-seen-item covariate proxy only; not exact candidate matching or propensity correction',
              'masked_search': 'zero normalized search columns during both training and inference; same 20-wide architecture'}
    (output / 'per_user.json').write_text(json.dumps(per_user), encoding='utf-8')
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Completed diagnostic factorial: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/ranking_controlled_ablation.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == '__main__':
    main()
