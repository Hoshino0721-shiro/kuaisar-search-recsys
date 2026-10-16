"""Train v2 rankers on v1 causal exposure caches; never reads test arrays."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from src.ctr_features import FEATURES
from src.v2.models import COARSE_COLUMNS, make_model


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def probability_metrics(y, p):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError('Invalid predictions')
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    logloss = -np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    positives = int(y.sum())
    negatives = len(y) - positives
    auc = None
    if positives and negatives:
        order = np.argsort(p, kind='stable')
        values, labels = p[order], y[order]
        starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
        lengths = np.diff(np.r_[starts, len(y)])
        pos = np.add.reduceat(labels.astype(np.int64), starts)
        neg = lengths - pos
        below = np.cumsum(neg) - neg
        auc = float(np.sum(pos * (below + 0.5 * neg)) / (positives * negatives))
    return {'rows': len(y), 'auc': auc, 'logloss': float(logloss)}


class Ranker:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.metadata = saved['metadata']
        self.columns = self.metadata['columns']
        self.mean = np.array(saved['mean'], dtype=np.float32)
        self.std = np.array(saved['std'], dtype=np.float32)
        self.model = make_model(self.metadata['kind'])
        self.model.load_state_dict(saved['state_dict'], strict=True)
        self.model.eval()

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
            raise ValueError('Expected finite 20-column causal features')
        return self.predict_selected(x[:, self.columns])

    def predict_selected(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.columns) or not np.isfinite(x).all():
            raise ValueError('Feature width differs from ranker column schema')
        pieces = []
        with torch.inference_mode():
            for lo in range(0, len(x), 16384):
                values = (x[lo:lo + 16384] - self.mean) / self.std
                pieces.append(torch.sigmoid(self.model(torch.from_numpy(values))).numpy())
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)


def train(kind, cache, output, sample_rows, epochs, batch_size, seed):
    if min(sample_rows, epochs, batch_size) < 1:
        raise ValueError('Training budgets must be positive')
    if output.exists():
        raise FileExistsError(f'Choose a new run directory: {output}')
    ready = json.loads((cache / 'ready.json').read_text(encoding='utf-8'))
    if ready['features'] != FEATURES:
        raise ValueError('Feature schema differs from the causal v1 cache')
    x = np.load(cache / 'train_X.npy', mmap_mode='r')
    y = np.load(cache / 'train_y.npy', mmap_mode='r')
    xv = np.load(cache / 'valid_X.npy', mmap_mode='r')
    yv = np.load(cache / 'valid_y.npy')
    for xx, yy in [(x, y), (xv, yv)]:
        if xx.ndim != 2 or xx.shape[1] != len(FEATURES) or len(xx) != len(yy) or not len(yy):
            raise ValueError('Invalid exposure cache shape')
        if not np.isin(yy, [0, 1]).all():
            raise ValueError('Exposure labels must be observed binary clicks')
    columns = COARSE_COLUMNS if kind == 'coarse_lr' else list(range(len(FEATURES)))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(x), min(sample_rows, len(x)), replace=False))
    xt = np.asarray(x[indices][:, columns], dtype=np.float32)
    if not np.isfinite(xt).all():
        raise ValueError('Nonfinite train features')
    mean = xt.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = xt.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    tx = torch.from_numpy((xt - mean) / std)
    ty = torch.from_numpy(np.asarray(y[indices], dtype=np.float32))
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model = make_model(kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    output.mkdir(parents=True)
    metadata = {'kind': kind, 'seed': seed, 'sample_rows': len(indices), 'epochs': epochs,
                'batch_size': batch_size, 'columns': columns,
                'feature_names': [FEATURES[i] for i in columns],
                'selection': 'minimum validation LogLoss across epochs; no test reads',
                'training_labels': 'observed rec exposure click; no candidate pseudo-negatives',
                'feature_cache_sha256': sha(cache / 'ready.json'),
                'feature_state_sha256': sha(cache / 'state.npz'),
                'model_source_sha256': sha(Path(__file__).with_name('models.py')),
                'trainer_source_sha256': sha(__file__), 'device': 'cpu'}
    best = float('inf')
    history = []
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tx))
        total = 0.0
        for lo in range(0, len(tx), batch_size):
            index = order[lo:lo + batch_size]
            loss = nn.functional.binary_cross_entropy_with_logits(model(tx[index]), ty[index])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        model.eval()
        prediction = []
        with torch.inference_mode():
            for lo in range(0, len(xv), 16384):
                values = (np.asarray(xv[lo:lo + 16384, columns]) - mean) / std
                prediction.append(torch.sigmoid(model(torch.from_numpy(values))).numpy())
        p = np.concatenate(prediction)
        validation = probability_metrics(yv, p)
        record = {'epoch': epoch, 'train_logloss': total / len(tx), 'valid': validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation['logloss'] < best:
            best = validation['logloss']
            torch.save({'metadata': {**metadata, 'selected_epoch': epoch},
                        'mean': mean.tolist(), 'std': std.tolist(),
                        'state_dict': model.state_dict()}, output / 'model.pt')
    selected = Ranker(output / 'model.pt')
    selected_p = selected.predict(xv)
    if (cache / 'valid_users.npy').exists():
        from src.ranking import metrics as exposure_metrics
        selected_validation = exposure_metrics(yv, selected_p, np.load(cache / 'valid_users.npy'))
    else:
        selected_validation = probability_metrics(yv, selected_p)
    report = {'metadata': metadata, 'history': history,
              'selected_epoch': selected.metadata['selected_epoch'],
              'selected_validation': selected_validation,
              'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'checkpoint_sha256': sha(output / 'model.pt'),
              'elapsed_seconds': perf_counter() - start, 'test_used': False,
              'scope': 'exposure validation only; cascade retention still requires separate evaluation'}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.save(output / 'sample_indices.npy', indices)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=['coarse_lr', 'fine_dcn'])
    parser.add_argument('--cache', type=Path, default=Path('data/processed/kuaisar_small_v1/ctr_features_v1'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sample-rows', type=int, default=1000000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    train(args.kind, args.cache, args.out, args.sample_rows, args.epochs, args.batch_size, args.seed)


if __name__ == '__main__':
    main()
