"""Staged, three-seed embedding/history DCN experiments on real exposures.

Validation development only. Existing checkpoints and defaults are preserved.
"""
import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.ctr_features import CACHE
from src.ranking import metrics
from src.v2.enhanced_data import EnhancedData
from src.v2.enhanced_model import EnhancedDCN, EnhancedRanker, spec_for
from src.v2.train_rankers import probability_metrics, sha


class Banks:
    def __init__(self, data, cache=CACHE):
        self.data = data
        if data.meta['feature_state_sha256'] != sha(cache/'state.npz'):
            raise ValueError('Numeric state and enhanced preparation differ')
        self.x = np.asarray(np.load(cache/'train_X.npy', mmap_mode='r')[data.sample_indices], dtype=np.float32)
        self.y = torch.tensor(np.load(cache/'train_y.npy', mmap_mode='r')[data.sample_indices], dtype=torch.float32)
        self.mean = self.x.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std = np.maximum(self.x.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
        self.numeric = torch.from_numpy((self.x-self.mean)/self.std)
        self.fields = torch.tensor(np.asarray(data.train_fields), dtype=torch.long)
        self.context = torch.tensor(np.asarray(data.train_context), dtype=torch.long)
        self.rec = torch.tensor(np.asarray(data.context_rec), dtype=torch.long)
        self.search = torch.tensor(np.asarray(data.context_search), dtype=torch.long)
        self.xv = np.load(cache/'valid_X.npy', mmap_mode='r')
        self.yv = np.load(cache/'valid_y.npy')
        self.uv = np.load(cache/'valid_users.npy')
        self.iv = np.load(cache/'valid_items.npy')
        self.qr = np.asarray(data.query_rec[self.uv])
        self.qs = np.asarray(data.query_search[self.uv])

    def coverage(self, spec, minimum=5):
        result = {}
        for name, column in [('author', 2), ('category', 3), ('user', 0)]:
            if (name in ('author','category') and spec['attrs']) or (name == 'user' and spec['user']):
                counts = torch.bincount(self.fields[:, column], minlength=self.data.meta['cardinalities'][name])
                result[name] = torch.nonzero(counts >= minimum).flatten()
        item = np.zeros(self.data.meta['cardinalities']['item'],dtype=np.int64)
        if spec['item']:
            item += np.bincount(self.data.train_fields[:,1],minlength=len(item))
        weights = np.bincount(self.data.train_context,minlength=len(self.data.context_users))
        for enabled,h in [(spec['rec'],self.data.context_rec),(spec['search'],self.data.context_search)]:
            if enabled:
                item += np.bincount(h.ravel(),weights=np.repeat(weights,h.shape[1]),minlength=len(item)).astype(np.int64)
        if spec['item'] or spec['rec'] or spec['search']:
            result['item'] = np.flatnonzero(item >= minimum)
        return result

    def validation_prediction(self, model, spec):
        result = []
        model.eval()
        with torch.inference_mode():
            for lo in range(0, len(self.yv), 16384):
                sl = slice(lo, lo+16384)
                z = torch.from_numpy((np.asarray(self.xv[sl])-self.mean)/self.std)
                f = torch.tensor(np.asarray(self.data.valid_fields[sl]), dtype=torch.long)
                r = torch.tensor(self.qr[sl], dtype=torch.long) if spec['rec'] else None
                s = torch.tensor(self.qs[sl], dtype=torch.long) if spec['search'] else None
                result.append(torch.sigmoid(model(z, f, r, s)).numpy())
        return np.concatenate(result)


def train(name, spec, seed, cfg, bank, output):
    if output.exists():
        raise FileExistsError('Preserve existing enhanced run: '+str(output))
    output.mkdir(parents=True)
    torch.manual_seed(seed)
    model = EnhancedDCN(spec, bank.data.meta['cardinalities'])
    model.set_coverage(bank.coverage(spec,cfg['minimum_frequency']))
    dense = [p for n, p in model.named_parameters() if not n.startswith('tables.')]
    optimizers = [torch.optim.Adam(dense, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])]
    embeddings = list(model.tables.parameters())
    if embeddings:
        optimizers.append(torch.optim.SparseAdam(embeddings, lr=cfg['learning_rate']))
    metadata = {'kind': 'enhanced_dcn_v2', 'experiment': name, 'spec': spec, 'seed': seed,
                'selection': cfg['selection'], 'data_sha256': sha(bank.data.path/'metadata.json'),
                'state_sha256': sha(CACHE/'state.npz'), 'sample_sha256': sha(bank.data.path/'sample_indices.npy'),
                'cardinalities': bank.data.meta['cardinalities'], 'test_used': False,
                'model_source_sha256': sha(Path(__file__).with_name('enhanced_model.py')),
                'trainer_source_sha256': sha(__file__)}
    best, history = float('inf'), []
    started = perf_counter()
    for epoch in range(cfg['epochs']):
        model.train()
        order = torch.randperm(len(bank.y))
        total = 0.
        for lo in range(0, len(order), cfg['batch_size']):
            group = order[lo:lo+cfg['batch_size']]
            c = bank.context[group]
            r = bank.rec[c] if spec['rec'] else None
            s = bank.search[c] if spec['search'] else None
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(bank.numeric[group], bank.fields[group], r, s), bank.y[group])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite enhanced loss')
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            total += loss.item()*len(group)
        prediction = bank.validation_prediction(model, spec)
        scores = probability_metrics(bank.yv, prediction)
        history.append({'epoch': epoch+1, 'train_logloss': total/len(order), 'valid': scores})
        print(f'{name}/s{seed} epoch {epoch+1}: AUC={scores["auc"]:.6f}, LogLoss={scores["logloss"]:.6f}', flush=True)
        if scores['logloss'] < best:
            best = scores['logloss']
            torch.save({'metadata': {**metadata, 'selected_epoch': epoch+1},
                        'mean': bank.mean.tolist(), 'std': bank.std.tolist(),
                        'state_dict': model.state_dict()}, output/'model.pt')
    selected = EnhancedRanker(output/'model.pt', bank.data, metadata['state_sha256'])
    p = bank.validation_prediction(selected.model, spec)
    np.save(output/'valid_predictions.npy', p)
    freq = bank.data.item_train_frequency[bank.iv]
    groups = {'all': np.ones(len(p), dtype=bool), 'unseen_item': freq == 0,
              'low_frequency_item': (freq > 0)&(freq < cfg['minimum_frequency']),
              'high_frequency_item': freq >= 20,
              'has_rec_history': (bank.qr != 0).any(axis=1),
              'empty_rec_history': ~(bank.qr != 0).any(axis=1),
              'has_search_history': (bank.qs != 0).any(axis=1),
              'no_search_history': ~(bank.qs != 0).any(axis=1)}
    report = {'metadata': selected.metadata, 'history': history,
              'validation': {g: metrics(bank.yv[m], p[m], bank.uv[m]) for g,m in groups.items() if m.any()},
              'parameters': sum(p.numel() for p in model.parameters()),
              'checkpoint_bytes': (output/'model.pt').stat().st_size,
              'checkpoint_sha256': sha(output/'model.pt'), 'elapsed_seconds': perf_counter()-started,
              'test_used': False}
    (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def reusable_report(source, protocol, spec, seed, data_digest, model_digest):
    if not (source/'metrics.json').exists():
        return None
    report = json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    m = report['metadata']
    if (m['spec'] != spec or m['seed'] != seed or m['data_sha256'] != data_digest
            or m['model_source_sha256'] != model_digest
            or m['trainer_source_sha256'] != protocol['source_sha256']
            or sha(source/'model.pt') != report['checkpoint_sha256']
            or not (source/'valid_predictions.npy').exists()):
        raise ValueError('Completed artifact cannot be safely reused: '+str(source))
    return report


def run(config, cache, output, reuse=None):
    if output.exists():
        raise FileExistsError('Choose a new enhancement run directory')
    cfg = json.loads(config.read_text(encoding='utf-8'))
    old_protocol = json.loads((reuse/'protocol.json').read_text(encoding='utf-8')) if reuse else None
    if old_protocol is not None and old_protocol['config'] != cfg:
        raise ValueError('Reuse configuration differs')
    torch.set_num_threads(4)
    with EnhancedData(cache) as data:
        if data.meta['sample_rows'] != cfg['sample_rows'] or data.meta['history_length'] != cfg['history_length']:
            raise ValueError('Cache/protocol mismatch')
        bank = Banks(data)
        output.mkdir(parents=True)
        protocol = {'config': cfg, 'cache': str(cache), 'source_sha256': sha(__file__),
                    'model_source_sha256': sha(Path(__file__).with_name('enhanced_model.py')),
                    'data_sha256': sha(cache/'metadata.json'), 'test_used': False}
        if reuse is not None:
            protocol['reuse_from'] = str(reuse)
            protocol['reused_training_source_sha256'] = old_protocol['source_sha256']
        (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        all_runs = {}
        def execute(name, spec):
            for seed in cfg['seeds']:
                key = name+'_s'+str(seed)
                reused = reusable_report(reuse/key, old_protocol, spec, seed,
                                         protocol['data_sha256'],protocol['model_source_sha256']) if reuse else None
                if reused is not None:
                    shutil.copytree(reuse/key, output/key)
                    all_runs[key] = {**reused,'reused_from':str(reuse/key)}
                    print(f'Verified/reused completed {key}',flush=True)
                else:
                    all_runs[key] = train(name, spec, seed, cfg, bank, output/key)
            (output/'progress.json').write_text(json.dumps(all_runs, indent=2), encoding='utf-8')
        def mean_loss(name):
            return float(np.mean([r['validation']['all']['logloss'] for r in all_runs.values()
                                  if r['metadata']['experiment'] == name]))
        for name in cfg['first_stage']:
            execute(name, spec_for(name))
        parents = ['B', 'C']
        if mean_loss('C') > mean_loss('B'):
            for name in cfg['id_split_if_C_worse']:
                execute(name, spec_for(name))
                parents.append(name)
        parent = min(parents, key=lambda n: (mean_loss(n), n))
        parent_spec = spec_for(parent)
        execute('D', spec_for('D', parent_spec))
        execute('E', spec_for('E', parent_spec))
        summary = {'protocol': protocol, 'history_parent': parent, 'runs': all_runs,
                   'test_used': False, 'default_model_changed': False,
                   'interpretation': 'validation development with reused validation window; not independent final test'}
        (output/'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'All stages completed: {output}; history parent={parent}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('configs/dcn_enhancement_v1.json'))
    p.add_argument('--cache', type=Path, default=Path('artifacts/dcn_enhancement_v1/cache'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reuse', type=Path)
    a = p.parse_args()
    run(a.config, a.cache, a.out, a.reuse)
