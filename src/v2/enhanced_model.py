"""Attribute/ID/history DCN, not DCN-v2 or sequence attention."""
import numpy as np
import torch
from torch import nn

from src.v2.models import FineDCN


def spec_for(name, parent=None):
    base = {'attrs': False, 'user': False, 'item': False, 'rec': False, 'search': False, 'capacity': False}
    if name == 'A':
        return base
    if name == 'A_capacity':
        return {**base, 'capacity': True}
    if name in ('B', 'C', 'user_only', 'item_only'):
        return {**base, 'attrs': True, 'user': name in ('C', 'user_only'), 'item': name in ('C', 'item_only')}
    if name in ('D', 'E') and parent is not None:
        return {**parent, 'rec': True, 'search': name == 'E'}
    raise ValueError('Unsupported enhancement experiment')


def masked_mean(embedded, tokens):
    mask = tokens.ne(0).unsqueeze(-1)
    count = mask.sum(dim=1)
    pooled = (embedded * mask).sum(dim=1) / count.clamp_min(1)
    return pooled, count.squeeze(-1)


class EnhancedDCN(nn.Module):
    def __init__(self, spec, cardinalities):
        super().__init__()
        self.spec = spec
        self.tables = nn.ModuleDict()
        names = []
        if spec['attrs']:
            names.extend(['author', 'category'])
        if spec['user']:
            names.append('user')
        if spec['item'] or spec['rec'] or spec['search']:
            names.append('item')
        for name in names:
            dimension = 16 if name in ('user', 'item') else 8
            table = nn.Embedding(cardinalities[name], dimension, padding_idx=0, sparse=True)
            nn.init.normal_(table.weight, std=.02)
            with torch.no_grad():
                table.weight[:2].zero_()  # UNK is trainable, but safe when never trained.
            self.tables[name] = table
            self.register_buffer('known_'+name, torch.ones(cardinalities[name], dtype=torch.bool))
        width = 20 + 16 * spec['attrs'] + 16 * spec['user'] + 16 * spec['item']
        width += 18 * spec['rec'] + 18 * spec['search']
        self.core = FineDCN(width)
        if spec['capacity']:
            self.core.deep = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
            self.core.head = nn.Linear(width + 64, 1)

    def set_coverage(self, coverage):
        for name in self.tables:
            mask = torch.zeros_like(getattr(self, 'known_'+name))
            mask[torch.as_tensor(coverage[name], dtype=torch.long)] = True
            mask[:2] = True
            getattr(self, 'known_'+name).copy_(mask)

    def embed(self, name, tokens):
        # Even a train-vocabulary ID can be absent from a sampled/model-specific input.
        known = getattr(self, 'known_'+name)
        safe = torch.where(known[tokens], tokens, torch.ones_like(tokens))
        return self.tables[name](safe)

    def forward(self, numeric, fields, rec=None, search=None):
        values = [numeric]
        if self.spec['attrs']:
            values.extend([self.embed('author', fields[:, 2]), self.embed('category', fields[:, 3])])
        if self.spec['user']:
            values.append(self.embed('user', fields[:, 0]))
        if self.spec['item']:
            values.append(self.embed('item', fields[:, 1]))
        for enabled, tokens in [(self.spec['rec'], rec), (self.spec['search'], search)]:
            if enabled:
                if tokens is None:
                    raise ValueError('Enabled history requires tokens')
                pooled, count = masked_mean(self.embed('item', tokens), tokens)
                flags = torch.stack([count.float() / tokens.shape[1], count.eq(0).float()], dim=1)
                values.extend([pooled, flags])
        return self.core(torch.cat(values, dim=1))


class EnhancedRanker:
    def __init__(self, checkpoint, data, state_hash):
        from src.v2.train_rankers import sha
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.metadata = saved['metadata']
        if self.metadata.get('kind') != 'enhanced_dcn_v2':
            raise ValueError('Unsupported enhanced checkpoint')
        if self.metadata['data_sha256'] != sha(data.path/'metadata.json') or self.metadata['state_sha256'] != state_hash:
            raise ValueError('Enhanced checkpoint/cache/feature state mismatch')
        if data.meta['feature_state_sha256'] != state_hash:
            raise ValueError('Prepared and live numeric snapshots differ')
        self.mean = np.asarray(saved['mean'], dtype=np.float32)
        self.std = np.asarray(saved['std'], dtype=np.float32)
        self.model = EnhancedDCN(self.metadata['spec'], data.meta['cardinalities'])
        self.model.load_state_dict(saved['state_dict'], strict=True)
        self.model.eval()

    def predict(self, x, fields, rec=None, search=None):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 20 or not np.isfinite(x).all() or len(fields) != len(x):
            raise ValueError('Invalid enhanced inputs')
        output = []
        with torch.inference_mode():
            for lo in range(0, len(x), 16384):
                sl = slice(lo, lo+16384)
                numeric = torch.from_numpy((x[sl]-self.mean)/self.std)
                f = torch.tensor(np.asarray(fields[sl]), dtype=torch.long)
                r = torch.tensor(np.asarray(rec[sl]), dtype=torch.long) if rec is not None else None
                s = torch.tensor(np.asarray(search[sl]), dtype=torch.long) if search is not None else None
                output.append(torch.sigmoid(self.model(numeric, f, r, s)).numpy())
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)
