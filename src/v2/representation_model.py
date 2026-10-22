"""Learned categorical/content item vectors; optional frozen-vector interest head.

This is not an LLM embedding and not a reproduction claim of the full DIN paper.
The interest ablation uses a frozen trained item encoder for CPU feasibility.
"""
import numpy as np
import torch
from torch import nn

from src.v2.models import FineDCN


def masked_mean(values, mask):
    return (values*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)


class RepresentationDCN(nn.Module):
    def __init__(self, cardinalities, content, dimension=16, id_dropout=.1):
        super().__init__()
        self.content = content
        self.dimension = dimension
        self.id_dropout = id_dropout
        self.tables = nn.ModuleDict({name: nn.Embedding(cardinalities[name], width, padding_idx=0, sparse=True)
            for name, width in [('user', 8), ('item', 16), ('author', 8), ('category', 8), ('word', 8)]})
        for table in self.tables.values():
            nn.init.normal_(table.weight, std=.02)
            with torch.no_grad(): table.weight[:2].zero_()
        for name, table in self.tables.items():
            seen = torch.zeros(table.num_embeddings, dtype=torch.bool)
            seen[:2] = True  # Reserved PAD and zero-initialized/trainable UNK.
            self.register_buffer('seen_'+name, seen)
        # Same allocated dimensions in ID/attribute and content arms.
        self.item_projection = nn.Sequential(nn.Linear(40, dimension), nn.ReLU())
        self.core = FineDCN(34+8+dimension)

    def embed(self, name, tokens):
        seen = getattr(self, 'seen_'+name)
        if self.training:
            seen[tokens] = True
            safe = tokens
        else:
            safe = torch.where(seen[tokens], tokens, torch.ones_like(tokens))
        return self.tables[name](safe)

    def item_vector(self, fields, words):
        item_ids = fields[:, 0]
        if self.training and self.id_dropout:
            item_ids = torch.where(torch.rand(len(item_ids)) < self.id_dropout,
                                   torch.ones_like(item_ids), item_ids)
        item = self.embed('item', item_ids)
        author = self.embed('author', fields[:, 1])
        category = self.embed('category', fields[:, 2])
        text = (masked_mean(self.embed('word', words), words.ne(0)) if self.content
                else torch.zeros((len(fields), 8), dtype=item.dtype))
        return self.item_projection(torch.cat([item, author, category, text], dim=1))

    def forward(self, numeric, users, fields, words):
        user_ids = users
        if self.training and self.id_dropout:
            user_ids = torch.where(torch.rand(len(users)) < self.id_dropout, torch.ones_like(users), users)
        item = self.item_vector(fields, words)
        return self.core(torch.cat([numeric, self.embed('user', user_ids), item], dim=1))


class InterestResidual(nn.Module):
    """Candidate-related DIN-style activation vs capacity-matched mean pooling.

    Item vectors and base scores are frozen from the content arm. Only the
    interest network is trained; do not call this end-to-end encoder fine-tuning.
    """
    def __init__(self, dimension=16, attention=True):
        super().__init__()
        self.attention = attention
        self.activation = nn.Sequential(nn.Linear(4*dimension, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(34+2*dimension+2, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def interest(self, candidate, history, mask):
        if not self.attention:
            return masked_mean(history, mask)
        q = candidate.unsqueeze(1).expand_as(history)
        logits = self.activation(torch.cat([q, history, q-history, q*history], dim=-1)).squeeze(-1)
        # Explicitly handle all-PAD contexts: their interest vector is zero.
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)*mask
        weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-8)
        return (history*weights.unsqueeze(-1)).sum(1)

    def forward(self, numeric, base_logits, candidate, history, mask):
        interest = self.interest(candidate, history, mask)
        flags = torch.stack([mask.float().mean(1), ~mask.any(1)], dim=1)
        correction = self.head(torch.cat([numeric, candidate, interest, flags], dim=1)).squeeze(-1)
        return base_logits+correction


def dense_and_sparse(model):
    sparse = list(model.tables.parameters()) if hasattr(model, 'tables') else []
    sparse_ids = {id(parameter) for parameter in sparse}
    return [p for p in model.parameters() if id(p) not in sparse_ids], sparse


class RepresentationScorer:
    def __init__(self, checkpoint, data):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.meta = saved['metadata']
        if self.meta.get('kind') != 'representation_dcn_v2':
            raise ValueError('Expected safe-coverage representation v2; preserve provisional v1 weights')
        self.mean = np.asarray(saved['mean'], np.float32)
        self.std = np.asarray(saved['std'], np.float32)
        self.model = RepresentationDCN(data.meta['cardinalities'], self.meta['content'], self.meta['dimension'])
        self.model.load_state_dict(saved['state_dict'])
        self.model.eval()
        self.data = data

    def logits(self, numeric, users, items):
        from src.v2.route_attribution import normalize
        result = []
        with torch.inference_mode():
            for lo in range(0, len(items), 8192):
                sl = slice(lo, lo+8192)
                x = torch.from_numpy(normalize(numeric[sl], self.mean, self.std, list(range(28, 34))))
                u = self.data.user_lookup[torch.tensor(np.asarray(users[sl]), dtype=torch.long)]
                ids = torch.tensor(np.asarray(items[sl]), dtype=torch.long)
                result.append(self.model(x, u, self.data.item_fields[ids], self.data.caption[ids]).numpy())
        return np.concatenate(result)

    def predict(self, numeric, users, items):
        return torch.sigmoid(torch.from_numpy(self.logits(numeric, users, items))).numpy()
