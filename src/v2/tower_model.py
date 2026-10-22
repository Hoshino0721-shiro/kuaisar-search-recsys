"""Feature-rich retrieval towers and explicit in-batch candidate semantics."""
import torch
import hashlib
from pathlib import Path
from torch import nn
from torch.nn import functional as F


def model_source_digest():
    source=Path(__file__).read_text(encoding='utf-8').replace('\r\n','\n')
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


class FeatureTwoTower(nn.Module):
    def __init__(self, cardinalities, config, item_fields):
        super().__init__()
        self.config = config
        dim = config.get('dimension',64)
        if dim < 2 or not 0 <= config.get('id_dropout',.1) < 1:
            raise ValueError('Invalid embedding dimension or ID dropout')
        widths = {'user':16,'item':32,'author':16,'category':8,'type':4}
        self.embeddings = nn.ModuleDict({name+'_embedding':nn.Embedding(cardinalities[name],width,padding_idx=0,sparse=True)
                                         for name,width in widths.items()})
        for embedding in self.embeddings.values():
            nn.init.normal_(embedding.weight,std=.05)
            with torch.no_grad():
                embedding.weight[0].zero_()
                if config.get('safe_embeddings', False):
                    embedding.weight[1].zero_()
        if config.get('safe_embeddings', False):
            for name, embedding in self.embeddings.items():
                seen = torch.zeros(embedding.num_embeddings, dtype=torch.bool)
                seen[:2] = True
                self.register_buffer('seen_'+name, seen)
        fields = torch.as_tensor(item_fields,dtype=torch.long)
        if fields.shape != (cardinalities['item'],3):
            raise ValueError('Item metadata shape mismatch')
        self.register_buffer('item_fields',fields)
        self.item_projection = nn.Sequential(nn.Linear(60,128),nn.ReLU(),nn.Linear(128,dim))
        self.user_projection = nn.Sequential(nn.Linear(16+2*dim+2,128),nn.ReLU(),nn.Linear(128,dim))

    def _id(self, ids):
        if not self.training:
            return ids
        drop = (torch.rand(ids.shape,device=ids.device)<self.config.get('id_dropout',.1)) & (ids>=2)
        return ids.masked_fill(drop,1)

    def _embed(self, name, ids, enabled=True):
        table = self.embeddings[name+'_embedding']
        if not enabled:
            return table.weight.new_zeros((*ids.shape, table.embedding_dim))
        if self.config.get('safe_embeddings', False):
            seen = getattr(self, 'seen_'+name+'_embedding')
            if self.training:
                seen[ids] = True
            else:
                ids = torch.where(seen[ids], ids, torch.ones_like(ids))
        return table(ids)

    def items(self, ids, attributes=None):
        fields = self.item_fields[ids] if attributes is None else attributes
        if fields.shape!=(*ids.shape,3):
            raise ValueError('Expected author/category/type attributes for each item')
        return self._project_items(self._id(ids), fields)*(ids!=0).unsqueeze(-1)

    def _project_items(self, ids, fields):
        parts = [self._embed('item',ids)]
        for j,name in enumerate(('author','category','type')):
            parts.append(self._embed(name, fields[...,j], self.config.get('use_attributes',True)))
        encoded = F.normalize(self.item_projection(torch.cat(parts,dim=-1)),dim=-1)
        return encoded

    def _pool(self, history, enabled):
        if not enabled:
            reference=self.embeddings['item_embedding'].weight
            return reference.new_zeros((len(history),self.config.get('dimension',64))),reference.new_zeros(len(history))
        mask = history!=0
        if self.config.get('deduplicate_history_compute', False):
            # Apply dropout PER OCCURRENCE before deduplication. Attribute inputs
            # remain those of the original item even when its ID becomes UNK.
            # Sharing the deterministic projection does not share dropout draws.
            dropped = self._id(history)
            keys = history*2+dropped.ne(history).long()
            unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            original = unique//2
            ids = torch.where(unique.remainder(2).bool(), torch.ones_like(original), original)
            encoded = self._project_items(ids, self.item_fields[original])
            vectors = encoded[inverse]*(history!=0).unsqueeze(-1)
        else:
            vectors = self.items(history)
        pooled = (vectors*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp_min(1).unsqueeze(-1)
        length = torch.log1p(mask.sum(1).float())
        return pooled,length

    def users(self, ids, rec_history, search_history):
        user = self._embed('user',self._id(ids),self.config.get('use_user_id',True))
        rec,rlen = self._pool(rec_history,self.config.get('use_history',True))
        search,slen = self._pool(search_history,self.config.get('use_search',False))
        return F.normalize(self.user_projection(torch.cat([user,rec,search,rlen[:,None],slen[:,None]],dim=-1)),dim=-1)

    def sparse_parameters(self):
        return [e.weight for e in self.embeddings.values()]


def in_batch_loss(queries, candidates, target_tokens, rec_history, search_history,
                  temperature=.1, logq=None, mask_history=True):
    """Candidates correspond to sorted unique target_tokens, not raw batch columns.

    Duplicate target IDs share one class. Same-user events are NOT automatically
    positives of one another. Only already available histories can mask known
    positives out of negatives. Unobserved candidates remain sampled contrasts,
    not fabricated exposure/click labels. logQ, if used, is an approximation.
    """
    if temperature<=0:
        raise ValueError('Temperature must be positive')
    unique,labels = torch.unique(target_tokens,sorted=True,return_inverse=True)
    if not len(unique) or bool((unique<2).any()):
        raise ValueError('Targets must be nonempty known training item IDs')
    if candidates.shape[0]!=len(unique) or queries.shape[0]!=len(target_tokens):
        raise ValueError('Candidate batch must encode unique target IDs')
    logits = queries@candidates.T/temperature
    if logq is not None:
        logits = logits-logq[unique][None,:]
    if mask_history:
        history = torch.cat([rec_history,search_history],dim=1)
        positions = torch.searchsorted(unique,history)
        safe = positions.clamp_max(len(unique)-1)
        known = (positions<len(unique)) & (unique[safe]==history) & (history>=2)
        rows = torch.arange(len(history),device=history.device)[:,None].expand_as(history)
        mask = torch.zeros_like(logits,dtype=torch.bool)
        mask[rows[known],safe[known]] = True
        mask[torch.arange(len(labels),device=labels.device),labels] = False
        logits = logits.masked_fill(mask,-torch.inf)
    return F.cross_entropy(logits,labels)
