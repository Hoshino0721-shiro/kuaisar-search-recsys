"""ID-only two-tower retrieval baseline, train fit and exact valid retrieval.

This first model deliberately has no sequence features. In-batch negatives are
sampled training positives, not exposed-but-unclicked records or CTR labels.
"""
import argparse
import hashlib
import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TwoTower(nn.Module):
    def __init__(self, users, items, dim=32):
        super().__init__()
        self.user_embedding=nn.Embedding(users,dim,sparse=True)
        self.item_embedding=nn.Embedding(items,dim,sparse=True)
        self.user_projection=nn.Sequential(nn.Linear(dim,dim),nn.LayerNorm(dim))
        self.item_projection=nn.Sequential(nn.Linear(dim,dim),nn.LayerNorm(dim))
        nn.init.normal_(self.user_embedding.weight,std=.1)
        nn.init.normal_(self.item_embedding.weight,std=.1)

    def users(self,ids):
        return F.normalize(self.user_projection(self.user_embedding(ids)),dim=-1)

    def items(self,ids):
        return F.normalize(self.item_projection(self.item_embedding(ids)),dim=-1)


def contrastive_loss(logits, users, items):
    # A repeated item or another logged positive of the same user is a positive.
    positives=(users[:,None]==users[None,:]) | (items[:,None]==items[None,:])
    return (torch.logsumexp(logits,dim=1)-torch.logsumexp(logits.masked_fill(~positives,-torch.inf),dim=1)).mean()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--run',default='r2_two_tower_valid')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<2: raise ValueError('Invalid training size')
    seed=2026; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path('experiments')/args.run
    if out.exists(): raise FileExistsError('Choose a new run name; existing experiment preserved')
    out.mkdir(parents=True)
    root=Path('data/processed/kuaisar_small_v1')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='ready': raise ValueError('Dataset is not ready')
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    db=sqlite3.connect((root/'dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    pairs=np.array(db.execute("SELECT user_id,item_id FROM clean_events WHERE scene='rec' AND split='train' AND label=1 ORDER BY id").fetchall(),dtype=np.int64)
    user_ids,user_inverse=np.unique(pairs[:,0],return_inverse=True)
    item_ids,item_inverse=np.unique(pairs[:,1],return_inverse=True)
    u=torch.from_numpy(user_inverse).to(device);i=torch.from_numpy(item_inverse).to(device)
    item_counts=Counter(pairs[:,1].tolist())
    popularity=sorted(item_counts,key=lambda x:(-item_counts[x],x))[:100]
    del pairs
    model=TwoTower(len(user_ids),len(item_ids)).to(device)
    sparse=torch.optim.SparseAdam([model.user_embedding.weight,model.item_embedding.weight],lr=.01)
    dense=torch.optim.Adam(list(model.user_projection.parameters())+list(model.item_projection.parameters()),lr=.001)
    config={'seed':seed,'epochs':args.epochs,'batch_size':args.batch_size,'dimension':32,'temperature':.1,
            'sparse_lr':.01,'dense_lr':.001,'fit':'rec train positive events',
            'candidate_catalog':'all distinct rec train clicked items; no unseen item vectors',
            'negative_sampling':'in-batch positive-event items; same-user or same-item pairs are positives',
            'query_features':'user ID only; no validation updates','retrieval':'exact cosine top100; no history filtering',
            'cold_user':'train-popular top100 fallback', 'ties':'GPU topk native, not guaranteed item-id order',
            'selection':'fixed final epoch; no checkpoint selection or parameter sweep','test_used':False,
            'device':str(device),'torch':torch.__version__,'numpy':np.__version__,
            'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
            'source_sha256':source_hash,'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Training',len(u),'positive events;',len(item_ids),'items on',device,flush=True)
    training=[]; start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();order=torch.randperm(len(u),device=device);loss_sum=torch.zeros((),device=device);steps=0
        for lo in range(0,len(order),args.batch_size):
            idx=order[lo:lo+args.batch_size]
            if len(idx)<2:continue
            ub,ib=u[idx],i[idx]
            loss=contrastive_loss(model.users(ub)@model.items(ib).T/.1,ub,ib)
            sparse.zero_grad(set_to_none=True);dense.zero_grad(set_to_none=True)
            loss.backward();sparse.step();dense.step();loss_sum+=loss.detach()*len(idx);steps+=len(idx)
            if lo and lo//args.batch_size%500==0: print('epoch',epoch+1,'records',lo,flush=True)
        torch.cuda.synchronize() if device.type=='cuda' else None
        value=float(loss_sum.item()/steps)
        if not np.isfinite(value):raise ValueError('Nonfinite training loss')
        training.append({'epoch':epoch+1,'loss':value})
        print(training[-1],flush=True)
    train_seconds=time.perf_counter()-start
    checkpoint=out/'model.pt'
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'user_ids':user_ids.tolist(),'item_ids':item_ids.tolist(),'config':config},checkpoint)
    positives=defaultdict(Counter);eval_users=set()
    for uid,iid,label in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
        eval_users.add(uid)
        if label:positives[uid][iid]+=1
    train_catalog={x[0] for x in db.execute("SELECT DISTINCT item_id FROM clean_events WHERE scene='rec' AND split='train'")}
    db.close()
    ids={int(uid):n for n,uid in enumerate(user_ids)};catalog=set(item_ids.tolist())
    model.eval();hits=Counter();baseline=Counter();total=0;warm=0;eligible=0;cold_users=0;coverage=set()
    start=time.perf_counter()
    with torch.inference_mode():
        vectors=torch.cat([model.items(torch.arange(lo,min(lo+32768,len(item_ids)),device=device)) for lo in range(0,len(item_ids),32768)])
        ordered=sorted(eval_users)
        with (out/'candidates.jsonl').open('w',encoding='utf-8') as handle:
            for lo in range(0,len(ordered),128):
                batch=ordered[lo:lo+128];known=[uid for uid in batch if uid in ids]
                ranking={}
                if known:
                    query=model.users(torch.tensor([ids[uid] for uid in known],device=device))
                    top=torch.topk(query@vectors.T,k=min(100,len(item_ids)),dim=1).indices.cpu().numpy()
                    ranking={uid:item_ids[row].tolist() for uid,row in zip(known,top)}
                for uid in batch:
                    cold_users+=uid not in ids
                    ranked=ranking.get(uid,popularity);coverage.update(ranked)
                    handle.write(json.dumps({'user_id':uid,'items':ranked})+'\n')
                    for k in (50,100):
                        selected=set(ranked[:k]);popset=set(popularity[:k])
                        hits[k]+=sum(n for iid,n in positives[uid].items() if iid in selected)
                        baseline[k]+=sum(n for iid,n in positives[uid].items() if iid in popset)
                    total+=sum(positives[uid].values())
                    warm+=sum(n for iid,n in positives[uid].items() if iid in train_catalog)
                    eligible+=sum(n for iid,n in positives[uid].items() if iid in catalog)
                if lo%1024==0:print('Exact retrieval users',min(lo+128,len(ordered)),flush=True)
    report={'config':config,'training':training,'train_positive_events':len(u),'candidate_items':len(item_ids),
            'eval_users':len(eval_users),'cold_users':cold_users,'positive_events':total,'warm_positive_events':warm,
            'catalog_positive_event_ceiling':eligible/total,'distinct_returned_items':len(coverage),
            'metrics':{str(k):{'hits':hits[k],'recall':hits[k]/total,'warm_recall':hits[k]/warm,
                               'popular_hits':baseline[k]} for k in (50,100)},
            'train_seconds':train_seconds,'exact_retrieval_seconds':time.perf_counter()-start,
            'timing_scope':'retrieval includes vector generation, scoring, topk, result serialization and evaluation; not serving latency',
            'checkpoint_sha256':None}
    digest=hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    report['checkpoint_sha256']=digest.hexdigest()
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
