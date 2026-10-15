"""Causal click-history + category/type two-tower validation experiment."""
import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter,defaultdict,deque
from itertools import groupby
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.two_tower import contrastive_loss


def causal_histories(events, user_map, item_map, length=20):
    """Input chronological (user,item,ms); no same-time event sees its peers."""
    histories=defaultdict(lambda:deque(maxlen=length))
    samples=[]; targets=[]; users=[]
    max_history_time=defaultdict(lambda:-1)
    for timestamp,group in groupby(events,key=lambda r:r[2]):
        batch=list(group)
        for uid,iid,ts in batch:
            assert max_history_time[uid]<ts
            h=list(histories[uid])
            samples.append([0]*(length-len(h))+h)
            targets.append(item_map[iid]);users.append(user_map[uid])
        for uid,iid,ts in batch:
            histories[uid].append(item_map[iid]);max_history_time[uid]=ts
    final=np.zeros((len(user_map),length),dtype=np.int32)
    for uid,h in histories.items():
        values=list(h);final[user_map[uid],-len(values):]=values
    return np.array(users,dtype=np.int32),np.array(targets,dtype=np.int32),np.array(samples,dtype=np.int32),final


class HistoryTower(nn.Module):
    def __init__(self,users,items,categories,types,dim=32,use_attributes=True):
        super().__init__()
        self.use_attributes=use_attributes
        self.user_embedding=nn.Embedding(users,dim,sparse=True)
        self.item_embedding=nn.Embedding(items+1,dim,padding_idx=0,sparse=True)
        self.category_embedding=nn.Embedding(int(max(categories))+1,16,padding_idx=0)
        self.type_embedding=nn.Embedding(int(max(types))+1,8,padding_idx=0)
        self.item_projection=nn.Sequential(nn.Linear(dim+24,dim),nn.LayerNorm(dim))
        self.user_projection=nn.Sequential(nn.Linear(dim*2,dim),nn.LayerNorm(dim))
        self.register_buffer('categories',torch.as_tensor(categories,dtype=torch.long))
        self.register_buffer('types',torch.as_tensor(types,dtype=torch.long))
        nn.init.normal_(self.user_embedding.weight,std=.1)
        nn.init.normal_(self.item_embedding.weight,std=.1)
        with torch.no_grad():self.item_embedding.weight[0].zero_()

    def items(self,ids):
        category=self.category_embedding(self.categories[ids])
        typ=self.type_embedding(self.types[ids])
        if not self.use_attributes:
            category=category*0;typ=typ*0
        values=torch.cat([self.item_embedding(ids),category,typ],dim=-1)
        return F.normalize(self.item_projection(values),dim=-1)

    def users(self,ids,history):
        mask=(history!=0).unsqueeze(-1)
        values=self.items(history)*mask
        pooled=values.sum(dim=1)/mask.sum(dim=1).clamp_min(1)
        return F.normalize(self.user_projection(torch.cat([self.user_embedding(ids),pooled],dim=-1)),dim=-1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='r2_history_attributes_valid')
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--no-attributes',action='store_true',help='Ablation: zero category/type vectors, identical architecture')
    args=parser.parse_args()
    if args.epochs<1:raise ValueError('epochs must be positive')
    out=Path('experiments')/args.run
    if out.exists():raise FileExistsError('Choose a fresh run name')
    out.mkdir(parents=True)
    torch.manual_seed(2026);np.random.seed(2026);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=Path('data/processed/kuaisar_small_v1')
    manifest_bytes=(root/'manifest.json').read_bytes()
    assert json.loads(manifest_bytes)['status']=='ready'
    db=sqlite3.connect((root/'dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    print('Reading train-only positive histories',flush=True)
    events=db.execute("SELECT user_id,item_id,timestamp_ms FROM clean_events WHERE scene='rec' AND split='train' AND label=1 ORDER BY timestamp_ms,user_id,item_id,id").fetchall()
    user_ids=sorted({r[0] for r in events}); item_ids=sorted({r[1] for r in events})
    user_map={u:n for n,u in enumerate(user_ids)};item_map={i:n+1 for n,i in enumerate(item_ids)}
    popular_counts=Counter(r[1] for r in events)
    popular=sorted(popular_counts,key=lambda i:(-popular_counts[i],i))[:100]
    # Vocab only built over training clicked items; category/type are static metadata.
    categories=np.zeros(len(item_ids)+1,dtype=np.int64);types=np.zeros_like(categories)
    cv={};tv={}
    for iid,category,typ in db.execute('SELECT item_id,category_id,feature_type FROM items'):
        if iid in item_map:
            j=item_map[iid]
            if category not in (None,'','0'):categories[j]=cv.setdefault(category,len(cv)+1)
            if typ not in (None,'','UNKNOWN'):types[j]=tv.setdefault(typ,len(tv)+1)
    print('Building strictly earlier histories',flush=True)
    u_np,i_np,h_np,final=causal_histories(events,user_map,item_map)
    del events
    config={'seed':2026,'epochs':args.epochs,'batch_size':1024,'history_length':20,'dimension':32,'temperature':.1,
            'use_attributes':not args.no_attributes,'sparse_lr':.01,'dense_lr':.001,
            'history_rule':'strictly earlier train clicks; exclude same-time peers; fixed train history for validation',
            'features':['user_id','mean_history_item_tower','item_id','first_level_category_id','feature_type'],
            'metadata_assumption':'category/type treated as static; release has no feature snapshot times',
            'candidate_catalog':'rec train clicked items','cold_user':'same train-popular top100 fallback as ID tower',
            'selection':'fixed final epoch; no sweep','test_used':False,
            'negative_sampling':'in-batch positive-event items; same-user/same-item pairs positive',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'device':str(device),'torch':str(torch.__version__),
            'ties':'native GPU topk','history_filter':False}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    u=torch.tensor(u_np,dtype=torch.long,device=device);items=torch.tensor(i_np,dtype=torch.long,device=device)
    histories=torch.tensor(h_np,dtype=torch.long,device=device);del u_np,i_np,h_np
    final_tensor=torch.tensor(final,dtype=torch.long,device=device)
    model=HistoryTower(len(user_ids),len(item_ids),categories,types,use_attributes=not args.no_attributes).to(device)
    sparse_params=[model.user_embedding.weight,model.item_embedding.weight]
    sparse=torch.optim.SparseAdam(sparse_params,lr=.01)
    dense=torch.optim.Adam([p for p in model.parameters() if all(p is not s for s in sparse_params)],lr=.001)
    training=[];start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();order=torch.randperm(len(u),device=device);total_loss=torch.zeros((),device=device);n=0
        for lo in range(0,len(u),1024):
            idx=order[lo:lo+1024]
            if len(idx)<2:continue
            ub,ib=u[idx],items[idx]
            logits=model.users(ub,histories[idx])@model.items(ib).T/.1
            loss=contrastive_loss(logits,ub,ib)
            sparse.zero_grad(set_to_none=True);dense.zero_grad(set_to_none=True)
            loss.backward();sparse.step();dense.step();total_loss+=loss.detach()*len(idx);n+=len(idx)
            if lo and lo//1024%500==0:print('epoch',epoch+1,'records',lo,flush=True)
        value=total_loss.item()/n
        if not np.isfinite(value):raise ValueError('Nonfinite loss')
        training.append({'epoch':epoch+1,'loss':value});print(training[-1],flush=True)
    train_seconds=time.perf_counter()-start
    checkpoint=out/'model.pt'
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'user_ids':user_ids,'item_ids':item_ids,
                'final_histories':torch.tensor(final,dtype=torch.long),'category_vocab':cv,'type_vocab':tv,'config':config},checkpoint)
    positive=defaultdict(Counter);eval_users=set()
    for uid,iid,y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
        eval_users.add(uid)
        if y:positive[uid][iid]+=1
    db.close();model.eval();hits=Counter();groups=defaultdict(Counter);coverage=set();total=0
    with torch.inference_mode(),(out/'candidates.jsonl').open('w',encoding='utf-8') as handle:
        vectors=torch.cat([model.items(torch.arange(lo,min(lo+32768,len(item_ids)+1),device=device)) for lo in range(1,len(item_ids)+1,32768)])
        raw_item_ids=np.array(item_ids)
        ordered=sorted(eval_users)
        for lo in range(0,len(ordered),128):
            batch=ordered[lo:lo+128];known=[uid for uid in batch if uid in user_map];rankings={}
            if known:
                idx=torch.tensor([user_map[uid] for uid in known],device=device)
                queries=model.users(idx,final_tensor[idx])
                top=torch.topk(queries@vectors.T,k=100,dim=1).indices.cpu().numpy()
                rankings={uid:raw_item_ids[row].tolist() for uid,row in zip(known,top)}
            for uid in batch:
                ranked=rankings.get(uid,popular);coverage.update(ranked)
                assert len(set(ranked))==100
                handle.write(json.dumps({'user_id':uid,'items':ranked})+'\n')
                group=groups['known_train_click_user' if uid in user_map else 'no_train_click_user']
                group['users']+=1;group['positive_events']+=sum(positive[uid].values())
                total+=sum(positive[uid].values())
                for k in (50,100):
                    selected=set(ranked[:k]);popset=set(popular[:k])
                    h=sum(n for iid,n in positive[uid].items() if iid in selected)
                    hits[k]+=h;group[f'hits_at_{k}']+=h
                    group[f'popular_hits_at_{k}']+=sum(n for iid,n in positive[uid].items() if iid in popset)
            if lo%1024==0:print('Exact retrieval users',min(lo+128,len(ordered)),flush=True)
    digest=hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    report={'config':config,'training':training,'train_seconds':train_seconds,'train_positive_events':len(u),
            'candidate_items':len(item_ids),'eval_users':len(eval_users),'positive_events':total,
            'groups':dict(groups),'distinct_returned_items':len(coverage),
            'metrics':{str(k):{'hits':hits[k],'recall':hits[k]/total} for k in (50,100)},
            'checkpoint_sha256':digest.hexdigest()}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
