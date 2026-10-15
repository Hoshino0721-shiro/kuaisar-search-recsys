"""Recall fusion and CTR scoring over candidates, user-period offline protocol."""
import hashlib
import json
import sqlite3
import time
from collections import Counter,defaultdict
from pathlib import Path
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.ctr_features import FeatureState,CACHE,DATA,connection
from src.ranking import GateRanker,torch_predict,OUT as RANK_OUT
from src.history_tower import HistoryTower

OUT=Path('experiments/pipeline_valid')


def read_candidates(path):
    result={}
    for line in Path(path).open():
        row=json.loads(line);result[row['user_id']]=row['items']
    return result


def rrf(sources,weights,k=300):
    scores=defaultdict(float)
    for items,w in zip(sources,weights):
        if w:
            for rank,item in enumerate(items,1):scores[item]+=w/(60+rank)
    return sorted(scores,key=lambda i:(-scores[i],i))[:k]


def rerank(items,scores,categories,penalty=0.,k=100):
    ids=np.asarray(items);values=np.asarray(scores,dtype=float)
    if penalty==0:return ids[np.lexsort((ids,-values))[:k]].tolist()
    order=np.argsort(ids);ids=ids[order];values=values[order].copy()
    cats=categories[ids];counts=np.zeros(int(cats.max())+1,dtype=int);chosen=[]
    for _ in range(min(k,len(ids))):
        adjusted=values-penalty*np.log1p(counts[cats])*(cats>0)
        j=int(np.argmax(adjusted));chosen.append(int(ids[j]));values[j]=-np.inf;counts[cats[j]]+=1
    return chosen


def quality(rankings,positive,categories):
    total=sum(sum(p.values()) for p in positive.values());hits=Counter();coverage=set();ndcg=[];mrr=[];diversity=[]
    for uid,ranked in rankings.items():
        coverage.update(ranked);relevant=set(positive[uid]);rank10=ranked[:10]
        for k in (50,100):
            selected=set(ranked[:k])
            hits[k]+=sum(n for item,n in positive[uid].items() if item in selected)
        cats={int(categories[i]) for i in rank10 if categories[i]>0}
        diversity.append(len(cats)/max(1,len(rank10)))
        if relevant:
            dcg=sum(1/np.log2(j+2) for j,item in enumerate(rank10) if item in relevant)
            ideal=sum(1/np.log2(j+2) for j in range(min(10,len(relevant))))
            ndcg.append(dcg/ideal)
            first=next((j for j,item in enumerate(rank10,1) if item in relevant),None)
            mrr.append(1/first if first else 0)
    return {'positive_events':total,'hits_at_50':hits[50],'hits_at_100':hits[100],
            'recall_at_50':hits[50]/total,'recall_at_100':hits[100]/total,
            'user_period_observed_positive_ndcg_at_10':float(np.mean(ndcg)),
            'user_period_observed_positive_mrr_at_10':float(np.mean(mrr)),
            'category_diversity_at_10':float(np.mean(diversity)),'distinct_candidates':len(coverage)}


class CandidateScorer:
    def __init__(self,name):
        torch.set_num_threads(4)
        self.name=name
        self.scaler=joblib.load(RANK_OUT/'scaler.joblib')
        if name in ('hgb_rec','hgb_search','lr'):self.model=joblib.load(RANK_OUT/f'{name}.joblib')
        elif name in ('mlp_concat','mlp_gate'):
            saved=torch.load(RANK_OUT/f'{name}.pt',map_location='cpu',weights_only=True)
            self.model=GateRanker(saved['gated']);self.model.load_state_dict(saved['state_dict']);self.model.eval()
        else:self.model=None

    def predict(self,x):
        if self.name=='constant':return np.full(len(x),.4972061833392528)
        if self.name=='item_ctr':return np.asarray(x[:,2])
        with threadpool_limits(limits=8):
            if self.name.startswith('hgb'):return self.model.predict_proba(x[:,:16] if self.name=='hgb_rec' else x)[:,1]
            if self.name=='lr':return self.model.predict_proba(self.scaler.transform(x))[:,1]
            return torch_predict(self.model,x,self.scaler,torch.device('cpu'))


def train_lists(db,state):
    popular=np.flatnonzero(state.ip>0).tolist()
    popular=sorted(popular,key=lambda i:(-state.ip[i],i))[:200]
    recent=[r[0] for r in db.execute("SELECT item_id,count(*) AS n FROM events WHERE scene='rec' AND split='train' AND exclusion IS NULL AND label=1 AND timestamp_ms>=1685203200000 GROUP BY item_id ORDER BY n DESC,item_id ASC LIMIT 200")]
    return popular,recent


def tower_for_users(users):
    saved=torch.load('experiments/r2_history_attributes_valid/model.pt',map_location='cpu',weights_only=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=HistoryTower(len(saved['user_ids']),len(saved['item_ids']),saved['state_dict']['categories'],saved['state_dict']['types'],use_attributes=saved['config'].get('use_attributes',True))
    model.load_state_dict(saved['state_dict']);model.to(device).eval()
    mapping={uid:i for i,uid in enumerate(saved['user_ids'])};item_ids=np.array(saved['item_ids'])
    histories=saved['final_histories'].to(device);result={}
    with torch.inference_mode():
        vectors=torch.cat([model.items(torch.arange(lo,min(lo+32768,len(item_ids)+1),device=device)) for lo in range(1,len(item_ids)+1,32768)])
        known=sorted(u for u in users if u in mapping)
        for lo in range(0,len(known),128):
            group=known[lo:lo+128];indices=torch.tensor([mapping[u] for u in group],device=device)
            top=torch.topk(model.users(indices,histories[indices])@vectors.T,100,dim=1).indices.cpu().numpy()
            result.update({u:item_ids[row].tolist() for u,row in zip(group,top)})
    return result


def make_sources(users,popular,recent,tower):
    variants={name:{} for name in ['popular','recent','rrf_equal','rrf_recent','rrf_tower','quota_recent']}
    for uid in users:
        t=tower.get(uid,popular[:100])
        variants['popular'][uid]=popular[:100];variants['recent'][uid]=recent[:100]
        variants['rrf_equal'][uid]=rrf([popular,recent,t],[1,1,0],100)
        variants['rrf_recent'][uid]=rrf([popular,recent,t],[1,2,.25],100)
        variants['rrf_tower'][uid]=rrf([popular,recent,t],[1,1,1],100)
        seed=recent[:70]+popular[:30]
        variants['quota_recent'][uid]=list(dict.fromkeys(seed+recent+popular))[:100]
    return variants


def rank_union(users,popular,recent,tower,state,scorer,timestamp):
    candidates={u:rrf([popular,recent,tower.get(u,popular[:100])],[1,1,1],300) for u in users}
    uid_array=[];item_array=[];offset=[0]
    for u in users:
        uid_array.extend([u]*len(candidates[u]));item_array.extend(candidates[u]);offset.append(len(item_array))
    u=np.array(uid_array);i=np.array(item_array);t=np.full(len(i),timestamp,dtype=np.int64)
    scores=[]
    for lo in range(0,len(i),100000):
        scores.append(scorer.predict(state.features(u[lo:lo+100000],i[lo:lo+100000],t[lo:lo+100000])))
        if lo%500000==0:print('Scored candidate rows',min(lo+100000,len(i)),flush=True)
    scores=np.concatenate(scores)
    return candidates,scores,offset


def main():
    if (OUT/'metrics.json').exists():raise FileExistsError('Preserve existing pipeline validation')
    OUT.mkdir(parents=True,exist_ok=True)
    state=FeatureState.load(CACHE/'state.npz');db=connection()
    popular,recent=train_lists(db,state);positive=defaultdict(Counter);users=set()
    for uid,iid,y in db.execute("SELECT user_id,item_id,label FROM events WHERE scene='rec' AND split='valid' AND exclusion IS NULL"):
        users.add(uid)
        if y:positive[uid][iid]+=1
    db.close();users=sorted(users)
    tower=read_candidates('experiments/r2_history_attributes_valid/candidates.jsonl')
    variants=make_sources(users,popular,recent,tower);print('Candidate sources ready',flush=True)
    result={name:quality(rankings,positive,state.categories) for name,rankings in variants.items()}
    print('Recall source evaluation ready',flush=True)
    selected_ranker=json.loads((RANK_OUT/'metrics.json').read_text())['selected']
    scorer=CandidateScorer(selected_ranker);start=time.perf_counter()
    candidates,scores,offset=rank_union(users,popular,recent,tower,state,scorer,1685289600000)
    for penalty in (0.,.01,.03):
        name=f'ctr_rerank_{penalty}'
        variants[name]={u:rerank(candidates[u],scores[offset[j]:offset[j+1]],state.categories,penalty) for j,u in enumerate(users)}
        result[name]=quality(variants[name],positive,state.categories)
        print(name,result[name],flush=True)
    # Selection is explicit, frozen before final test. NDCG and recall tie-break, then name.
    selected=max(result,key=lambda name:(result[name]['recall_at_100'],result[name]['user_period_observed_positive_ndcg_at_10'],name))
    selected_rerank=max((n for n in result if n.startswith('ctr_rerank_')),key=lambda n:result[n]['user_period_observed_positive_ndcg_at_10'])
    config={'split':'rec valid','protocol':'one fixed recommendation list per user at train cutoff; positive events in next validation period',
            'candidate_lists':'train global top200; last train day top200; history-attribute tower top100',
            'ranking_model':selected_ranker,'selection':'max Recall@100, tie NDCG@10; frozen before test',
            'selected_retrieval':selected,'selected_ctr_pipeline':selected_rerank,'test_used':False,
            'rerank_selection':'max observed-positive user-period NDCG@10 among three penalties',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report={'config':config,'variants':result,'candidate_scoring_and_evaluation_seconds':time.perf_counter()-start}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name in {selected,selected_rerank}:
        with (OUT/f'{name}_candidates.jsonl').open('w') as h:
            for u,ranked in variants[name].items():h.write(json.dumps({'user_id':u,'items':ranked})+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
