"""Train-only truncated-history cosine ItemCF, validation-only evaluation."""
import hashlib
import argparse
import heapq
import json
import sqlite3
import time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import csr_matrix


def rank_items(scores, items, k=100):
    # Explicit tie order, no label-dependent candidate selection.
    return [int(items[i]) for i in heapq.nsmallest(k, scores, key=lambda i: (-scores[i], int(items[i])))]


def rank_sparse(result, history, items, k=100):
    result.sort_indices()
    # Cosine diagonal is one. Only subtract where the historical item exists.
    positions=np.searchsorted(result.indices,history.indices)
    inside=positions<len(result.indices)
    positions=positions[inside]
    expected=history.indices[inside]
    result.data[positions[result.indices[positions]==expected]]-=1.0
    keep=result.data>1e-10
    indices=result.indices[keep];values=result.data[keep]
    if len(values)>k:
        threshold=np.partition(values,-k)[-k]
        keep=values>=threshold
        indices=indices[keep];values=values[keep]
    order=np.lexsort((items[indices],-values))[:k]
    return [int(i) for i in items[indices[order]]]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-cooccurrence',action='store_true')
    args=parser.parse_args()
    start=time.perf_counter()
    root=Path('data/processed/kuaisar_small_v1')
    db=sqlite3.connect((root/'dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA cache_size=-131072')
    # Most recent 30 distinct clicked items per user; same-time order is item ID.
    history=defaultdict(dict); seeds=defaultdict(dict); popular=Counter(); catalog=set()
    for u,i,y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='train' ORDER BY timestamp_ms DESC,item_id ASC,id ASC"):
        catalog.add(i)
        if y:
            popular[i]+=1
            if len(seeds[u])<30: seeds[u][i]=None
            if args.full_cooccurrence or len(history[u])<30: history[u][i]=None
    print('Training histories ready',len(history),flush=True)
    users=sorted(history); user_index={u:n for n,u in enumerate(users)}
    items=np.array(sorted({i for h in history.values() for i in h}),dtype=np.int64)
    item_index={int(i):n for n,i in enumerate(items)}
    rr=[];cc=[]
    for u,h in history.items():
        for i in h: rr.append(user_index[u]);cc.append(item_index[i])
    binary=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(users),len(items)))
    degree=np.asarray(binary.sum(axis=0)).ravel()
    normalized=binary.multiply(1/np.sqrt(degree)).tocsr()
    transposed=normalized.T.tocsr()
    sr=[];sc=[]
    for u,h in seeds.items():
        for i in h:sr.append(user_index[u]);sc.append(item_index[i])
    query=csr_matrix((np.ones(len(sr)),(sr,sc)),shape=binary.shape)
    positive=defaultdict(Counter); eval_users=set()
    for u,i,y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
        eval_users.add(u)
        if y: positive[u][i]+=1
    popularity=sorted(popular,key=lambda i:(-popular[i],i))[:100]
    hits={'itemcf':Counter(),'itemcf_popular_fill':Counter(),'popular':Counter()}
    candidate_items=set(); lengths=[]; latency=[]; warm=0; total=0; graph_positive=0
    for num,u in enumerate(sorted(eval_users),1):
        tick=time.perf_counter()
        if u in user_index:
            h=query.getrow(user_index[u])
            # h * cosine(B.T B), computed without a huge all-item similarity matrix.
            result=((h @ transposed) @ normalized).tocsr()
            ranked=rank_sparse(result,h,items)
        else: ranked=[]
        latency.append((time.perf_counter()-tick)*1000)
        lengths.append(len(ranked));candidate_items.update(ranked)
        filled=(ranked+[i for i in popularity if i not in set(ranked)])[:100]
        for name,ranking in [('itemcf',ranked),('itemcf_popular_fill',filled),('popular',popularity)]:
            for k in (50,100):
                selected=set(ranking[:k])
                hits[name][k]+=sum(n for i,n in positive[u].items() if i in selected)
        total+=sum(positive[u].values())
        warm+=sum(n for i,n in positive[u].items() if i in catalog)
        graph_positive+=sum(n for i,n in positive[u].items() if i in item_index)
        if num%2000==0: print('Evaluated users',num,flush=True)
    db.close()
    report={'protocol':{'fit':'rec train','evaluate':'rec valid','history':'last 30 distinct train clicks, binary per user','similarity':'cosine cooccurrence; diagonal zero; no neighbor pruning',
                        'cooccurrence_history':'all distinct train clicks' if args.full_cooccurrence else 'last 30 distinct train clicks',
                        'scores':'sum similarities over train history','history_filter':False,'test_used':False,
                        'recall':'positive-event weighted, same as R0','latency':'CPU per-user sparse computation including ranking, not serving benchmark'},
            'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'numpy':np.__version__,'scipy':scipy.__version__,
            'history_items':len(items),'train_catalog_items':len(catalog),'evaluated_users':len(eval_users),
            'positive_events':total,'warm_positive_events':warm,
            'cooccurrence_catalog_positive_events':graph_positive,
            'cooccurrence_catalog_recall_ceiling':graph_positive/total,
            'metrics':{name:{str(k):{'hits':counts[k],'recall':counts[k]/total,'warm_recall':counts[k]/warm} for k in (50,100)} for name,counts in hits.items()},
            'itemcf_user_coverage':sum(n>0 for n in lengths)/len(lengths),
            'itemcf_distinct_candidates':len(candidate_items),'itemcf_mean_candidates':float(np.mean(lengths)),
            'latency_ms':{str(p):float(np.percentile(latency,p)) for p in (50,95,99)},
            'elapsed_seconds':time.perf_counter()-start}
    out=Path('experiments/r1_itemcf_full_valid' if args.full_cooccurrence else 'experiments/r1_itemcf_valid');out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
