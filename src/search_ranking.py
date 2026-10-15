"""Structured-feature search-session ranking, separate from recommendation CTR."""
import json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from src.ctr_features import CACHE,FeatureState,connection
from src.ranking import metrics

OUT=Path('experiments/search_valid')
SEARCH_CACHE=CACHE/'search'
TYPES=['VIDEO','IMAGE_ATLAS','ADVERT','USER','MUSIC','LIVE','COMMODITY','UNKNOWN']
SOURCES=['TRENDING','SUGGESTION','PLACEHOLDER','USER_INPUT','RELATED_FEED_QUERY','HISTORY','UNKNOWN_PAGE']


def fetch_split(split):
    db=connection()
    rows=db.execute("SELECT user_id,item_id,timestamp_ms,label,session_id,result_type,search_source,keyword FROM events WHERE scene='search' AND split=? AND exclusion IS NULL ORDER BY timestamp_ms,user_id,session_id,id",(split,)).fetchall()
    db.close();return rows


def features(state,rows):
    numeric=np.array([r[:4] for r in rows],dtype=np.int64)
    base=state.features(numeric[:,0],numeric[:,1],numeric[:,2])
    typ=np.array([[int(r[5]==v) for v in TYPES] for r in rows],dtype=np.float32)
    source=np.array([[int(r[6]==v) for v in SOURCES] for r in rows],dtype=np.float32)
    qlen=np.array([0 if r[7]=='[]' else r[7].count(',')+1 for r in rows],dtype=np.float32)[:,None]
    groups=np.array([int(r[0])*1000000+int(r[4]) for r in rows],dtype=np.int64)
    assert all(0<=int(r[4])<1000000 for r in rows)
    return np.column_stack([base,typ,source,qlen]),numeric[:,3].astype(np.int8),groups,numeric[:,1]


def lists_metrics(y,p,groups,items):
    order=np.lexsort((items,-p,groups));g=groups[order]
    cuts=np.r_[0,np.flatnonzero(np.diff(g))+1,len(g)]
    ndcg=[];mrr=[];zero=0
    for a,b in zip(cuts[:-1],cuts[1:]):
        ys=y[order[a:b]];count=int(ys.sum())
        if count==0:zero+=1;continue
        gain=float(np.sum(ys[:10]/np.log2(np.arange(min(10,len(ys)))+2)))
        ideal=float(np.sum(1/np.log2(np.arange(min(10,count))+2)))
        ndcg.append(gain/ideal)
        first=np.flatnonzero(ys[:10]);mrr.append(1/(first[0]+1) if len(first) else 0)
    return {'sessions':len(cuts)-1,'positive_sessions':len(ndcg),'zero_click_sessions':zero,
            'ndcg_at_10_positive_sessions':float(np.mean(ndcg)),
            'ndcg_at_10_all_sessions':float(np.sum(ndcg)/(len(cuts)-1)),
            'mrr_at_10_positive_sessions':float(np.mean(mrr)),
            'ties':'score descending, item_id ascending; labels click_cnt>0'}


def prepare(split):
    SEARCH_CACHE.mkdir(exist_ok=True)
    if split=='train':
        rec=FeatureState.load(CACHE/'state.npz')
        state=FeatureState(rec.categories,len(rec.un));del rec
        rows=fetch_split('train');xs=[];ys=[];gs=[];ii=[]
        # A Beijing calendar day corresponds to a stable integer day number.
        start=0
        while start<len(rows):
            day=(rows[start][2]//1000+8*3600)//86400;end=start+1
            while end<len(rows) and (rows[end][2]//1000+8*3600)//86400==day:end+=1
            batch=rows[start:end];x,y,g,i=features(state,batch)
            xs.append(x);ys.append(y);gs.append(g);ii.append(i)
            u=np.array([r[0] for r in batch]);state.update(u,i,y)
            state.update_search(u[y>0],i[y>0]);start=end
        state.save(SEARCH_CACHE/'state.npz')
        values=[np.concatenate(v) for v in (xs,ys,gs,ii)]
    else:
        state=FeatureState.load(SEARCH_CACHE/'state.npz')
        values=features(state,fetch_split(split))
    for name,value in zip(('X','y','groups','items'),values):np.save(SEARCH_CACHE/f'{split}_{name}.npy',value)


def main():
    if OUT.exists():raise FileExistsError('Preserve existing search run')
    OUT.mkdir(parents=True);prepare('train');prepare('valid')
    x=np.load(SEARCH_CACHE/'train_X.npy',mmap_mode='r');y=np.load(SEARCH_CACHE/'train_y.npy',mmap_mode='r')
    idx=np.random.default_rng(2026).choice(len(x),min(1000000,len(x)),replace=False)
    model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,learning_rate=.06,l2_regularization=10,min_samples_leaf=100,max_bins=128,early_stopping=False,random_state=2026)
    with threadpool_limits(limits=8):model.fit(x[idx],y[idx])
    joblib.dump(model,OUT/'model.joblib')
    xv=np.load(SEARCH_CACHE/'valid_X.npy',mmap_mode='r');yv=np.load(SEARCH_CACHE/'valid_y.npy');g=np.load(SEARCH_CACHE/'valid_groups.npy');i=np.load(SEARCH_CACHE/'valid_items.npy')
    with threadpool_limits(limits=8):p=model.predict_proba(xv)[:,1]
    np.save(OUT/'predictions.npy',p)
    baseline=np.full(len(yv),float(y.mean()))
    report={'protocol':'search logged session lists; binary click_cnt>0; prior-day search statistics and request type/source/query length; no semantic text matching',
            'test_used':False,'fit_sample_rows':len(idx),'constant':metrics(yv,baseline,g),
            'model':metrics(yv,p,g),'lists':lists_metrics(yv,p,g,i),'constant_lists':lists_metrics(yv,baseline,g,i)}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:report['model'][k] for k in ('auc','gauc','logloss')},report['lists'],flush=True)


if __name__=='__main__':main()
