"""Independent SQL feature-prefix replay and published experiment consistency."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import numpy as np
import torch
from src.ctr_features import CACHE, connection
from src.itemcf_index import ItemCFIndex, DAY_MS
from src.v2.temporal_route_train import verify_cache, FactorialRanker
from src.v2.temporal_route_data import write_json
from src.v2.train_rankers import sha
from src.v2.train_rankers import Ranker
from src.v2.cascade import Candidate, Cascade, Budget
from src.ctr_features import FeatureState


def sql_counts(db,items,lower,upper):
    placeholders=','.join('?' for _ in items)
    rows=db.execute("SELECT item_id,count(*),sum(label) FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<? AND item_id IN ("+placeholders+") GROUP BY item_id",[lower,upper]+items)
    return {int(i):(int(n),int(p)) for i,n,p in rows}


def audit(cache,output,models=None,evaluation=None):
    if output.exists():
        raise FileExistsError('Existing audit preserved')
    meta=verify_cache(cache);torch.set_num_threads(4)
    x=np.load(cache/'train_X.npy',mmap_mode='r');idx=np.load(cache/'sample_indices.npy')
    days=np.load(cache/'train_days.npy');items=np.load(CACHE/'train_items.npy',mmap_mode='r')[idx]
    users=np.load(CACHE/'train_users.npy',mmap_mode='r')[idx]
    base=np.load(CACHE/'train_X.npy',mmap_mode='r')
    if not np.array_equal(x[:,:20],base[idx]):
        raise ValueError('Original training columns changed')
    vx=np.load(cache/'valid_X.npy',mmap_mode='r')
    if not np.array_equal(vx[:,:20],np.load(CACHE/'valid_X.npy',mmap_mode='r')):
        raise ValueError('Original validation columns changed')
    vi=np.load(CACHE/'valid_items.npy');vu=np.load(CACHE/'valid_users.npy')
    rng=np.random.default_rng(2026);checks=[];max_error=0.
    db=connection()
    try:
        for cutoff in sorted(set(days.tolist())|{meta['cutoff_ms']}):
            valid=cutoff==meta['cutoff_ms']
            eligible=np.arange(len(vx)) if valid else np.flatnonzero(days==cutoff)
            selected=rng.choice(eligible,min(32,len(eligible)),replace=False)
            raw_i=vi[selected] if valid else items[selected]
            raw_u=vu[selected] if valid else users[selected]
            raw_x=vx[selected] if valid else x[selected]
            unique=sorted(set(int(i) for i in raw_i))
            one=sql_counts(db,unique,cutoff-DAY_MS,cutoff)
            three=sql_counts(db,unique,cutoff-3*DAY_MS,cutoff)
            total=sql_counts(db,unique,0,cutoff)
            n,p=db.execute("SELECT count(*),coalesce(sum(label),0) FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms<?",(cutoff,)).fetchone()
            prior=p/n if n else .5
            recent=[int(i) for i,c in db.execute("SELECT item_id,sum(label) c FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<? GROUP BY item_id HAVING c>0 ORDER BY c DESC,item_id LIMIT ?",(cutoff-DAY_MS,cutoff,meta['config']['recent_k']))]
            recent_rank={i:r for r,i in enumerate(recent,1)}
            record=next((r for r in meta['days'] if r['cutoff_ms']==cutoff),None)
            path=meta['config']['final_index'] if valid else record['index_path']
            index=ItemCFIndex(Path(str(path).replace('\\','/')),meta['manifest_sha256']) if path else None
            try:
                if index:
                    if index.metadata['cutoff_ms']!=cutoff or np.any(index.seed_times>=cutoff):
                        raise ValueError('Noncausal index seeds')
                    expected=np.fromiter((r[0] for r in db.execute("SELECT DISTINCT item_id FROM clean_events WHERE scene='rec' AND split='train' AND label=1 AND timestamp_ms<? ORDER BY item_id",(cutoff,))),dtype=np.int64)
                    if not np.array_equal(index.items,expected):
                        raise ValueError('Prefix item catalog mismatch')
                for row,i,u in zip(raw_x,raw_i,raw_u):
                    i,u=int(i),int(u)
                    a,b=one.get(i,(0,0));c,d=three.get(i,(0,0));_,t=total.get(i,(0,0))
                    expected_t=[np.log1p(a),np.log1p(b),(b+20*prior)/(a+20),np.log1p(c),np.log1p(d),(d+20*prior)/(c+20),b/(t+1),np.log1p(b)-np.log1p((d-b)/2)]
                    cf=index.query(u,meta['config']['itemcf_k']) if index else {'items':[],'scores':[]}
                    cr={j:r for r,j in enumerate(cf['items'],1)};cs=dict(zip(cf['items'],cf['scores']))
                    r,s=recent_rank.get(i),cr.get(i)
                    rr=60/(60+r) if r else 0;cc=60/(60+s) if s else 0
                    expected_r=[int(r is not None),int(s is not None),rr,cc,np.log1p(cs.get(i,0)),rr+cc]
                    error=float(np.max(np.abs(row[20:]-np.asarray(expected_t+expected_r,dtype=np.float32))))
                    max_error=max(max_error,error)
                    if error>1e-6:
                        raise ValueError('SQL/route replay differs from cache')
                checks.append({'cutoff_ms':cutoff,'sampled_rows':len(selected),'catalog_and_seed_prefix_checked':True})
            finally:
                if index:index.close()
            print(f'Audited feature prefix {len(checks)}/8',flush=True)
    finally:
        db.close()
    result={'passed':True,'test_used':False,'source_sha256':sha(__file__),
            'cache_metadata_sha256':sha(cache/'metadata.json'),'prefix_checks':checks,
            'temporal_route_max_abs_error':max_error,'original_numeric_columns_exact':True,
            'scope':'SQL feature replay on 32 rows per cutoff; all prefix catalogs and seed times, not exhaustive graph rederivation'}
    if models:
        report=json.loads((models/'metrics.json').read_text(encoding='utf-8'))
        sample=rng.choice(len(vx),4096,replace=False)
        matched=defaultdict(set)
        for name,record in report['runs'].items():
            checkpoint=models/name/'model.pt'
            if sha(checkpoint)!=record['checkpoint_sha256']:
                raise ValueError('Checkpoint digest mismatch')
            model=FactorialRanker(checkpoint)
            expected=np.load(models/name/'valid_predictions.npy',mmap_mode='r')[sample]
            if not np.allclose(model.predict(vx[sample]),expected,rtol=0,atol=2e-7):
                raise ValueError('Model replay prediction differs')
            if model.metadata['epochs']!=meta['config']['epochs']:
                raise ValueError('Checkpoint epoch changed')
            matched[model.metadata['seed']].add((model.metadata['initial_parameters_sha256'],
                    model.metadata['training_order_sha256'],record['parameters']))
        if any(len(v)!=1 for v in matched.values()):
            raise ValueError('Initialization, orders or capacity differ across factorial arms')
        result['models_checked']=len(report['runs'])
        result['all_arms_matched_initialization_order_and_capacity']=True
    if evaluation:
        ev=json.loads((evaluation/'metrics.json').read_text(encoding='utf-8'))
        pool_users=np.load(cache/'pool_users.npy')
        raw_items=np.load(cache/'pool_items.npy',mmap_mode='r')
        pool_x=np.load(cache/'pool_X.npy',mmap_mode='r')
        state=FeatureState.load(CACHE/'state.npz')
        coarse=Ranker(meta['config']['coarse_model'])
        selected_users=rng.choice(len(pool_users),16,replace=False)
        positive=defaultdict(Counter)
        db=connection()
        try:
            for u,i,n in db.execute("SELECT user_id,item_id,count(*) FROM clean_events WHERE scene='rec' AND split='valid' AND label=1 GROUP BY user_id,item_id"):
                positive[u][i]=n
        finally:
            db.close()
        stage_replays=0
        for name,record in ev['runs'].items():
            arrays=np.load(evaluation/(name+'_per_user.npz'))
            if not np.array_equal(arrays['users'],pool_users):
                raise ValueError('User alignment differs')
            for stage in record['stages']:
                h=arrays[stage+'_hits'];n=arrays[stage+'_ndcg']
                if int(h.sum())!=record['stages'][stage]['all']['hit_events']:
                    raise ValueError('Hit aggregation mismatch')
                if abs(float(np.nanmean(n))-record['stages'][stage]['all']['ndcg_at20'])>1e-12:
                    raise ValueError('NDCG aggregation mismatch')
            scores=np.load(evaluation/(name+'_scores.npy'),mmap_mode='r')
            model=FactorialRanker(models/name/'model.pt')
            for pos in selected_users:
                uid=int(pool_users[pos]);raw=np.asarray(raw_items[pos]);p=np.asarray(scores[pos])
                if not np.allclose(model.predict(pool_x[pos]),p,rtol=0,atol=2e-7):
                    raise ValueError('Pool model score replay differs')
                pc=coarse.predict(pool_x[pos,:,:20]);cm=dict(zip(raw,pc));fm=dict(zip(raw,p))
                actual=Cascade(lambda u,ids,t:[cm[i] for i in ids],lambda u,ids,t:[fm[i] for i in ids],
                    Budget(**meta['config']['budget']),categories={int(i):int(state.categories[i]) for i in raw})
                trace=actual.run(uid,meta['cutoff_ms'],[Candidate(int(i),'frozen_pool',r,meta['cutoff_ms']) for r,i in enumerate(raw,1)],positives=positive[uid])
                if trace['stages'][-1]['hit_events']!=int(arrays['cascade_hits'][pos]):
                    raise ValueError('Actual cascade executor hit replay differs')
                nd=trace['stages'][-1]['user_period_ndcg_at_20']
                if nd is not None and abs(nd-float(arrays['cascade_ndcg'][pos]))>1e-12:
                    raise ValueError('Actual cascade NDCG replay differs')
                direct=raw[np.lexsort((raw,-p))[:20]]
                if sum(positive[uid].get(int(i),0) for i in direct)!=int(arrays['direct_hits'][pos]):
                    raise ValueError('Direct hit replay differs')
                stage_replays+=1
        result['evaluation_models_checked']=len(ev['runs'])
        result['actual_executor_user_model_replays']=stage_replays
        result['pool_positive_hits']=ev['pool_hits']
        result['candidate_sha256']=ev['candidate_sha256']
    write_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--models',type=Path);p.add_argument('--evaluation',type=Path)
    args=p.parse_args();audit(args.cache,args.out,args.models,args.evaluation)
