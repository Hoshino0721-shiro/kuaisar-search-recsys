"""Reload every new model and independently replay sampled predictions/lists."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from src.ctr_features import connection, CACHE
from src.v2.route_attribution import Scorer
from src.v2.temporal_route_train import FactorialRanker
from src.v2.temporal_route_data import write_json
from src.v2.train_rankers import sha


def independent_metrics(items, scores, positives):
    order = sorted(zip(items.tolist(), scores.tolist()),key=lambda p:(-p[1],p[0]))[:20]
    hits = sum(positives[i] for i,p in order)
    relevant = set(positives)
    dcg = sum(1/np.log2(r+2) for r,(i,p) in enumerate(order) if i in relevant)
    ideal = sum(1/np.log2(r+2) for r in range(min(20,len(relevant))))
    return hits,dcg/ideal if ideal else np.nan


def audit(root, cache, previous_models, previous_evaluation):
    torch.set_num_threads(4)
    result = json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    protocol = result['protocol']
    if result['status']!='complete' or result['test_used'] or result['default_changed']:
        raise ValueError('Not a completed isolated run')
    for filename,key in [('configs/route_attribution.json','config_sha256'),
                         ('src/v2/route_attribution.py','source_sha256')]:
        if sha(filename)!=protocol[key]:
            raise ValueError('Source/protocol mismatch')
    if sha(cache/'metadata.json')!=protocol['cache_metadata_sha256']:
        raise ValueError('Cache metadata changed')
    db = connection()
    labels = defaultdict(Counter)
    try:
        for u,i in db.execute("SELECT user_id,item_id FROM clean_events WHERE scene='rec' AND split='valid' AND label=1"):
            labels[u][i]+=1
    finally:
        db.close()
    users=np.load(cache/'pool_users.npy');items=np.load(cache/'pool_items.npy',mmap_mode='r')
    x=np.load(cache/'pool_X.npy',mmap_mode='r')
    rng=np.random.default_rng(20261022)
    selected=np.sort(rng.choice(len(users),32,replace=False))
    model_checks=[];max_prediction_error=0.;replayed_lists=0
    for name,record in result['ablation']['runs'].items():
        reused=record.get('reused',False)
        scorer=FactorialRanker(previous_models/name/'model.pt') if reused else Scorer(root/'models'/name/'model.pt')
        scorepath=(previous_evaluation if reused else root)/(name+'_scores.npy')
        p=np.load(scorepath,mmap_mode='r')
        expected=scorer.predict(x[selected].reshape(-1,34)).reshape(len(selected),2000)
        err=float(np.abs(expected-p[selected]).max());max_prediction_error=max(max_prediction_error,err)
        np.testing.assert_allclose(expected,p[selected],rtol=1e-6,atol=1e-7)
        arrays=np.load((previous_evaluation if reused else root)/(name+'_per_user.npz'))
        hkey,dkey=('direct_hits','direct_ndcg') if reused else ('hits','ndcg')
        if int(arrays[hkey].sum())!=record['direct']['hit_events']:
            raise ValueError('Full user hit sum differs')
        if not np.isclose(np.nanmean(arrays[dkey]),record['direct']['ndcg_at20'],rtol=0,atol=1e-14):
            raise ValueError('Full user NDCG sum differs')
        for j in selected:
            h,d=independent_metrics(items[j],p[j],labels[int(users[j])]);replayed_lists+=1
            if h!=arrays[hkey][j] or not np.isclose(d,arrays[dkey][j],equal_nan=True,rtol=0,atol=1e-14):
                raise ValueError('Independent list replay differs')
        if not reused and sha(root/'models'/name/'model.pt')!=record['training']['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        model_checks.append(name)
    diagnosis=json.loads((root/'diagnosis.json').read_text(encoding='utf-8'))
    clicks=np.load(root/'pool_click_counts.npy');counts=np.load(root/'pool_exposure_counts.npy')
    if np.any(clicks>counts) or clicks[:,:20].sum()!=1745:
        raise ValueError('Exposure/click counts mismatch')
    for name,record in diagnosis['runs'].items():
        if sum(record['rrf_positive_event_destination'].values())!=1745:
            raise ValueError('Lost hit accounting differs')
        if record['rrf_hits_retained']+record['new_hits_outside_rrf_top20']!=record['direct_hits']:
            raise ValueError('Direct hit accounting differs')
    # Verify normalization is refitted on the exact strict temporal prefix.
    tx=np.load(cache/'train_X.npy',mmap_mode='r');days=np.load(cache/'train_days.npy')
    rolling_checks=[]
    for cutoff_s,window in result['rolling'].items():
        cutoff=int(cutoff_s);dest=root/cutoff_s
        normalization=np.load(dest/'normalization.npz')
        xt=np.asarray(tx[days<cutoff])
        np.testing.assert_array_equal(normalization['mean'],xt.mean(axis=0,dtype=np.float64).astype(np.float32))
        np.testing.assert_array_equal(normalization['std'],np.maximum(xt.std(axis=0,dtype=np.float64),1e-6).astype(np.float32))
        if window['numeric_feature_replay_max_error']!=0 or window['train_max_day_ms']>=cutoff:
            raise ValueError('Temporal boundary mismatch')
        if sha(dest/'pool_items.npy')!=window['candidate_items_sha256']:
            raise ValueError('Rolling candidate pool changed')
        rusers=np.load(dest/'users.npy');ri=np.load(dest/'pool_items.npy',mmap_mode='r')
        chosen=np.sort(rng.choice(len(rusers),16,replace=False))
        db=connection();positive=defaultdict(Counter)
        try:
            for u,i in db.execute("SELECT user_id,item_id FROM clean_events WHERE scene='rec' AND split='train' AND label=1 AND timestamp_ms>=? AND timestamp_ms<?",(cutoff,cutoff+86400000)):
                positive[u][i]+=1
        finally:
            db.close()
        from src.v2.route_attribution import prefix_state
        from src.v2.temporal_route_features import recent_order,route_values
        from src.itemcf_index import ItemCFIndex
        state,last_clicks=prefix_state(cutoff)
        recent=recent_order(last_clicks,2000)
        meta=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
        day=next(d for d in meta['days'] if d['cutoff_ms']==cutoff)
        index=ItemCFIndex(day['index_path'],meta['manifest_sha256'])
        blocks=[]
        try:
            for j in chosen:
                uid=int(rusers[j]);cf=index.query(uid,1000)
                xx=np.zeros((2000,34),dtype=np.float32)
                xx[:,:20]=state.features(np.full(2000,uid),ri[j],np.full(2000,cutoff))
                xx[:,28:]=route_values(ri[j],recent,cf['items'],cf['scores'])
                blocks.append(xx)
        finally:
            index.close()
        xx=np.concatenate(blocks)
        for name,record in window['runs'].items():
            path=dest/'models'/name/'model.pt'
            if sha(path)!=record['training']['checkpoint_sha256']:
                raise ValueError('Rolling checkpoint changed')
            scorer=Scorer(path)
            p=np.load(dest/(name+'_scores.npy'),mmap_mode='r')
            expected=scorer.predict(xx).reshape(len(chosen),2000)
            err=float(np.abs(expected-p[chosen]).max());max_prediction_error=max(max_prediction_error,err)
            np.testing.assert_allclose(expected,p[chosen],rtol=1e-6,atol=1e-7)
            arrays=np.load(dest/(name+'_per_user.npz'))
            if int(arrays['hits'].sum())!=record['direct']['hit_events']:
                raise ValueError('Rolling user hit sum differs')
            for j in chosen:
                h,d=independent_metrics(ri[j],p[j],positive[int(rusers[j])]);replayed_lists+=1
                if h!=arrays['hits'][j] or not np.isclose(d,arrays['ndcg'][j],equal_nan=True,rtol=0,atol=1e-14):
                    raise ValueError('Rolling independent list replay differs')
            rolling_checks.append(cutoff_s+'/'+name)
    report={'passed':True,'new_models':result['new_model_runs'],'validation_models_reloaded':model_checks,
            'rolling_models_reloaded':rolling_checks,'sampled_lists_independently_replayed':replayed_lists,
            'max_prediction_abs_error':max_prediction_error,'strict_prefix_normalization_replayed':True,
            'rrf_hit_accounting_passed':True,'test_used':False,'default_changed':False,
            'source_sha256':sha(__file__),'metrics_sha256':sha(root/'metrics.json'),
            'scope':'sampled prediction/list replay plus full per-user sums; not exhaustive graph proof'}
    write_json(root/'audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('artifacts/temporal_route_20261022/cache'))
    p.add_argument('--previous-models',type=Path,default=Path('artifacts/temporal_route_20261022/models'))
    p.add_argument('--previous-evaluation',type=Path,default=Path('artifacts/temporal_route_20261022/evaluation'))
    a=p.parse_args();audit(a.root,a.cache,a.previous_models,a.previous_evaluation)
