"""All-seed, identical-pool direct and cascaded evaluation; no model selection."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from src.ctr_features import CACHE, FeatureState, connection
from src.v2.cascade import observed_positive_metrics
from src.v2.temporal_route_data import write_json
from src.v2.temporal_route_train import FactorialRanker, verify_cache
from src.v2.train_rankers import sha
from src.v2.ranking_root_cause import weighted_exposure_metrics


def cascade_lists(items, probabilities, coarse_positions, categories, budget):
    items=np.asarray(items);p=np.asarray(probabilities)
    kept=np.asarray(coarse_positions,dtype=np.int64)
    order=kept[np.lexsort((items[kept],-p[kept]))[:budget['fine']]]
    fine=items[order].tolist()
    without=fine[:budget['output']]
    if budget['category_penalty']==0:
        return fine,without,without
    ids=items[order];scores=p[order].astype(np.float64);cat=np.asarray(categories)[order]
    counts=np.zeros(int(cat.max())+1,dtype=np.int32)
    available=np.ones(len(ids),dtype=bool);chosen=[]
    for _ in range(min(budget['output'],len(ids))):
        penalty=np.where(cat>0,budget['category_penalty']*np.log1p(counts[cat]),0.)
        utility=np.where(available,scores-penalty,-np.inf)
        j=np.lexsort((ids,-utility))[0]
        chosen.append(int(ids[j]));available[j]=False;counts[cat[j]]+=1
    return fine,without,chosen


def describe(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'std_across_training_seeds':float(a.std()),'values':a.tolist()}


def paired_intervals(delta_ndcg,delta_hits,positive_events,seed=2026,draws=1000):
    rng=np.random.default_rng(seed)
    keep=np.asarray(positive_events)>0
    nd=np.asarray(delta_ndcg)[keep]
    hit=np.asarray(delta_hits)[keep];positive=np.asarray(positive_events)[keep]
    nd_draw=[];hit_draw=[]
    for _ in range(draws):
        ix=rng.integers(0,len(nd),len(nd))
        nd_draw.append(float(nd[ix].mean()))
        hit_draw.append(float(hit[ix].sum()/positive[ix].sum()))
    return {'ndcg_difference':float(nd.mean()),'ndcg_95_interval':np.percentile(nd_draw,[2.5,97.5]).tolist(),
            'event_recall_difference':float(np.asarray(delta_hits).sum()/np.asarray(positive_events).sum()),
            'event_recall_95_interval':np.percentile(hit_draw,[2.5,97.5]).tolist(),
            'scope':'exploratory paired user bootstrap after averaging seeds; repeated development day, no multiple-testing adjustment'}


def evaluate(cache,models,output):
    if output.exists():
        raise FileExistsError('Choose new evaluation directory')
    meta=verify_cache(cache)
    training=json.loads((models/'metrics.json').read_text(encoding='utf-8'))
    if training['test_used'] or training['protocol']['cache_metadata_sha256']!=sha(cache/'metadata.json'):
        raise ValueError('Training/cache mismatch')
    cfg=meta['config'];torch.set_num_threads(cfg['threads'])
    users=np.load(cache/'pool_users.npy');items=np.load(cache/'pool_items.npy',mmap_mode='r')
    x=np.load(cache/'pool_X.npy',mmap_mode='r');flat=x.reshape(-1,x.shape[-1])
    coarse=np.load(cache/'coarse_positions.npy');state=FeatureState.load(CACHE/'state.npz')
    positive=defaultdict(Counter);exposure=defaultdict(Counter);observed_users=set()
    db=connection()
    try:
        for u,i,y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
            observed_users.add(u)
            exposure[u][i]+=1
            if y:
                positive[u][i]+=1
    finally:
        db.close()
    if users.tolist()!=sorted(observed_users):
        raise ValueError('User population changed')
    positive_counts=np.array([sum(positive[int(u)].values()) for u in users])
    output.mkdir(parents=True)
    write_json(output/'protocol.json',{'cache_metadata_sha256':sha(cache/'metadata.json'),
               'training_report_sha256':sha(models/'metrics.json'),'source_sha256':sha(__file__),
               'candidate_sha256':meta['candidate_sha256'],'test_used':False})
    # All users share recent-route ranks; reconstruct the independent recent Top20.
    temporal=np.load(cache/'temporal_state.npy',mmap_mode='r')
    recent_counts=temporal[1]
    ids=np.flatnonzero(recent_counts>0)
    recent_ids=ids[np.lexsort((ids,-recent_counts[ids]))[:20]].tolist()
    baseline={name:{'hits':[],'ndcg':[]} for name in ('rrf','recent')}
    pool_hits=0
    matched_index=[];matched_count=[];matched_clicks=[]
    for j,uid in enumerate(users):
        labels=positive[int(uid)]
        available=set(items[j].tolist())
        pool_hits+=sum(n for i,n in labels.items() if i in available)
        positions={int(i):r for r,i in enumerate(items[j])}
        for i,count in exposure[int(uid)].items():
            if i in positions:
                matched_index.append(j*items.shape[1]+positions[i])
                matched_count.append(count);matched_clicks.append(labels.get(i,0))
        for name,order in [('rrf',items[j,:20].tolist()),('recent',recent_ids)]:
            q=observed_positive_metrics(order,labels)
            baseline[name]['hits'].append(q['hit_events'])
            baseline[name]['ndcg'].append(q['user_period_ndcg_at_20'] if q['user_period_ndcg_at_20'] is not None else np.nan)
    if pool_hits!=34647 or int(positive_counts.sum())!=354626:
        raise ValueError('Published full-user pool coverage did not reproduce')
    baseline_summary={n:{'hit_events':int(sum(v['hits'])),'event_recall_at20':float(sum(v['hits'])/positive_counts.sum()),
                         'ndcg_at20':float(np.nanmean(v['ndcg']))} for n,v in baseline.items()}
    matched_index=np.asarray(matched_index,dtype=np.int64)
    all_arrays={};runs={};started=perf_counter()
    for name,record in training['runs'].items():
        checkpoint=models/name/'model.pt'
        if sha(checkpoint)!=record['checkpoint_sha256']:
            raise ValueError('Training model changed')
        ranker=FactorialRanker(checkpoint)
        if ranker.metadata['cache_metadata_sha256']!=sha(cache/'metadata.json'):
            raise ValueError('Model cache changed')
        p=ranker.predict(flat).reshape(items.shape)
        np.save(output/(name+'_scores.npy'),p)
        arrays={stage:{'hits':np.zeros(len(users),dtype=np.int64),'ndcg':np.full(len(users),np.nan)}
                for stage in ('direct','coarse','fine','without_rerank','cascade')}
        for j,uid in enumerate(users):
            raw=np.asarray(items[j]);scores=p[j];labels=positive[int(uid)]
            direct=raw[np.lexsort((raw,-scores))[:20]].tolist()
            cats=state.categories[raw]
            fine,without,cascade=cascade_lists(raw,scores,coarse[j],cats,cfg['budget'])
            orders={'direct':direct,'coarse':raw[coarse[j]].tolist(),'fine':fine,
                    'without_rerank':without,'cascade':cascade}
            for stage,order in orders.items():
                q=observed_positive_metrics(order,labels)
                arrays[stage]['hits'][j]=q['hit_events']
                if q['user_period_ndcg_at_20'] is not None:
                    arrays[stage]['ndcg'][j]=q['user_period_ndcg_at_20']
        warm=state.up[users]>0
        summary={}
        for stage,v in arrays.items():
            summary[stage]={}
            for group,mask in [('all',np.ones(len(users),dtype=bool)),('warm',warm),('cold',~warm)]:
                summary[stage][group]={'users':int(mask.sum()),'positive_events':int(positive_counts[mask].sum()),
                    'hit_events':int(v['hits'][mask].sum()),'event_recall':float(v['hits'][mask].sum()/positive_counts[mask].sum()),
                    'ndcg_at20':float(np.nanmean(v['ndcg'][mask]))}
        all_arrays[name]=arrays
        np.savez_compressed(output/(name+'_per_user.npz'),users=users,positive_events=positive_counts,
                            **{stage+'_'+metric:value for stage,v in arrays.items() for metric,value in v.items()})
        runs[name]={'checkpoint_sha256':record['checkpoint_sha256'],'stages':summary,'exposure':record['exposure'],
                    'pool_logged_exposures':weighted_exposure_metrics(p.ravel()[matched_index],matched_count,matched_clicks)}
        write_json(output/'progress.json',{'runs':runs,'test_used':False})
        print(f'Evaluated {name}: direct hits={summary["direct"]["all"]["hit_events"]}, NDCG={summary["direct"]["all"]["ndcg_at20"]:.6f}, cascade hits={summary["cascade"]["all"]["hit_events"]}; elapsed {perf_counter()-started:.1f}s',flush=True)
    aggregate={}
    for arm in cfg['arms']:
        names=[f'{arm}_s{s}' for s in cfg['seeds']]
        aggregate[arm]={'exposure':{metric:describe([runs[n]['exposure']['all'][metric] for n in names])
                        for metric in ('auc','gauc','logloss','brier')},
                        'pool_logged_exposures':{metric:describe([runs[n]['pool_logged_exposures'][metric] for n in names])
                        for metric in ('auc','logloss')},
                        'stages':{stage:{metric:describe([runs[n]['stages'][stage]['all'][metric] for n in names])
                        for metric in ('hit_events','event_recall','ndcg_at20')} for stage in ('direct','fine','without_rerank','cascade')}}
    paired={}
    contrasts={'T-base':{'T':1,'base':-1},'R-base':{'R':1,'base':-1},'TR-base':{'TR':1,'base':-1},
               'TR-T':{'TR':1,'T':-1},'interaction_TR-T-R+base':{'TR':1,'T':-1,'R':-1,'base':1}}
    for stage in ('direct','cascade'):
        means={arm:{m:np.mean([all_arrays[f'{arm}_s{s}'][stage][m] for s in cfg['seeds']],axis=0)
                    for m in ('hits','ndcg')} for arm in cfg['arms']}
        paired[stage]={}
        for contrast,weights in contrasts.items():
            delta={m:sum(w*means[arm][m] for arm,w in weights.items()) for m in ('hits','ndcg')}
            paired[stage][contrast]=paired_intervals(delta['ndcg'],delta['hits'],positive_counts)
    result={'users':len(users),'positive_events':int(positive_counts.sum()),'pool_hits':pool_hits,
            'candidate_sha256':meta['candidate_sha256'],'cache_metadata_sha256':sha(cache/'metadata.json'),
            'training_report_sha256':sha(models/'metrics.json'),'runs':runs,'aggregate':aggregate,
            'baselines':baseline_summary,'paired':paired,'test_used':False,'default_changed':False,
            'elapsed_seconds':perf_counter()-started,'source_sha256':sha(__file__),
            'scope':'same previously used validation day; fixed last epoch, matched capacity, no final or online acceptance'}
    write_json(output/'metrics.json',result)
    print('All-seed factorial evaluation complete',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();evaluate(args.cache,args.models,args.out)
