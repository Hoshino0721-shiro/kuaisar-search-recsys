"""Full-exposure study integrity, full row alignment and sampled SQL/serving replay."""
import argparse
import json
from math import log2
from pathlib import Path
import numpy as np
import torch
from src.ctr_features import CACHE, FeatureState, connection
from src.itemcf_index import ItemCFIndex, DAY_MS
from src.v2.full_exposure_control import verify_prepared, repeated_indices
from src.v2.temporal_route_train import verify_cache
from src.v2.temporal_route_features import recent_order, route_values
from src.v2.route_attribution import Scorer, load_labels, metric_arrays
from src.v2.train_rankers import sha
from src.v2.cascade import Cascade, Candidate, Budget
from src.v2.temporal_route_data import write_json


def audit(reference,study,output):
    if output.exists():raise FileExistsError('Preserve audit')
    meta=verify_cache(reference);report=json.loads((study/'metrics.json').read_text(encoding='utf-8'))
    if report['status']!='complete' or report['test_used']:raise ValueError('Completed validation study required')
    cfg=report['protocol']['config'];cache=study/'cache'
    if report['protocol']['prepared_cache']:cache=Path(report['protocol']['prepared_cache'])
    prepared=verify_prepared(cache,reference,cfg);torch.set_num_threads(4)
    if report['protocol']['feature_state_sha256']!=sha(CACHE/'state.npz'):raise ValueError('State changed')
    full=np.load(cache/'full_X.npy',mmap_mode='r');old=np.load(CACHE/'train_X.npy',mmap_mode='r')
    sample=np.load(reference/'sample_indices.npy');ref=np.load(reference/'train_X.npy',mmap_mode='r')
    repeat=np.load(cache/'repeat_indices.npy');expected=repeated_indices(len(sample),len(full),cfg['repeat_seed'])
    np.testing.assert_array_equal(repeat,expected)
    rx=np.load(cache/'sample_repeat_X.npy',mmap_mode='r');ry=np.load(cache/'sample_repeat_y.npy',mmap_mode='r')
    sy=np.load(reference/'train_y.npy');active=list(range(20))+list(range(28,34))
    for lo in range(0,len(full),100000):
        sl=slice(lo,lo+100000)
        np.testing.assert_array_equal(full[sl,:20],old[sl]);np.testing.assert_array_equal(rx[sl],ref[repeat[sl]])
        np.testing.assert_array_equal(ry[sl],sy[repeat[sl]])
        if np.any(full[sl,20:28]):raise ValueError('Temporal features should be inactive')
    for lo in range(0,len(sample),100000):
        sl=slice(lo,lo+100000);np.testing.assert_array_equal(full[sample[sl]][:,active],ref[sl][:,active])
    np.testing.assert_array_equal(np.load(cache/'full_mean.npy'),full.mean(axis=0,dtype=np.float64).astype(np.float32))
    np.testing.assert_array_equal(np.load(cache/'full_std.npy'),np.maximum(full.std(axis=0,dtype=np.float64),1e-6).astype(np.float32))
    tu=np.load(CACHE/'train_users.npy',mmap_mode='r');ti=np.load(CACHE/'train_items.npy',mmap_mode='r')
    state=FeatureState.load(CACHE/'state.npz');rng=np.random.default_rng(2026);db=connection();checks=[];cursor=0
    try:
        for day in meta['days']:
            cutoff=day['cutoff_ms'];count=day['source_rows']
            selected=np.sort(rng.choice(count,min(32,count),replace=False))+cursor;cursor+=count
            prior=np.zeros(len(state.ip))
            for i,n in db.execute("SELECT item_id,sum(label) FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<? GROUP BY item_id",(cutoff-DAY_MS,cutoff)):prior[i]=n
            recent=recent_order(prior,meta['config']['recent_k'])
            index=ItemCFIndex(day['index_path'],meta['manifest_sha256']) if day['index_path'] else None
            try:
                if index is not None and (index.metadata['cutoff_ms']!=cutoff or np.any(index.seed_times>=cutoff)):
                    raise ValueError('Future historical index')
                for row in selected:
                    cf=index.query(int(tu[row]),meta['config']['itemcf_k']) if index else {'items':[],'scores':[]}
                    actual=route_values(np.array([ti[row]]),recent,cf['items'],cf['scores'])[0]
                    np.testing.assert_array_equal(actual,full[row,28:])
            finally:
                if index:index.close()
            checks.append({'cutoff_ms':cutoff,'rows_checked':len(selected),'matched':True})
    finally:db.close()
    items=np.load(reference/'pool_items.npy',mmap_mode='r');users=np.load(reference/'pool_users.npy')
    pool=np.load(reference/'pool_X.npy',mmap_mode='r');coarse=np.load(reference/'coarse_positions.npy')
    labels,positive,_,_=load_labels(users,items);replays=0
    for name,record in report['runs'].items():
        cp=study/'models'/name/'model.pt';training=record['training']
        if sha(cp)!=training['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        if len(training['history'])!=cfg['epochs']:raise ValueError('Incomplete training')
        scorer=Scorer(cp);scores=np.load(study/(name+'_scores.npy'),mmap_mode='r')
        saved=np.load(study/(name+'_per_user.npz'))
        hits,ndcg=metric_arrays(items,scores,labels)
        np.testing.assert_array_equal(hits,saved['direct_hits']);np.testing.assert_allclose(ndcg,saved['direct_ndcg'],equal_nan=True)
        if int(hits.sum())!=record['direct']['hit_events'] or int(saved['cascade_hits'].sum())!=record['cascade']['hit_events']:
            raise ValueError('Per-user sum mismatch')
        for j in rng.choice(len(items),min(8,len(items)),replace=False):
            prediction=scorer.predict(pool[j]);np.testing.assert_allclose(prediction,scores[j],atol=1e-7,rtol=1e-6)
            ids=items[j].tolist();cs=np.full(len(ids),-100000.,dtype=float);cs[coarse[j]]=-np.arange(len(coarse[j]))
            ranked=sorted(range(len(ids)),key=lambda k:(-float(scores[j,k]),ids[k]))[:20]
            if sum(labels[j].get(ids[k],0) for k in ranked)!=saved['direct_hits'][j]:raise ValueError('Independent direct ordering differs')
            ideal=sum(1/log2(k+2) for k in range(min(20,len(labels[j]))))
            dcg=sum(1/log2(k+2) for k,pos in enumerate(ranked) if labels[j].get(ids[pos],0)>0)
            np.testing.assert_allclose(dcg/ideal if ideal else np.nan,saved['direct_ndcg'][j],equal_nan=True)
            # Isolate executor equivalence from batch-shape floating-point ties;
            # checkpoint predictions were independently checked just above.
            cm=dict(zip(ids,cs));fm=dict(zip(ids,scores[j]));categories=dict(zip(ids,state.categories[items[j]].tolist()))
            cascade=Cascade(lambda u,ii,t:[cm[i] for i in ii],lambda u,ii,t:[fm[i] for i in ii],Budget(**meta['config']['budget']),categories=categories)
            result=cascade.run(int(users[j]),meta['cutoff_ms'],[Candidate(i,'frozen',k,meta['cutoff_ms']) for k,i in enumerate(ids,1)],positives=labels[j])
            if result['stages'][-1]['hit_events']!=saved['cascade_hits'][j]:raise ValueError('Executor replay mismatch')
            np.testing.assert_allclose(result['stages'][-1]['user_period_ndcg_at_20'] if positive[j] else np.nan,saved['cascade_ndcg'][j],equal_nan=True)
            replays+=1
        scores._mmap.close();saved.close()
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            runs=[report['runs'][f'{f}_{arm}_s{seed}']['training'] for f in cfg['families']]
            for key in ('initial_parameters_sha256','training_order_sha256'):
                if len({r['metadata'][key] for r in runs})!=1:raise ValueError('Budget/init mismatch')
            if len({r['optimizer_steps'] for r in runs})!=1:raise ValueError('Update count mismatch')
    freeze=json.loads(Path('experiments/final_test/freeze.json').read_text())
    if any(sha(p)!=h for p,h in freeze['asset_hashes'].items()):raise ValueError('Frozen v1 assets changed')
    write_json(output,{'passed':True,'test_used':False,'models':len(report['runs']),'full_rows':len(full),
        'all_original_columns_and_repeat_rows_match':True,'sample_overlap_route_columns_exact':True,
        'full_normalization_recomputed':True,'sampled_sql_prefix_checks':checks,'actual_cascade_replays':replays,
        'all_direct_per_user_metrics_recomputed':True,'cascade_sums_verified':True,
        'candidate_sha256':meta['candidate_sha256'],'frozen_v1_unchanged':True,'source_sha256':sha(__file__)})
    for a in (full,old,ref,rx,ry,tu,ti,items,pool):a._mmap.close()
    print('Full exposure audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.reference,a.study,a.out)
