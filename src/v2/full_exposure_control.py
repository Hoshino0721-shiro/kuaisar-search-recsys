"""Full-data and step-matched repeated-sample DCN control, validation only.

Reuse immutable daily-prefix ItemCF indexes. Temporal columns stay inactive.
All labels are actual recommendation exposures; no unlabeled candidate negatives.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from src.ctr_features import CACHE, DATA, FeatureState
from src.itemcf_index import IndexConfig, ItemCFIndex
from src.v2.temporal_route_train import verify_cache
from src.v2.temporal_route_data import fill_route_groups, write_json
from src.v2.temporal_route_features import WIDTH, recent_order
from src.v2.route_attribution import fit, metric_arrays, summary, load_labels
from src.v2.temporal_route_evaluate import cascade_lists, paired_intervals
from src.v2.train_rankers import sha


def repeated_indices(sample_size, output_size, seed):
    if sample_size<1 or output_size<sample_size:
        raise ValueError('Repeat control must retain every original sampled row')
    rng=np.random.default_rng(seed)
    permutation=rng.permutation(sample_size)
    result=np.tile(permutation,(output_size+sample_size-1)//sample_size)[:output_size].copy()
    rng.shuffle(result)
    return result


def coverage(reference, destination):
    ti=np.load(CACHE/'train_items.npy',mmap_mode='r')
    tu=np.load(CACHE/'train_users.npy',mmap_mode='r')
    indices=np.load(reference/'sample_indices.npy')
    pool=np.load(reference/'pool_items.npy',mmap_mode='r'); users=np.load(reference/'pool_users.npy')
    a=np.bincount(ti); b=np.bincount(ti[indices],minlength=len(a))
    ua=np.bincount(tu);ub=np.bincount(tu[indices],minlength=len(ua))
    unique=np.unique(pool)
    seen=b[pool]>0
    result={'full_rows':len(ti),'sample_rows':len(indices),
            'full_unique_items':int((a>0).sum()),'sample_unique_items':int((b>0).sum()),
            'candidate_unique_items':len(unique),'candidate_unique_items_absent_sample':int((b[unique]==0).sum()),
            'candidate_occurrences_absent_sample_fraction':float((~seen).mean()),
            'rrf_top20_occurrences_absent_sample_fraction':float((b[pool[:,:20]]==0).mean()),
            'validation_users_absent_full':int((ua[users]==0).sum()),
            'validation_users_seen_full_but_absent_sample':int(((ua[users]>0)&(ub[users]==0)).sum()),
            'caveat':'numeric DCN has no item/user ID embeddings; cumulative numeric state already uses all causal preceding logs'}
    np.save(destination/'pool_item_seen_sample.npy',seen)
    write_json(destination/'coverage.json',result)
    return result


def prepare(reference, destination, cfg, meta):
    destination.mkdir()
    tx=np.load(CACHE/'train_X.npy',mmap_mode='r');ty=np.load(CACHE/'train_y.npy',mmap_mode='r')
    tu=np.load(CACHE/'train_users.npy',mmap_mode='r');ti=np.load(CACHE/'train_items.npy',mmap_mode='r')
    x=np.lib.format.open_memmap(destination/'full_X.npy',mode='w+',dtype=np.float32,shape=(len(tx),WIDTH))
    x[:,:20]=tx; x[:,20:]=0
    old=np.load(reference/'train_X.npy',mmap_mode='r');sample=np.load(reference/'sample_indices.npy')
    prior=np.zeros(len(FeatureState.load(CACHE/'state.npz').ip),dtype=np.float64)
    cumulative=prior.copy();cursor=0;days=[]
    index_cfg=IndexConfig(**json.loads(Path(meta['config']['itemcf_config']).read_text(encoding='utf-8')))
    for day in meta['days']:
        lo=cursor;hi=lo+day['source_rows']
        recent=recent_order(prior,meta['config']['recent_k'])
        index=None
        if day['index_path']:
            index=ItemCFIndex(day['index_path'],meta['manifest_sha256'])
            if index.metadata['cutoff_ms']!=day['cutoff_ms'] or index.metadata['config']!=asdict(index_cfg):
                raise ValueError('Daily prefix configuration changed')
            if index.metadata['version']!=day['index_version'] or np.any(index.seed_times>=day['cutoff_ms']):
                raise ValueError('Wrong historical graph or future seeds')
            np.testing.assert_array_equal(index.items,np.flatnonzero(cumulative>0))
        elif cumulative.sum():
            raise ValueError('Missing nonempty prefix graph')
        try:
            fill_route_groups(tu[lo:hi],ti[lo:hi],x[lo:hi,28:],recent,index,meta['config']['itemcf_k'])
        finally:
            if index is not None:
                index.close()
        prior=np.bincount(ti[lo:hi],weights=ty[lo:hi],minlength=len(prior))
        cumulative+=prior
        cursor=hi;x.flush()
        days.append({'cutoff_ms':day['cutoff_ms'],'rows':hi-lo,'index_version':day['index_version']})
        print(f'Full-data historical routes {len(days)}/{len(meta["days"])}: {cursor}/{len(tx)}',flush=True)
    if cursor!=len(tx):
        raise ValueError('Source row count mismatch')
    active=list(range(20))+list(range(28,34))
    # Independent sample was prepared in a previous experiment; exact overlap is required.
    max_error=0.
    for lo in range(0,len(sample),100000):
        actual=x[sample[lo:lo+100000]][:,active]
        expected=old[lo:lo+100000][:,active]
        max_error=max(max_error,float(np.abs(actual-expected).max()))
    if max_error!=0:
        raise ValueError('Full/sample feature definitions differ')
    mean=x.mean(axis=0,dtype=np.float64).astype(np.float32)
    std=np.maximum(x.std(axis=0,dtype=np.float64),1e-6).astype(np.float32)
    np.save(destination/'full_mean.npy',mean);np.save(destination/'full_std.npy',std)
    repeat=repeated_indices(len(sample),len(tx),cfg['repeat_seed'])
    np.save(destination/'repeat_indices.npy',repeat)
    repeated=np.lib.format.open_memmap(destination/'sample_repeat_X.npy',mode='w+',dtype=np.float32,shape=x.shape)
    repeated_y=np.lib.format.open_memmap(destination/'sample_repeat_y.npy',mode='w+',dtype=np.int8,shape=(len(tx),))
    sample_y=np.load(reference/'train_y.npy')
    for lo in range(0,len(tx),100000):
        ix=repeat[lo:lo+100000]
        repeated[lo:lo+len(ix)]=old[ix]
        repeated_y[lo:lo+len(ix)]=sample_y[ix]
    repeated.flush();repeated_y.flush();x.flush()
    report={'rows':len(tx),'unique_sample_rows':len(sample),'repeat_multiplicity_min':int(np.bincount(repeat).min()),
            'repeat_multiplicity_max':int(np.bincount(repeat).max()),'days':days,
            'active_columns_sample_replay_max_error':max_error,'inactive_T_columns':True,
            'files':{p.name:sha(p) for p in destination.glob('*.npy')},'test_used':False,'status':'ready'}
    write_json(destination/'metadata.json',report)
    return report


def evaluate_one(name,scorer,train,reference,output,labels,positive,old_protocol):
    items=np.load(reference/'pool_items.npy',mmap_mode='r');pool=np.load(reference/'pool_X.npy',mmap_mode='r')
    coarse=np.load(reference/'coarse_positions.npy');state=FeatureState.load(CACHE/'state.npz')
    p=scorer.predict(pool.reshape(-1,WIDTH)).reshape(items.shape)
    np.save(output/(name+'_scores.npy'),p)
    h,d=metric_arrays(items,p,labels);ch=np.zeros(len(items),dtype=np.int64);cd=np.full(len(items),np.nan)
    for j in range(len(items)):
        _,_,order=cascade_lists(items[j],p[j],coarse[j],state.categories[items[j]],old_protocol['budget'])
        q=metric_arrays(np.array([order]),np.broadcast_to(-np.arange(len(order)),(1,len(order))),[labels[j]])
        ch[j],cd[j]=q[0][0],q[1][0]
    groups={}
    seen=np.load(output/'pool_item_seen_sample.npy',mmap_mode='r')
    for group,wanted in [('seen_sample',True),('unseen_sample',False)]:
        hits=0;selected_count=0
        for j in range(len(items)):
            order=np.lexsort((items[j],-p[j]))[:20]
            ix=order[seen[j,order]==wanted]
            selected_count+=len(ix)
            hits+=sum(labels[j].get(int(items[j,k]),0) for k in ix)
        groups[group]={'selected_pairs':selected_count,'hit_events':hits}
    np.savez_compressed(output/(name+'_per_user.npz'),direct_hits=h,direct_ndcg=d,cascade_hits=ch,cascade_ndcg=cd)
    record={'training':train,'direct':summary(h,d,int(positive.sum())),
            'cascade':summary(ch,cd,int(positive.sum())),'selection_by_sample_item_coverage':groups}
    print(f'Evaluated {name}: direct {h.sum()}, cascade {ch.sum()}',flush=True)
    return record,(h,d,ch,cd)


def run(reference,previous_models,previous_evaluation,output,config):
    if output.exists():
        raise FileExistsError('Choose a new experiment output')
    started=perf_counter();meta=verify_cache(reference)
    cfg=json.loads(config.read_text(encoding='utf-8'));torch.set_num_threads(cfg['threads'])
    output.mkdir(parents=True)
    protocol={'config':cfg,'config_sha256':sha(config),'source_sha256':sha(__file__),
              'reference_metadata_sha256':sha(reference/'metadata.json'),
              'candidate_sha256':meta['candidate_sha256'],'feature_state_sha256':sha(CACHE/'state.npz'),
              'trainer_source_sha256':sha(Path(__file__).with_name('route_attribution.py')),
              'test_used':False,'default_changed':False}
    write_json(output/'protocol.json',protocol)
    cov=coverage(reference,output)
    prepared=prepare(reference,output/'cache',cfg,meta)
    users=np.load(reference/'pool_users.npy');items=np.load(reference/'pool_items.npy',mmap_mode='r')
    labels,positive,_,_=load_labels(users,items)
    xv=np.load(reference/'valid_X.npy',mmap_mode='r');yv=np.load(CACHE/'valid_y.npy')
    records={};arrays={}
    for family in cfg['families']:
        if family=='full':
            x=np.load(output/'cache/full_X.npy',mmap_mode='r');y=np.load(CACHE/'train_y.npy',mmap_mode='r')
            mean=np.load(output/'cache/full_mean.npy');std=np.load(output/'cache/full_std.npy')
        else:
            x=np.load(output/'cache/sample_repeat_X.npy',mmap_mode='r');y=np.load(output/'cache/sample_repeat_y.npy',mmap_mode='r')
            mean=np.load(reference/'mean.npy');std=np.load(reference/'std.npy')
        for arm,columns in cfg['arms'].items():
            for seed in cfg['seeds']:
                name=f'{family}_{arm}_s{seed}'
                scorer,train=fit(x,y,xv,yv,mean,std,columns,seed,cfg,output/'models'/name,
                    {'family':family,'original_unique_rows':cov['full_rows'] if family=='full' else cov['sample_rows'],
                     'preparation_sha256':sha(output/'cache/metadata.json')})
                train['optimizer_steps']=cfg['epochs']*((len(x)+cfg['batch_size']-1)//cfg['batch_size'])
                record,a=evaluate_one(name,scorer,train,reference,output,labels,positive,meta['config'])
                records[name]=record;arrays[name]=a
                write_json(output/'progress.json',records)
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            names=[f'{f}_{arm}_s{seed}' for f in cfg['families']]
            for key in ('initial_parameters_sha256','training_order_sha256'):
                if len({records[n]['training']['metadata'][key] for n in names})!=1:
                    raise ValueError('Full/repeat initial or row-position order differs')
            name=f'sample5_{arm}_s{seed}';old=f'{arm}_s{seed}'
            a=np.load(previous_evaluation/(old+'_per_user.npz'))
            arrays[name]=tuple(a[k] for k in ('direct_hits','direct_ndcg','cascade_hits','cascade_ndcg'))
            tr=json.loads((previous_models/old/'metrics.json').read_text(encoding='utf-8'))
            records[name]={'reused':True,'direct':summary(arrays[name][0],arrays[name][1],int(positive.sum())),
                           'cascade':summary(arrays[name][2],arrays[name][3],int(positive.sum())),
                           'training':{'exposure':tr['exposure']['all'],'metadata':tr['metadata']}}
    means={};aggregate={}
    for family in cfg['families']+['sample5']:
        for arm in cfg['arms']:
            key=family+'_'+arm;names=[key+f'_s{s}' for s in cfg['seeds']]
            a=[np.mean([arrays[n][j] for n in names],axis=0) for j in range(4)];means[key]=a
            aggregate[key]={'direct':summary(a[0],a[1],int(positive.sum())),
                            'cascade':summary(a[2],a[3],int(positive.sum())),
                            'exposure_auc':float(np.mean([records[n]['training']['exposure']['auc'] for n in names])),
                            'exposure_logloss':float(np.mean([records[n]['training']['exposure']['logloss'] for n in names]))}
            aggregate[key]['direct']['hit_events']=float(a[0].sum());aggregate[key]['cascade']['hit_events']=float(a[2].sum())
    paired={}
    for arm in cfg['arms']:
        for family in ['sample5','sample_repeat']:
            a=means['full_'+arm];b=means[family+'_'+arm]
            paired['full_'+arm+'-'+family+'_'+arm]=paired_intervals(a[1]-b[1],a[0]-b[0],positive)
    result={'protocol':protocol,'preparation':prepared,'coverage':cov,'runs':records,
            'aggregate':aggregate,'paired_direct':paired,'new_model_runs':12,
            'baselines':json.loads((previous_evaluation/'metrics.json').read_text())['baselines'],
            'test_used':False,'default_changed':False,'status':'complete','elapsed_seconds':perf_counter()-started}
    if sha(__file__)!=protocol['source_sha256']:
        raise ValueError('Source changed during run')
    write_json(output/'metrics.json',result)
    print('Full exposure controlled experiment complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,default=Path('artifacts/temporal_route_20261022/cache'))
    p.add_argument('--previous-models',type=Path,default=Path('artifacts/temporal_route_20261022/models'))
    p.add_argument('--previous-evaluation',type=Path,default=Path('artifacts/temporal_route_20261022/evaluation'))
    p.add_argument('--config',type=Path,default=Path('configs/full_exposure_control.json'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.reference,a.previous_models,a.previous_evaluation,a.out,a.config)
