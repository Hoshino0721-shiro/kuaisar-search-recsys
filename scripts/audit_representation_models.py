"""Independent safe-embedding model, exposure, full-list and serving checks."""
import argparse
import json
from math import log2
from pathlib import Path

import numpy as np
import torch

from src.ctr_features import CACHE,FeatureState
from src.ranking import metrics as independent_metrics
from src.v2.cascade import Budget,Candidate,Cascade
from src.v2.ranking_convergence import frozen_assets,write_json
from src.v2.representation_train import load_data
from src.v2.representation_model import RepresentationScorer
from src.v2.representation_evaluate import InterestScorer
from src.v2.route_attribution import load_labels,metric_arrays
from src.v2.train_rankers import sha


def audit(cache,target,interest,target_eval,interest_eval,reference,output):
    if output.exists():raise FileExistsError('Preserve audit')
    torch.set_num_threads(4); data=load_data(cache)
    reports=[json.loads((p/'metrics.json').read_text()) for p in (target,interest,target_eval,interest_eval)]
    for r in reports:
        if r['status']!='complete' or r['test_used']:raise ValueError('Complete non-test reports required')
    for directory,report,source,key in [(target,reports[0],'representation_train.py','model_source_sha256'),
                                      (interest,reports[1],'representation_interest.py','model_sha256')]:
        if sha(Path('src/v2')/source)!=report['protocol']['source_sha256']:raise ValueError('Trainer changed')
        if sha('src/v2/representation_model.py')!=report['protocol'][key]:raise ValueError('Embedding model changed')
    for ev,models in [(reports[2],target),(reports[3],interest)]:
        if ev['protocol']['models_sha256']!=sha(models/'metrics.json'):raise ValueError('Evaluation provenance mismatch')
    xv=np.load(reference/'valid_X.npy',mmap_mode='r');yv=np.load(CACHE/'valid_y.npy')
    uv=np.load(CACHE/'valid_users.npy');iv=np.load(CACHE/'valid_items.npy')
    users=np.load(reference/'pool_users.npy');items=np.load(reference/'pool_items.npy',mmap_mode='r')
    pool=np.load(reference/'pool_X.npy',mmap_mode='r');coarse=np.load(reference/'coarse_positions.npy')
    meta=json.loads((reference/'metadata.json').read_text());labels,positive,_,_=load_labels(users,items)
    categories=FeatureState.load(CACHE/'state.npz').categories;rng=np.random.default_rng(2026)
    times=np.load(CACHE/'valid_times.npy');order=np.lexsort((times,uv))
    cuts=np.r_[0,np.flatnonzero((np.diff(uv[order])!=0)|(np.diff(times[order])!=0))+1,len(yv)]
    pos=np.add.reduceat(yv[order].astype(np.int64),cuts[:-1])
    mixed=np.flatnonzero((pos>0)&(pos<np.diff(cuts)))
    checks=[];coverage={}
    for mode,models,evaluation,report in [('target',target,target_eval,reports[2]),('interest',interest,interest_eval,reports[3])]:
        for name,record in report['runs'].items():
            train=record['training'];cp=models/name/'model.pt'
            if sha(cp)!=train['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            saved=torch.load(models/name/'last.pt',map_location='cpu',weights_only=True)
            history=train['history'];epoch0=history[0]['epoch']==0
            selected=min(history,key=lambda r:r['valid']['logloss'])['epoch']
            if selected!=train['selected_epoch'] or saved['history']!=history or not saved['finished']:
                raise ValueError('Selection/history/complete flag mismatch')
            if train['early_stopping_reason']=='validation_plateau':
                cfg=saved['protocol']['config']
                if saved['bad']<cfg['early_patience'] or saved['reductions']<2 or train['epochs_completed']<cfg['min_epochs']:
                    raise ValueError('Premature early stop')
            elif train['epochs_completed']!=saved['protocol']['config']['max_epochs']:
                raise ValueError('Wrong budget stop')
            expected_steps=train['epochs_completed']*((len(data.train_context)+saved['protocol']['config']['batch_size']-1)//saved['protocol']['config']['batch_size'])
            optimizers=saved['optimizers'] if mode=='target' else [saved['optimizer']]
            actual_steps={int(state['step'].item()) if torch.is_tensor(state['step']) else int(state['step'])
                          for optimizer in optimizers for state in optimizer['state'].values()}
            if actual_steps!={expected_steps}:raise ValueError('Actual optimizer update counts differ')
            if mode=='target':scorer=RepresentationScorer(cp,data);model=scorer.model
            else:
                seed=train['metadata']['seed'];directory=interest/'frozen'/f's{seed}'
                freeze=json.loads((directory/'metadata.json').read_text())
                if sha(directory/'items.npy')!=freeze['items_sha256']:raise ValueError('Frozen vectors changed')
                if freeze['context']['checkpoint_sha256']!=sha(target/f'content_s{seed}'/'model.pt'):
                    raise ValueError('Frozen encoder mismatch')
                scorer=InterestScorer(cp,target/f'content_s{seed}'/'model.pt',torch.from_numpy(np.load(directory/'items.npy')),data)
                model=scorer.model
            for parameter,value in model.state_dict().items():
                if not torch.equal(value,saved['best_state'][parameter]):raise ValueError('Best state mismatch')
            if mode=='target':
                flags={k:v.clone() for k,v in model.named_buffers()}
                if train['metadata']['kind']!='representation_dcn_v2':raise ValueError('Unsafe provisional model')
                for key,table in model.tables.items():
                    torch.testing.assert_close(table.weight[0],torch.zeros_like(table.weight[0]))
                    unseen=torch.nonzero(~getattr(model,'seen_'+key)).flatten()[:32]
                    if len(unseen):
                        torch.testing.assert_close(model.embed(key,unseen),table(torch.ones_like(unseen)))
                if name.startswith('content_'):
                    coverage[name]={}
                    for kind in ('rec','search'):
                        raw=getattr(data,'query_'+kind).long();real=raw.ne(0)
                        fields=data.item_fields[raw];words=data.caption[raw]
                        known_id=(fields[:,:,0]>=2)&model.seen_item[fields[:,:,0]]
                        known_word=((words>=2)&model.seen_word[words]).any(-1)
                        coverage[name][kind]={'nonpad_items':int(real.sum()),
                            'unsupported_id_fraction':float((~known_id[real]).float().mean()),
                            'with_supported_word_fraction':float(known_word[real].float().mean())}
            p=scorer.predict(xv,uv,iv)
            np.testing.assert_array_equal(p,np.load(evaluation/(name+'_valid.npy')))
            actual=independent_metrics(yv,p,uv)
            for metric in ('auc','gauc','logloss','brier'):
                np.testing.assert_allclose(actual[metric],train['selected_validation'][metric],atol=1e-10,rtol=0)
            ndcg=[]
            for g in mixed:
                ix=order[cuts[g]:cuts[g+1]]
                ranked=sorted(ix.tolist(),key=lambda j:(-float(p[j]),int(iv[j]),j))[:20]
                ideal=sum(1/log2(k+2) for k in range(min(20,int(pos[g]))))
                ndcg.append(sum(int(yv[j])/log2(k+2) for k,j in enumerate(ranked))/ideal)
            np.testing.assert_allclose(np.mean(ndcg),train['selected_validation']['same_timestamp']['ndcg_at20'],atol=1e-12,rtol=0)
            if mode=='target':
                for key,value in model.named_buffers():torch.testing.assert_close(value,flags[key])
            scores=np.load(evaluation/(name+'_scores.npy'),mmap_mode='r')
            arrays=np.load(evaluation/(name+'_per_user.npz'))
            h,d=metric_arrays(items,scores,labels)
            np.testing.assert_array_equal(h,arrays['direct_hits']);np.testing.assert_allclose(d,arrays['direct_ndcg'],equal_nan=True)
            if int(h.sum())!=record['direct']['hit_events'] or int(arrays['cascade_hits'].sum())!=record['cascade']['hit_events']:
                raise ValueError('Full-user summary mismatch')
            for j in rng.choice(len(users),8,replace=False):
                pred=scorer.predict(pool[j],np.full(items.shape[1],users[j]),items[j])
                np.testing.assert_allclose(pred,scores[j],atol=2e-7,rtol=1e-6)
                ids=items[j].tolist();cs=np.full(len(ids),-100000.);cs[coarse[j]]=-np.arange(len(coarse[j]))
                cm,fm=dict(zip(ids,cs)),dict(zip(ids,scores[j]))
                executor=Cascade(lambda u,ii,t:[cm[i] for i in ii],lambda u,ii,t:[fm[i] for i in ii],
                    Budget(**meta['config']['budget']),categories=dict(zip(ids,categories[items[j]].tolist())))
                served=executor.run(int(users[j]),meta['cutoff_ms'],
                    [Candidate(i,'frozen',k,meta['cutoff_ms']) for k,i in enumerate(ids,1)],positives=labels[j])
                if served['stages'][-1]['hit_events']!=arrays['cascade_hits'][j]:raise ValueError('Executor mismatch')
            checks.append({'name':name,'selected_epoch':selected,'exposure_metrics_independently_verified':True,
                           'timestamp_groups_checked':len(mixed),'actual_optimizer_steps':expected_steps,
                           'full_user_direct_metrics':True,'executor_replays':8})
            scores._mmap.close();arrays.close()
    write_json(output,{'status':'passed','models':checks,'effective_history_coverage':coverage,
               'frozen_assets_verified':frozen_assets(),'test_used':False,'default_changed':False,'source_sha256':sha(__file__)})
    for a in (xv,items,pool):a._mmap.close()
    print('Safe representation model audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();base=Path('artifacts/representation_20261022')
    p.add_argument('--cache',type=Path,default=base/'cache');p.add_argument('--target',type=Path,default=base/'models_safe')
    p.add_argument('--interest',type=Path,default=base/'interest_safe');p.add_argument('--target-eval',type=Path,default=base/'evaluation_target_safe')
    p.add_argument('--interest-eval',type=Path,default=base/'evaluation_interest_safe')
    p.add_argument('--reference',type=Path,default=Path('artifacts/full_exposure_local_20261022/reference'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.cache,a.target,a.interest,a.target_eval,a.interest_eval,a.reference,a.out)
