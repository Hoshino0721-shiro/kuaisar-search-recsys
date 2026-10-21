"""Check final pairwise models against actual inputs and independent sums."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.ctr_features import CACHE
from src.v2.enhanced_data import EnhancedData
from src.v2.enhanced_model import EnhancedRanker
from src.v2.train_rankers import sha


def audit(root,pair_root,output):
    if output.exists():raise FileExistsError('Preserve final audit')
    train=json.loads((root/'models/metrics.json').read_text());evaluation=json.loads((root/'evaluation/metrics.json').read_text())
    if train['test_used'] or evaluation['test_used']:raise ValueError('Only validation accepted')
    cfg=train['protocol']['config'];meta=json.loads((pair_root/'metadata.json').read_text())
    pair_hash=sha(pair_root/'metadata.json');state_hash=sha(CACHE/'state.npz')
    assert pair_hash==train['protocol']['pairs_sha256']
    assert state_hash==meta['state_sha256']
    for name,digest in meta['files'].items():assert Path(name).name==name and sha(pair_root/name)==digest
    pool=Path(cfg['candidate_pool']);poolmeta=json.loads((pool/'metadata.json').read_text())
    assert sha(pool/'candidates.jsonl')==evaluation['candidate_sha256']==poolmeta['candidate_sha256']
    selected=np.load(Path(cfg['enhanced_cache'])/'sample_indices.npy')
    arrays={split:{k:np.load(CACHE/f'{split}_{k}.npy',mmap_mode='r') for k in ('users','items','y')} for split in ('train','valid')}
    pairs={}
    for split in arrays:
        p,n=[np.load(pair_root/f'{split}_{k}.npy') for k in ('positive','negative')]
        values={k:np.asarray(v[selected]) if split=='train' else np.asarray(v) for k,v in arrays[split].items()}
        assert np.all(values['users'][p]==values['users'][n]) and np.all(values['items'][p]!=values['items'][n])
        assert np.all(values['y'][p]==1) and np.all(values['y'][n]==0)
        if split=='valid':
            times=np.load(CACHE/'valid_times.npy');assert np.all(times[p]//cfg['window_ms']==times[n]//cfg['window_ms'])
        pairs[split]=(p,n)
    valid_times=np.load(CACHE/'valid_times.npy');vp,vn=pairs['valid']
    same_time=valid_times[vp]==valid_times[vn]
    assert int(same_time.sum())==meta['splits']['valid']['equal_timestamp_pairs']
    torch.set_num_threads(4);checks={}
    with EnhancedData(cfg['enhanced_cache']) as data:
        assert sha(data.path/'metadata.json')==meta['enhanced_data_sha256']
        x=np.load(CACHE/'valid_X.npy',mmap_mode='r')
        for name,run in train['runs'].items():
            model_path=root/'models'/name/'model.pt';m=run['metadata']
            assert sha(model_path)==run['checkpoint_sha256'] and m['pair_metadata_sha256']==pair_hash
            assert m['model_source_sha256']==sha(Path(__file__).with_name('enhanced_model.py'))
            selected_epoch=min(run['history'],key=lambda h:h['valid_pair']['logloss'])['epoch']
            assert selected_epoch==m['selected_epoch']
            ranker=EnhancedRanker(model_path,data,state_hash)
            objective=m['experiment'].split('_',1)[1]
            if m['spec']['attrs']:
                inputs=np.asarray(data.train_fields)
                if objective not in ('full_bce','full_mixed'):
                    inputs=inputs[np.r_[pairs['train'][0],pairs['train'][1]]]
                for field,col in [('author',2),('category',3)]:
                    expected=np.bincount(inputs[:,col],minlength=data.meta['cardinalities'][field])>=5
                    expected[:2]=True
                    assert np.array_equal(expected,getattr(ranker.model,'known_'+field).numpy()),(name,field)
            cached=np.load(root/'models'/name/'valid_predictions.npy');logits=np.load(root/'models'/name/'valid_logits.npy')
            assert len(cached)==len(arrays['valid']['y']) and np.isfinite(cached).all() and np.isfinite(logits).all()
            actual=ranker.predict(x[:4096],data.valid_fields[:4096]);delta=float(np.max(np.abs(actual-cached[:4096])))
            assert delta<1e-6,(name,delta)
            p,n=pairs['valid'];difference=logits[p].astype(float)-logits[n].astype(float)
            loss=float(np.logaddexp(0,-difference).mean())
            assert abs(loss-run['valid_pair']['logloss'])<1e-10
            sensitivity={}
            for group,mask in [('same_timestamp',same_time),('different_timestamp',~same_time)]:
                d=difference[mask]
                sensitivity[group]={'pairs':int(mask.sum()),'pair_logloss':float(np.logaddexp(0,-d).mean()),
                    'accuracy_half_ties':float(np.mean((d>0)+.5*(d==0)))}
            checks[name]={'checkpoint_verified':True,'input_coverage_verified':True,'selected_epoch_verified':True,
                'reload_probability_max_error':delta,'valid_pair_loss_recomputed':loss,
                'posthoc_time_matching_sensitivity':sensitivity,
                'sigmoid_saturation_rows':int(((cached==0)|(cached==1)).sum())}
        x._mmap.close()
    users=[];direct={n:0 for n in train['runs']};stages={n:{s:0 for s in st} for n,st in evaluation['cascade'].items()}
    for line in (root/'evaluation/per_user_progress.jsonl').open(encoding='utf-8'):
        row=json.loads(line);users.append(row['user_id'])
        for n,q in row['methods'].items():direct[n]+=q['hit_events']
        for n,st in row['cascade'].items():
            for s,q in st.items():stages[n][s]+=q['hit_events']
    assert users==poolmeta['users'] and len(set(users))==len(users)
    for n,hits in direct.items():assert hits==evaluation['direct'][n]['hit_events']
    for n,st in stages.items():
        for s,hits in st.items():assert hits==evaluation['cascade'][n][s]['hit_events']
    freeze=json.loads(Path('experiments/final_test/freeze.json').read_text())
    assert all(sha(p)==digest for p,digest in freeze['asset_hashes'].items())
    for split in arrays.values():
        for v in split.values():v._mmap.close()
    report={'passed':True,'test_used':False,'models':checks,'users':len(users),'all_pair_labels_and_users_match':True,
        'valid_pair_time_bins_verified':True,'train_time_alignment':'pair preparation already replayed full DB/cache order; not independently replayed again here',
        'all_model_specific_coverage_exact':True,'all_per_user_sums_match':True,'frozen_v1_unchanged':True,
        'source_sha256':sha(__file__),'candidate_sha256':evaluation['candidate_sha256'],'pair_metadata_sha256':pair_hash}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8');print('Pairwise final audit passed:',len(checks),'models',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('artifacts/pairwise_final_v1'))
    p.add_argument('--pairs',type=Path,default=Path('artifacts/pairwise_v1/pairs'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.run,a.pairs,a.out)
