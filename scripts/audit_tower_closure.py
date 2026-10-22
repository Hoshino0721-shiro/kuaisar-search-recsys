"""Independent data alignment, optimizer/selection and full-user metric audit."""
import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.itemcf_index import DATA, digest
from src.v2.ranking_convergence import write_json
from src.v2.closure_integrity import verify_v1_archive
from src.v2.tower_data import load_metadata, MISSING
from src.v2.tower_model import FeatureTwoTower, model_source_digest
from src.v2.tower_closure import load_training, population, selector, query_arrays
from src.v2.tower_retrieval import TowerIndex


def aligned_training(data, prepared):
    meta=load_metadata(prepared)
    a={k:np.load(prepared/(k+'.npy'),mmap_mode='r') for k in
       ('items','users','times','sample_contexts','context_users','targets','item_fields')}
    cutoff=meta['cutoff_ms']; offset=0; attributes=0
    try:
        with closing(sqlite3.connect((data/'dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)) as db:
            cursor=db.execute("SELECT user_id,item_id,timestamp_ms FROM clean_events WHERE scene='rec' "
                "AND split='train' AND label=1 AND timestamp_ms<? ORDER BY timestamp_ms,item_id,id",(cutoff,))
            while True:
                batch=cursor.fetchmany(65536)
                if not batch: break
                rows=np.asarray(batch,np.int64); n=len(rows); sl=slice(offset,offset+n)
                context=a['sample_contexts'][sl]
                np.testing.assert_array_equal(a['users'][a['context_users'][context]-2],rows[:,0])
                np.testing.assert_array_equal(a['items'][a['targets'][sl]-2],rows[:,1])
                np.testing.assert_array_equal(a['times'][sl],rows[:,2])
                offset+=n
            assert offset==meta['samples']==len(a['targets'])
            vocab=json.loads((prepared/'vocab.json').read_text(encoding='utf-8'))
            cursor=db.execute("SELECT i.item_id,i.author_id,i.category_id,i.feature_type FROM items i JOIN "
                "(SELECT DISTINCT item_id FROM clean_events WHERE scene='rec' AND split='train' "
                "AND label=1 AND timestamp_ms<?) r ON r.item_id=i.item_id ORDER BY i.item_id",(cutoff,))
            for iid,*fields in cursor:
                token=int(np.searchsorted(a['items'],iid))+2
                for col,(name,value) in enumerate(zip(('author','category','type'),fields)):
                    expected=0 if value in MISSING or value==0 else vocab[name][str(value)]
                    assert a['item_fields'][token,col]==expected,(iid,name)
                attributes+=1
            assert attributes==len(a['items'])
        return {'all_training_rows_aligned':offset,'all_item_attributes_checked':attributes,
                'max_training_time':int(a['times'].max()),'cutoff_ms':cutoff,
                'metadata_sha256':digest(prepared/'metadata.json')}
    finally:
        for value in a.values(): value._mmap.close()


def manual_rrf(recent,cf,tower=()):
    scores=defaultdict(float)
    for ids in (recent,cf,tower):
        for j,i in enumerate(ids,1):
            if i>=0: scores[int(i)]+=1/(60+j)
    return sorted(scores,key=lambda i:(-scores[i],i))[:2000]


def manual_quality(ids,positive):
    hits=sum(positive.get(int(i),0) for i in set(ids) if i>=0)
    count=min(20,len(positive))
    ideal=sum(1/np.log2(j+2) for j in range(count))
    dcg=sum(1/np.log2(j+2) for j,i in enumerate(ids[:20]) if int(i) in positive)
    return hits,dcg/ideal if ideal else np.nan


def check_training(model_path, prepared, data):
    report=json.loads((model_path/'metrics.json').read_text(encoding='utf-8'))
    p=report['protocol']; cfg=p['config']
    assert report['status']=='complete' and p['test_used'] is False
    assert p['prepared_sha256']==digest(prepared/'metadata.json')
    assert p['source_sha256']==digest('src/v2/tower_closure.py')
    assert p['model_source_sha256']==model_source_digest()
    assert report['checkpoint_sha256']==digest(model_path/'model.pt')
    best=torch.load(model_path/'model.pt',weights_only=True)
    last=torch.load(model_path/'last.pt',weights_only=True)
    selected=1+int(np.argmax([h['validation']['event_recall'] for h in report['history']]))
    assert selected==report['selected_epoch']==last['best_epoch']==best['selected_epoch']
    for key,value in best['state_dict'].items():
        assert torch.equal(value,last['best_state'][key]),key
    meta,tensors=load_training(prepared)
    model=FeatureTwoTower(meta['cardinalities'],cfg,tensors['item_fields'])
    model.load_state_dict(best['state_dict']); model.eval()
    assert all(torch.count_nonzero(e.weight[0]).item()==0 for e in model.embeddings.values())
    generator=torch.Generator().manual_seed(cfg['seed']+1000)
    best_score=plateau=-1.; bad=lr_bad=reductions=steps=0
    lrs=[cfg['dense_lr'],cfg['sparse_lr']]
    for h in report['history']:
        np.testing.assert_array_equal(h['learning_rates'],lrs)
        order=torch.randperm(meta['samples'],generator=generator)
        assert hashlib.sha256(order.numpy().tobytes()).hexdigest()==h['order_sha256']
        assert h['trained_events']+h['skipped_events']==meta['samples']
        steps+=h['optimizer_steps']
        score=h['validation']['event_recall']
        best_score=max(best_score,score)
        if score>plateau+cfg['min_delta']: plateau=score; bad=lr_bad=0
        else: bad+=1; lr_bad+=1
        if lr_bad>=cfg['lr_patience']:
            lrs=[lr*cfg['lr_factor'] for lr in lrs]; reductions+=1; lr_bad=0
    assert steps==report['optimizer_steps']
    for optimizer in last['optimizers']:
        for state in optimizer['state'].values():
            value=state.get('step')
            if value is not None: assert int(value)==steps
    stop=len(report['history'])>=cfg['min_epochs'] and bad>=cfg['early_patience'] and reductions>=2
    assert (report['early_stopping_reason']=='validation_plateau')==stop
    if not stop: assert len(report['history'])==cfg['max_epochs']
    _,positive=population(data,meta['cutoff_ms'])
    seen={k:v.clone() for k,v in model.named_buffers() if k.startswith('seen_')}
    valid=selector(model,tensors,p['selector_users'],positive,cfg)
    assert valid['hit_events']==report['selected_validation']['hit_events']
    assert valid['positive_events']==report['selected_validation']['positive_events']
    for key,value in seen.items(): assert torch.equal(value,getattr(model,key))
    index=TowerIndex(model_path/'index',digest(data/'manifest.json'))
    try:
        rows=np.sort(np.random.default_rng(2026).choice(len(index.items),min(128,len(index.items)),replace=False))
        with torch.inference_mode(): actual=model.items(torch.from_numpy(rows+2)).numpy()
        np.testing.assert_allclose(actual,index.vectors[rows],atol=2e-6,rtol=2e-5)
        qrows=np.sort(np.random.default_rng(2027).choice(len(index.users),min(32,len(index.users)),replace=False))
        inputs=query_arrays(np.asarray(index.users[qrows]),tensors)
        with torch.inference_mode(): actual=model.users(*inputs).numpy()
        np.testing.assert_allclose(actual,index.queries[qrows],atol=2e-6,rtol=2e-5)
    finally: index.close()
    return {'selected_epoch':selected,'epochs':len(report['history']),'optimizer_steps':steps,
            'selector_hit_events_recomputed':valid['hit_events'],'supported_buffers':len(seen),
            'sampled_export_items':len(rows),'sampled_export_queries':len(qrows)}


def check_metrics(root,job,data):
    name=job['name']; output=root/'evaluation'/name
    report=json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    assert report['source_sha256']==digest('src/v2/tower_closure_evaluate.py')
    for file,sha in report['files'].items(): assert digest(output/file)==sha
    baseline=root/('baseline' if job['window']=='main' else 'earlier_baseline')
    assert report['baseline_metadata_sha256']==digest(baseline/'metadata.json')
    cutoff=json.loads((baseline/'metadata.json').read_text(encoding='utf-8'))['cutoff_ms']
    users,positive=population(data,cutoff)
    ids=np.load(output/'tower_items.npy',mmap_mode='r')
    base={k:np.load(baseline/(k+'.npy'),mmap_mode='r') for k in ('recent','cf_ids','pool','hits','ndcg')}
    try:
        with np.load(output/'per_user.npz') as a:
            np.testing.assert_array_equal(a['users'],users)
            np.testing.assert_array_equal(a['counts'],[sum(positive[u].values()) for u in users])
            for row,uid in enumerate(users):
                p=positive[uid]
                raw=[int(i) for i in ids[row] if i>=0]
                assert len(raw)==len(set(raw)) and len(raw) in (0,1000)
                assert bool(raw)==a['has_query'][row]
                old=manual_rrf(base['recent'],base['cf_ids'][row])
                np.testing.assert_array_equal(old,base['pool'][row,:len(old)])
                current=manual_rrf(base['recent'],base['cf_ids'][row],raw)
                h,n=manual_quality(current,p); h20,_=manual_quality(current[:20],p)
                assert [h,h20]==a['fused_hits'][row].tolist()
                np.testing.assert_allclose(n,a['ndcg'][row],atol=1e-14,rtol=1e-12,equal_nan=True)
                assert [manual_quality(raw[:100],p)[0],manual_quality(raw,p)[0]]==a['tower_hits'][row].tolist()
                assert [h-base['hits'][row,0],h20-base['hits'][row,1]]==a['delta_hits'][row].tolist()
                assert [sum(p.get(i,0) for i in set(current)-set(old)),
                        sum(p.get(i,0) for i in set(old)-set(current))]==a['gained_lost'][row].tolist()
            expected=report['fusion']['all']
            assert int(a['fused_hits'][:,0].sum())==expected['hit_events_at_2000']
            assert int(a['fused_hits'][:,1].sum())==expected['hit_events_at_20']
            np.testing.assert_allclose(np.nanmean(a['ndcg']),expected['period_ndcg_at_20'],atol=1e-14)
        return {'all_users_recomputed':len(users),'positive_events':expected['positive_events'],
                'fusion_hits_at_2000':expected['hit_events_at_2000'],'fusion_hits_at_20':expected['hit_events_at_20']}
    finally:
        ids._mmap.close()
        for value in base.values(): value._mmap.close()


def audit(root,data=DATA):
    torch.set_num_threads(4)
    if (root/'audit.json').exists(): raise FileExistsError('Preserve final audit')
    spec=json.loads((root/'study.json').read_text(encoding='utf-8'))
    prepared={name:aligned_training(data,root/name) for name in ('data','earlier_data')}
    runs={}
    for job in spec['jobs']:
        name=job['name']; print('Auditing '+name,flush=True)
        runs[name]={'training':check_training(Path(job['model']),Path(job['prepared']),data),
                    'metrics':check_metrics(root,job,data)}
    archive=verify_v1_archive()
    result={'status':'passed','prepared':prepared,'runs':runs,'frozen_assets_verified':archive['verified_assets'],
            'v1_archive':archive,
            'checked_sources':{p.as_posix():digest(p) for folder in ('src','scripts')
                               for p in sorted(Path(folder).rglob('*.py'))},
            'test_used':False,'default_changed':False,'source_sha256':digest(Path(__file__)),
            'scope':'full sample alignment / attributes, checkpoint and optimizer accounting, exact selector re-evaluation, full-user logged metrics; not online quality'}
    write_json(root/'audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/tower_closure_20261022'))
    p.add_argument('--data',type=Path,default=DATA);a=p.parse_args()
    print(json.dumps(audit(a.root,a.data),indent=2))
