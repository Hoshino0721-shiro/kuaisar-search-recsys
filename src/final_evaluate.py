"""Freeze validated choices before materializing held-out features and scores."""
import hashlib
import json
import time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from src.ctr_features import CACHE,DATA,write_eval,FeatureState,connection
from src.ranking import OUT as RANK_OUT,metrics
from src.retrieval_pipeline import OUT as PIPE_OUT,CandidateScorer,train_lists,tower_for_users,make_sources,rank_union,rerank,quality
from src.search_ranking import OUT as SEARCH_OUT,SEARCH_CACHE,prepare as prepare_search,lists_metrics

OUT=Path('experiments/final_test')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as h:
        for b in iter(lambda:h.read(1024*1024),b''):digest.update(b)
    return digest.hexdigest()


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'freeze.json'
    if path.exists():
        frozen=json.loads(path.read_text())
    else:
        rank=json.loads((RANK_OUT/'metrics.json').read_text())
        pipeline=json.loads((PIPE_OUT/'metrics.json').read_text())['config']
        search=json.loads((SEARCH_OUT/'metrics.json').read_text())
        frozen={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'selection_split':'valid',
                'ranker':rank['selected'],'retrieval':pipeline['selected_retrieval'],
                'ctr_pipeline':pipeline['selected_ctr_pipeline'],'inference_timestamp_ms':1685289600000,
                'search_model':'hgb' if search['model']['logloss']<search['constant']['logloss'] else 'constant',
                'test_policy':'one report, no parameter changes based on test; training and feature statistics stay frozen',
                'relevance_policy':'rec list metrics are user-period observed positives, search list metrics are logged sessions'}
        assets=list(Path('src').glob('*.py'))+[DATA/'manifest.json',CACHE/'state.npz',SEARCH_CACHE/'state.npz',
            RANK_OUT/'scaler.joblib',RANK_OUT/'config.json',RANK_OUT/'mlp_gate.pt',RANK_OUT/'metrics.json',
            PIPE_OUT/'metrics.json',SEARCH_OUT/'model.joblib',SEARCH_OUT/'metrics.json',
            Path('experiments/r2_history_attributes_valid/model.pt')]
        frozen['asset_hashes']={str(p):sha(p) for p in assets}
        path.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    for asset,expected in frozen['asset_hashes'].items():
        if sha(asset)!=expected:raise ValueError(f'Frozen asset changed: {asset}')
    return frozen


def main():
    if (OUT/'metrics.json').exists():
        print('Final test already completed; keeping original report.');return
    frozen=freeze();print('Configuration frozen before test feature materialization',frozen['ranker'],frozen['retrieval'],flush=True)
    write_eval('test');scorer=CandidateScorer(frozen['ranker'])
    x=np.load(CACHE/'test_X.npy',mmap_mode='r');y=np.load(CACHE/'test_y.npy');u=np.load(CACHE/'test_users.npy')
    selected=scorer.predict(x);np.save(OUT/'ctr_predictions.npy',selected)
    exposure={}
    predictions={'constant':np.full(len(y),.4972061833392528),'item_ctr':np.asarray(x[:,2]),frozen['ranker']:selected}
    for name,p in predictions.items():
        exposure[name]=metrics(y,p,u)
        exposure[name]['groups']={}
        for group,mask in [('has_search',x[:,19]>0),('no_search',x[:,19]==0),('warm_item',x[:,10]>0),('cold_item',x[:,10]==0)]:
            if mask.any():exposure[name]['groups'][group]=metrics(y[mask],p[mask],u[mask])
        print('Test CTR',name,{k:exposure[name][k] for k in ('auc','gauc','logloss')},flush=True)
    state=FeatureState.load(CACHE/'state.npz');db=connection();popular,recent=train_lists(db,state)
    positive=defaultdict(Counter);users=set()
    for uid,iid,label in db.execute("SELECT user_id,item_id,label FROM events WHERE scene='rec' AND split='test' AND exclusion IS NULL"):
        users.add(uid)
        if label:positive[uid][iid]+=1
    db.close();users=sorted(users);start=time.perf_counter()
    tower=tower_for_users(users);variants=make_sources(users,popular,recent,tower)
    lists={'popular':variants['popular'],'selected_retrieval':variants[frozen['retrieval']]}
    candidates,scores,offset=rank_union(users,popular,recent,tower,state,scorer,frozen['inference_timestamp_ms'])
    penalty=float(frozen['ctr_pipeline'].split('_')[-1])
    lists['selected_ctr_pipeline']={uid:rerank(candidates[uid],scores[offset[j]:offset[j+1]],state.categories,penalty) for j,uid in enumerate(users)}
    list_report={name:quality(ranked,positive,state.categories) for name,ranked in lists.items()}
    for name,ranked in lists.items():
        with (OUT/f'{name}_candidates.jsonl').open('w') as handle:
            for uid,items in ranked.items():
                assert len(items)==len(set(items))==100
                handle.write(json.dumps({'user_id':uid,'items':items})+'\n')
    pipeline_seconds=time.perf_counter()-start
    prepare_search('test');sx=np.load(SEARCH_CACHE/'test_X.npy',mmap_mode='r');sy=np.load(SEARCH_CACHE/'test_y.npy')
    sg=np.load(SEARCH_CACHE/'test_groups.npy');si=np.load(SEARCH_CACHE/'test_items.npy')
    prior=float(np.load(SEARCH_CACHE/'train_y.npy').mean());model=joblib.load(SEARCH_OUT/'model.joblib')
    with threadpool_limits(limits=8):sp=model.predict_proba(sx)[:,1] if frozen['search_model']=='hgb' else np.full(len(sy),prior)
    np.save(OUT/'search_predictions.npy',sp)
    search={'model':metrics(sy,sp,sg),'lists':lists_metrics(sy,sp,sg,si),
            'constant':metrics(sy,np.full(len(sy),prior),sg),'constant_lists':lists_metrics(sy,np.full(len(sy),prior),sg,si)}
    report={'freeze':frozen,'ctr_exposure_metrics':exposure,'recommendation_list_metrics':list_report,
            'search':search,'test_pipeline_seconds_including_loading_scoring_evaluation':pipeline_seconds,
            'test_completed_at_utc':datetime.now(timezone.utc).isoformat()}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Final recall',{k:v['recall_at_100'] for k,v in list_report.items()},flush=True)
    print('Search test',search['lists'],flush=True)


if __name__=='__main__':main()
