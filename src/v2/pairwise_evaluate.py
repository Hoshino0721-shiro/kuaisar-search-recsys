"""Fixed-pool pairwise objective comparison; validation only, no new labels."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from src.ctr_features import CACHE, FeatureState
from src.v2.cascade import Budget, Candidate, Cascade, observed_positive_metrics
from src.v2.enhanced_data import EnhancedData
from src.v2.enhanced_model import EnhancedRanker
from src.v2.enhanced_evaluate import labels
from src.v2.features import coarse_features
from src.v2.ranking_diagnostics import rank_pool
from src.v2.ranking_sensitivity import paired_interval
from src.v2.train_rankers import Ranker, sha


def evaluate(root,output):
    if output.exists():raise FileExistsError('Preserve completed pool results')
    training=json.loads((root/'models/metrics.json').read_text())
    cfg=training['protocol']['config'];pool=Path(cfg['candidate_pool'])
    meta=json.loads((pool/'metadata.json').read_text())
    if meta['test_used'] or training['test_used'] or meta['candidate_sha256']!=sha(pool/'candidates.jsonl'):
        raise ValueError('Invalid validation artifacts')
    state_hash=sha(CACHE/'state.npz')
    if state_hash!=meta['state_sha256']:raise ValueError('Candidate snapshot mismatch')
    torch.set_num_threads(4)
    users,positives=labels()
    if users!=meta['users']:raise ValueError('User set changed')
    output.mkdir(parents=True)
    data=EnhancedData(cfg['enhanced_cache']);state=FeatureState.load(CACHE/'state.npz')
    models={}
    for name,r in training['runs'].items():
        if r['checkpoint_sha256']!=sha(root/'models'/name/'model.pt'):raise ValueError('Checkpoint changed')
        models[name]=EnhancedRanker(root/'models'/name/'model.pt',data,state_hash)
    coarse=Ranker(meta['config']['coarse_model'])
    if coarse.metadata['feature_state_sha256']!=state_hash:raise ValueError('Coarse state mismatch')
    budget=Budget(**meta['config']['budget']);methods=cfg['cascade_methods']
    direct={n:{'hits':0,'ndcg':[]} for n in models}
    stages={n:{s:{'hits':0,'ndcg':[]} for s in ('recall','coarse','fine','rerank','without_rerank')} for n in methods}
    baselines={n:{'hits':0,'ndcg':[]} for n in ('rrf','recent')}
    per_user=[];durations=[]
    (output/'protocol.json').write_text(json.dumps({'config':cfg,'source_sha256':sha(__file__),
        'candidate_sha256':meta['candidate_sha256'],'models_report_sha256':sha(root/'models/metrics.json'),
        'test_used':False,'timing':'batched cached quality evaluation; NOT serving latency'},indent=2),encoding='utf-8')
    tick=perf_counter()
    def add(aggregate,q):
        aggregate['hits']+=q['hit_events']
        if q['user_period_ndcg_at_20'] is not None:aggregate['ndcg'].append(q['user_period_ndcg_at_20'])
    def process(block,progress):
        raw=np.concatenate([np.asarray(r['items']) for r in block])
        u=np.concatenate([np.full(len(r['items']),r['user_id']) for r in block])
        ts=np.concatenate([np.full(len(r['items']),r['timestamp_ms']) for r in block])
        x=state.features(u,raw,ts);fields=data.fields(u,raw)
        scores={n:m.predict(x,fields) for n,m in models.items()}
        cheap=coarse.predict_selected(coarse_features(state,u,raw))
        offset=0
        for row in block:
            uid,ids,t=row['user_id'],row['items'],row['timestamp_ms'];sl=slice(offset,offset+len(ids));offset+=len(ids)
            record={'user_id':uid,'methods':{},'cascade':{}}
            for n in models:
                q=observed_positive_metrics(rank_pool(ids,scores[n][sl])[:20],positives[uid])
                add(direct[n],q);record['methods'][n]=q
            for n,order in [('rrf',ids[:20]),('recent',meta['recent_top20'])]:
                add(baselines[n],observed_positive_metrics(order,positives[uid]))
            coarse_map=dict(zip(ids,cheap[sl].tolist()))
            categories={i:int(state.categories[i]) for i in ids}
            routes=[Candidate(i,'frozen',j,t) for j,i in enumerate(ids,1)]
            for n in methods:
                score_map=dict(zip(ids,scores[n][sl].tolist()))
                cascade=Cascade(lambda u,items,t:[coarse_map[i] for i in items],
                    lambda u,items,t:[score_map[i] for i in items],budget,categories=categories)
                result=cascade.run(uid,t,routes,positives=positives[uid])
                record['cascade'][n]={}
                for stage in result['stages']:
                    s=stage['stage'];q={k:stage[k] for k in ('hit_events','positive_events','user_period_ndcg_at_20')}
                    add(stages[n][s],q);record['cascade'][n][s]=q
                fine=next(s['items'] for s in result['stages'] if s['stage']=='fine')
                q=observed_positive_metrics(fine[:20],positives[uid])
                add(stages[n]['without_rerank'],q);record['cascade'][n]['without_rerank']=q
            per_user.append(record);progress.write(json.dumps(record,separators=(',',':'))+'\n')
    try:
        with (output/'per_user_progress.jsonl').open('x',encoding='utf-8',newline='\n') as progress:
            block=[]
            for line in (pool/'candidates.jsonl').open(encoding='utf-8'):
                block.append(json.loads(line))
                if len(block)==8:
                    process(block,progress);block=[]
                    if len(per_user)%256==0:
                        progress.flush();print(f'Pairwise fixed-pool {len(per_user)}/{len(users)}',flush=True)
            if block:process(block,progress)
        if [r['user_id'] for r in per_user]!=users:raise ValueError('Incomplete or reordered pool')
        def compact(a):return {'hit_events':a['hits'],'event_recall_at20':a['hits']/meta['positive_events'],
            'ndcg_at20':float(np.mean(a['ndcg']))}
        direct_report={n:compact(a) for n,a in direct.items()}
        # Stage budgets vary: event recall name must not claim Top20 for 2000/400/80.
        stage_report={n:{s:{'hit_events':a['hits'],'ndcg_at20':float(np.mean(a['ndcg']))} for s,a in st.items()} for n,st in stages.items()}
        comparisons={}
        for architecture in cfg['models']:
            pairs=[('full_bce','paired_bce'),('paired_bce','pairwise'),('paired_bce','mixed')]
            if 'full_mixed' in cfg['objectives']:pairs.append(('full_bce','full_mixed'))
            for before,after in pairs:
                delta=[]
                for row in per_user:
                    a=[row['methods'][f'{architecture}_{before}_s{seed}']['user_period_ndcg_at_20'] for seed in cfg['seeds']]
                    b=[row['methods'][f'{architecture}_{after}_s{seed}']['user_period_ndcg_at_20'] for seed in cfg['seeds']]
                    if all(v is not None for v in a+b):delta.append(np.mean(b)-np.mean(a))
                comparisons[f'{architecture}: {after} minus {before}']=paired_interval(delta)
        report={'split':'valid','test_used':False,'users':len(users),'positive_events':meta['positive_events'],
            'direct':direct_report,'cascade':stage_report,'baselines':{n:compact(a) for n,a in baselines.items()},
            'paired_ndcg':comparisons,'candidate_sha256':meta['candidate_sha256'],
            'elapsed_seconds':perf_counter()-tick,'timing':'batched cached quality, NOT latency or service SLA',
            'bootstrap_scope':'exploratory user bootstrap after averaging seeds; one reused validation day; no multiple-testing adjustment',
            'default_changed':False,'source_sha256':sha(__file__)}
        (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Pairwise evaluation complete',flush=True)
    finally:data.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('artifacts/pairwise_v1'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();evaluate(a.run,a.out)
