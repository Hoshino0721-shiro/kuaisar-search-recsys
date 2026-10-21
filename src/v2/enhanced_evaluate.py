"""Freeze optimized recall and compare enhanced DCNs on the same validation pool."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.ctr_features import CACHE, FeatureState, connection
from src.v2.cascade import Candidate, Cascade, observed_positive_metrics, Budget
from src.v2.features import coarse_features
from src.v2.enhanced_data import EnhancedData
from src.v2.enhanced_model import EnhancedRanker
from src.v2.recall_fusion import select_recall
from src.v2.recommend import Backend
from src.v2.ranking_diagnostics import rank_pool
from src.v2.train_rankers import sha


def labels():
    db = connection()
    users, positives = set(), defaultdict(Counter)
    try:
        for u,i,y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
            users.add(u)
            if y:
                positives[u][i] += 1
    finally:
        db.close()
    return sorted(users), positives


def freeze(config_path, output):
    if output.exists():
        raise FileExistsError('Frozen candidate directory exists')
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    backend = Backend(cfg)
    users, positives = labels()
    output.mkdir(parents=True)
    hits, count = 0, 0
    try:
        with (output/'candidates.jsonl').open('x', encoding='utf-8', newline='\n') as f:
            for uid in users:
                ranks = {(c.item_id,c.source):c.rank for c in backend.retrieve(uid,backend.snapshot)}
                ids = select_recall(ranks, backend.budget.recall, cfg.get('route_weights'), cfg.get('fusion'))
                f.write(json.dumps({'user_id':uid,'timestamp_ms':backend.snapshot,'items':ids}, separators=(',',':'))+'\n')
                hits += sum(n for i,n in positives[uid].items() if i in set(ids))
                count += len(ids)
        meta = {'split':'valid','test_used':False,'users':users,'config':cfg,'snapshot_ms':backend.snapshot,
                'candidate_sha256':sha(output/'candidates.jsonl'), 'hit_events':hits,
                'positive_events':sum(sum(v.values()) for v in positives.values()),
                'mean_candidates':count/len(users), 'itemcf_version':backend.itemcf_index.metadata['version'],
                'recent_top20':backend.recent[:20], 'state_sha256':sha(CACHE/'state.npz')}
        (output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in meta.items() if k not in ('users','config')},indent=2))
    finally:
        backend.close()


def evaluate(models_path, cache, pool, output):
    if output.exists():
        raise FileExistsError('Evaluation output exists')
    training = json.loads((models_path/'metrics.json').read_text(encoding='utf-8'))
    meta = json.loads((pool/'metadata.json').read_text(encoding='utf-8'))
    if meta['test_used'] or meta['split'] != 'valid' or sha(pool/'candidates.jsonl') != meta['candidate_sha256']:
        raise ValueError('Invalid frozen validation pool')
    data = EnhancedData(cache)
    state = FeatureState.load(CACHE/'state.npz')
    state_hash = sha(CACHE/'state.npz')
    if state_hash != meta['state_sha256'] or state_hash != data.meta['feature_state_sha256']:
        raise ValueError('Snapshot mismatch')
    torch.set_num_threads(4)
    users, positives = labels()
    if users != meta['users']:
        raise ValueError('Validation user set changed')
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps({'models':str(models_path),'cache':str(cache),
        'candidate_sha256':meta['candidate_sha256'],'data_sha256':sha(cache/'metadata.json'),
        'source_sha256':sha(__file__),'test_used':False},indent=2),encoding='utf-8')
    progress_file = (output/'per_user_progress.jsonl').open('x',encoding='utf-8',newline='\n')
    models = {name:EnhancedRanker(models_path/name/'model.pt',data,state_hash) for name in training['runs']}
    aggregate = {name:{'hits':0,'ndcg':[],'times':[]} for name in models}
    baselines = {name:{'hits':0,'ndcg':[]} for name in ('rrf','recent')}
    # Same coarse model and same budgets for original-statistical and selected fine models.
    from src.v2.train_rankers import Ranker
    coarse_model = Ranker(meta['config']['coarse_model'])
    best_name = min(training['runs'],key=lambda n:(training['runs'][n]['validation']['all']['logloss'],n))
    coarse_compare = ['A_s2026'] + ([best_name] if best_name != 'A_s2026' else [])
    stage_aggregate = {n:{s:{'hits':0,'ndcg':[]} for s in ('recall','coarse','fine','rerank')} for n in coarse_compare}
    budget = Budget(**meta['config']['budget'])
    per_user = []
    for line in (pool/'candidates.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        uid, ids, ts = row['user_id'],row['items'],row['timestamp_ms']
        raw = np.asarray(ids,dtype=np.int64)
        u = np.full(len(ids),uid)
        x = state.features(u,raw,np.full(len(ids),ts))
        fields = data.fields(u,raw)
        rec = np.repeat(np.asarray(data.query_rec[uid])[None,:],len(ids),axis=0)
        search = np.repeat(np.asarray(data.query_search[uid])[None,:],len(ids),axis=0)
        record = {'user_id':uid,'methods':{}}
        record['scoring_ms'] = {}
        record['cascade'] = {}
        score_maps = {}
        for name,model in models.items():
            tick = perf_counter()
            scores = model.predict(x,fields,rec if model.metadata['spec']['rec'] else None,
                                   search if model.metadata['spec']['search'] else None)
            elapsed = (perf_counter()-tick)*1000
            order = rank_pool(ids,scores)[:20]
            q = observed_positive_metrics(order,positives[uid])
            aggregate[name]['hits'] += q['hit_events']
            if q['user_period_ndcg_at_20'] is not None:
                aggregate[name]['ndcg'].append(q['user_period_ndcg_at_20'])
            aggregate[name]['times'].append(elapsed)
            record['methods'][name] = q
            record['scoring_ms'][name] = elapsed
            if name in coarse_compare:
                score_maps[name] = dict(zip(ids,scores.tolist()))
        for name,order in [('rrf',ids[:20]),('recent',meta['recent_top20'])]:
            q = observed_positive_metrics(order,positives[uid])
            baselines[name]['hits'] += q['hit_events']
            if q['user_period_ndcg_at_20'] is not None:
                baselines[name]['ndcg'].append(q['user_period_ndcg_at_20'])
        coarse_scores = dict(zip(ids,coarse_model.predict_selected(coarse_features(state,u,raw)).tolist()))
        categories = {i:int(state.categories[i]) for i in ids}
        routes = [Candidate(i,'frozen',j,ts) for j,i in enumerate(ids,1)]
        for name in coarse_compare:
            cascade = Cascade(lambda u,items,t:[coarse_scores[i] for i in items],
                              lambda u,items,t:[score_maps[name][i] for i in items],budget,categories=categories)
            result = cascade.run(uid,ts,routes,positives=positives[uid])
            record['cascade'][name] = {}
            for stage in result['stages']:
                a = stage_aggregate[name][stage['stage']]
                a['hits'] += stage['hit_events']
                if stage['user_period_ndcg_at_20'] is not None:
                    a['ndcg'].append(stage['user_period_ndcg_at_20'])
                record['cascade'][name][stage['stage']] = {k:stage[k] for k in ('hit_events','positive_events','user_period_ndcg_at_20')}
        per_user.append(record)
        progress_file.write(json.dumps(record,separators=(',',':'))+'\n')
        if len(per_user)%500 == 0:
            progress_file.flush()
            print(f'Enhanced fixed-pool evaluation {len(per_user)}/{len(users)}',flush=True)
    total = meta['positive_events']
    summary = {name:{'hit_events':a['hits'],'event_recall_at20':a['hits']/total,
                    'ndcg_at20':float(np.mean(a['ndcg'])),
                    'scoring_p50_ms':float(np.percentile(a['times'],50)),
                    'scoring_p95_ms':float(np.percentile(a['times'],95))} for name,a in aggregate.items()}
    stage_summary = {}
    for name, stages in stage_aggregate.items():
        previous = None
        stage_summary[name] = {}
        for stage,a in stages.items():
            stage_summary[name][stage] = {'hit_events':a['hits'],'retention':a['hits']/previous if previous else None,
                                         'ndcg_at20':float(np.mean(a['ndcg']))}
            previous = a['hits']
    progress_file.close()
    report = {'split':'valid','test_used':False,'candidate_metadata':meta,'direct':summary,
              'baselines':{n:{'hit_events':a['hits'],'ndcg_at20':float(np.mean(a['ndcg']))} for n,a in baselines.items()},
              'cascade':stage_summary,'cascade_selected_model':best_name,
              'timing':'CPU fine scorer only, includes field tensor conversion; excludes shared numeric/attribute/history construction, retrieval, network, concurrency; first request included',
              'cascade_timing':'cached scores for quality diagnosis only, NOT serving latency',
              'default_changed':False,'source_sha256':sha(__file__)}
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'per_user.json').write_text(json.dumps(per_user),encoding='utf-8')
    data.close()
    print(f'Enhanced evaluation complete: {output}',flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['freeze','evaluate'])
    p.add_argument('--config',type=Path,default=Path('configs/cascade_v2_recall_optimized.json'))
    p.add_argument('--cache',type=Path,default=Path('artifacts/dcn_enhancement_v1/cache'))
    p.add_argument('--models',type=Path,default=Path('artifacts/dcn_enhancement_v1/models'))
    p.add_argument('--pool',type=Path,default=Path('artifacts/dcn_enhancement_v1/candidates'))
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    freeze(a.config,a.out) if a.command == 'freeze' else evaluate(a.models,a.cache,a.pool,a.out)
