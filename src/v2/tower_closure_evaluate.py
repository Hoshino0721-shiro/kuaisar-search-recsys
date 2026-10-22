"""Full-user fixed-budget dual-tower evaluation and cacheable strong controls."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np

from src.itemcf_index import DATA, ItemCFIndex, digest
from src.v2.cascade import Budget, Candidate, observed_positive_metrics
from src.v2.ranking_convergence import write_json
from src.v2.recall_fusion import select_recall
from src.v2.rrf_output import recommend_rrf
from src.v2.tower_closure import population
from src.v2.tower_retrieval import TowerIndex
from src.v2.train_tower import export


def quality(ids, positive):
    result = observed_positive_metrics([int(i) for i in ids if i >= 0], positive)
    return result['hit_events'], result['user_period_ndcg_at_20']


def prepare_baseline(data, cf_path, output):
    expected = {'manifest_sha256': digest(data/'manifest.json'),
                'itemcf_metadata_sha256': digest(cf_path/'metadata.json'),
                'source_sha256': digest(Path(__file__)),
                'options': {'neighbor_limit': 300, 'half_life_days': None}, 'test_used': False}
    if output.exists():
        meta = json.loads((output/'metadata.json').read_text(encoding='utf-8'))
        if meta['protocol'] != expected: raise ValueError('Baseline protocol differs')
        for name, sha in meta['files'].items():
            if digest(output/name) != sha: raise ValueError('Baseline checksum mismatch')
        return meta
    cf = ItemCFIndex(cf_path, expected['manifest_sha256'])
    try:
        cutoff = cf.metadata['cutoff_ms']
        users, positives = population(data, cutoff)
        with closing(sqlite3.connect((data/'dataset.sqlite').resolve().as_uri()+'?mode=ro', uri=True)) as db:
            recent = [r[0] for r in db.execute("SELECT item_id,count(*) n FROM clean_events "
                "WHERE scene='rec' AND split='train' AND label=1 AND timestamp_ms>=? AND timestamp_ms<? "
                "GROUP BY item_id ORDER BY n DESC,item_id LIMIT 2000",(cutoff-86400000,cutoff))]
        output.mkdir(parents=True, exist_ok=False)
        cf_ids = np.full((len(users),1000),-1,dtype=np.int64)
        pools = np.full((len(users),2000),-1,dtype=np.int64)
        hits = np.zeros((len(users),2),np.int64)
        ndcg = np.full(len(users),np.nan)
        counts = np.array([sum(positives[u].values()) for u in users],np.int64)
        query_ms = []
        for row, uid in enumerate(users):
            tick = perf_counter()
            ids = cf.query(uid,1000,neighbor_limit=300,half_life_days=None)['items']
            query_ms.append((perf_counter()-tick)*1000)
            cf_ids[row,:len(ids)] = ids
            ranks = {(i,'recent'):j for j,i in enumerate(recent,1)}
            ranks.update({(i,'itemcf'):j for j,i in enumerate(ids,1)})
            selected = select_recall(ranks,2000,{'recent':1.,'itemcf':1.})
            pools[row,:len(selected)] = selected
            hits[row,0], _ = quality(selected, positives[uid])
            hits[row,1], value = quality(selected[:20], positives[uid])
            if value is not None: ndcg[row] = value
        for name, value in [('users',users),('recent',recent),('cf_ids',cf_ids),('pool',pools),
                            ('hits',hits),('counts',counts),('ndcg',ndcg)]:
            np.save(output/(name+'.npy'),np.asarray(value))
        meta = {'status':'ready','protocol':expected,'cutoff_ms':cutoff,'users':len(users),
                'positive_events':int(counts.sum()),'hit_events_at_2000':int(hits[:,0].sum()),
                'hit_events_at_20':int(hits[:,1].sum()),'period_ndcg_at_20':float(np.nanmean(ndcg)),
                'cf_query_p50_ms':float(np.percentile(query_ms,50)),
                'cf_query_p95_ms':float(np.percentile(query_ms,95)),
                'files':{p.name:digest(p) for p in output.glob('*.npy')}}
        write_json(output/'metadata.json',meta)
        return meta
    finally: cf.close()


def summarize_arrays(values, counts, mask=None):
    mask = np.ones(len(counts),bool) if mask is None else mask
    count = int(counts[mask].sum())
    return {'users':int(mask.sum()),'positive_events':count,
            'hit_events_at_2000':int(values['fused_hits'][mask,0].sum()),
            'recall_at_2000':float(values['fused_hits'][mask,0].sum()/count) if count else None,
            'hit_events_at_20':int(values['fused_hits'][mask,1].sum()),
            'recall_at_20':float(values['fused_hits'][mask,1].sum()/count) if count else None,
            'period_ndcg_at_20':float(np.nanmean(values['ndcg'][mask])) if np.isfinite(values['ndcg'][mask]).any() else None,
            'net_hit_events_at_2000':int(values['delta_hits'][mask,0].sum()),
            'net_hit_events_at_20':int(values['delta_hits'][mask,1].sum())}


def evaluate(prepared, model_dir, baseline, output, data=DATA):
    if (output/'metrics.json').exists():
        result = json.loads((output/'metrics.json').read_text(encoding='utf-8'))
        if result['checkpoint_sha256'] != digest(model_dir/'model.pt'):
            raise ValueError('Existing evaluation belongs to different weights')
        for name, sha in result['files'].items():
            if digest(output/name) != sha: raise ValueError('Evaluation checksum mismatch')
        return result
    if output.exists(): raise FileExistsError('Partial evaluation retained; use a new output path')
    index_path = model_dir/'index'
    if not index_path.exists(): export(prepared,model_dir/'model.pt',index_path,'cpu')
    index = TowerIndex(index_path,digest(data/'manifest.json'))
    try:
        base_meta = json.loads((baseline/'metadata.json').read_text(encoding='utf-8'))
        if base_meta['cutoff_ms'] != index.metadata['cutoff_ms']:
            raise ValueError('Baseline and tower cutoff differ')
        if index.metadata['checkpoint_sha256'] != digest(model_dir/'model.pt'):
            raise ValueError('Tower export differs from checkpoint')
        users, positives = population(data, index.metadata['cutoff_ms'])
        base = {name:np.load(baseline/(name+'.npy'),mmap_mode='r')
                for name in ('users','recent','cf_ids','pool','hits','counts','ndcg')}
        np.testing.assert_array_equal(base['users'],users)
        protocol = json.loads((model_dir/'protocol.json').read_text(encoding='utf-8'))
        selection = np.isin(users, protocol['selector_users'])
        output.mkdir(parents=True,exist_ok=False)
        ids = np.lib.format.open_memmap(output/'tower_items.npy',mode='w+',dtype='int64',shape=(len(users),1000))
        ids[:] = -1
        signal = np.zeros(len(users),bool)
        batched_ms = []
        start = perf_counter()
        for lo in range(0,len(users),64):
            tick = perf_counter()
            rows = index.query_batch(users[lo:lo+64],1000)
            batched_ms.append((perf_counter()-tick)*1000)
            for j,row in enumerate(rows):
                ids[lo+j,:len(row['items'])] = row['items']
                signal[lo+j] = bool(row['items'])
            if lo//64 % 16 == 0:
                print(f'{model_dir.name}: exact retrieval {min(lo+64,len(users))}/{len(users)}',flush=True)
        retrieval_seconds = perf_counter()-start
        ids.flush()
        tower_hits = np.zeros((len(users),2),np.int64)
        fused_hits = np.zeros((len(users),2),np.int64)
        gained_lost = np.zeros((len(users),2),np.int64)
        outside = np.zeros(len(users),np.int64)
        ndcg = np.full(len(users),np.nan)
        catalog_counts = np.zeros(len(users),np.int64)
        catalog = set(index.items.tolist())
        for row,uid in enumerate(users):
            p = positives[uid]
            tower = [int(i) for i in ids[row] if i>=0]
            recent = [int(i) for i in base['recent']]
            cf = [int(i) for i in base['cf_ids'][row] if i>=0]
            ranks = {(i,'recent'):j for j,i in enumerate(recent,1)}
            ranks.update({(i,'itemcf'):j for j,i in enumerate(cf,1)})
            ranks.update({(i,'tower'):j for j,i in enumerate(tower,1)})
            fused = select_recall(ranks,2000,{'recent':1.,'itemcf':1.,'tower':1.})
            tower_hits[row,0], _ = quality(tower[:100],p)
            tower_hits[row,1], _ = quality(tower,p)
            fused_hits[row,0], _ = quality(fused,p)
            fused_hits[row,1], value = quality(fused[:20],p)
            if value is not None: ndcg[row] = value
            old = set(int(i) for i in base['pool'][row] if i>=0)
            new = set(fused)
            gained_lost[row] = [sum(p.get(i,0) for i in new-old),sum(p.get(i,0) for i in old-new)]
            outside[row] = sum(p.get(i,0) for i in set(tower)-old)
            catalog_counts[row] = sum(n for i,n in p.items() if i in catalog)
        values = {'users':np.asarray(users),'counts':np.asarray(base['counts']),'tower_hits':tower_hits,
                  'fused_hits':fused_hits,'delta_hits':fused_hits-base['hits'], 'ndcg':ndcg,
                  'delta_ndcg':ndcg-base['ndcg'],'gained_lost':gained_lost,'outside_base':outside,
                  'catalog_counts':catalog_counts,'has_query':signal,'selection':selection}
        np.savez_compressed(output/'per_user.npz',**values)
        # Actual single-user full-catalog calls: offline warm-process latency,
        # plus served RRF replay. Not network/QPS/online-serving performance.
        measured = np.random.default_rng(2026).choice(len(users),min(8,len(users)),replace=False)
        latency, replays, set_differences = [],0,[]
        for row in measured:
            tick = perf_counter(); result = index.query(users[row],1000)
            latency.append((perf_counter()-tick)*1000)
            tower = result['items']
            set_differences.append(len(set(tower)^set(int(i) for i in ids[row] if i>=0)))
            routes = [Candidate(int(i),s,j,index.metadata['cutoff_ms']) for s,items in
                [('recent',base['recent']),('itemcf',[i for i in base['cf_ids'][row] if i>=0]),('tower',tower)]
                for j,i in enumerate(items,1)]
            served = recommend_rrf(users[row],index.metadata['cutoff_ms'],routes,Budget(),
                                   {'recent':1.,'itemcf':1.,'tower':1.},positives[users[row]])
            independent = select_recall({(c.item_id,c.source):c.rank for c in routes},2000,
                                        {'recent':1.,'itemcf':1.,'tower':1.})
            assert served['items'] == independent[:20]
            replays += 1
        denom = int(base['counts'].sum())
        result = {'status':'complete','test_used':False,'default_changed':False,
                  'scope':'frozen one-day logged-positive coverage, not unbiased relevance or online CTR',
                  'config':index.metadata['config'],'source_sha256':digest(Path(__file__)),
                  'checkpoint_sha256':index.metadata['checkpoint_sha256'],'index_version':index.metadata['version'],
                  'baseline_metadata_sha256':digest(baseline/'metadata.json'),
                  'catalog_items':len(index.items),'catalog_positive_events':int(catalog_counts.sum()),
                  'catalog_event_coverage':float(catalog_counts.sum()/denom),
                  'pure_tower':{'hits_at_100':int(tower_hits[:,0].sum()),'hits_at_1000':int(tower_hits[:,1].sum()),
                                'recall_at_100':float(tower_hits[:,0].sum()/denom),
                                'recall_at_1000':float(tower_hits[:,1].sum()/denom),
                                'distinct_returned_items':len(np.unique(ids[ids>=0])),
                                'has_query_users':int(signal.sum()),'mean_candidates':float(signal.mean()*1000)},
                  'fusion':{name:summarize_arrays(values,base['counts'],mask) for name,mask in
                            [('all',None),('selector_users',selection),('other_development_users',~selection),
                             ('has_query',signal),('no_query',~signal)]},
                  'baseline':{k:base_meta[k] for k in ('hit_events_at_2000','hit_events_at_20','period_ndcg_at_20')},
                  'gained_events':int(gained_lost[:,0].sum()),'lost_events':int(gained_lost[:,1].sum()),
                  'tower_positive_events_outside_base_pool':int(outside.sum()),
                  'index_array_bytes':sum((index_path/name).stat().st_size for name in index.metadata['files']),
                  'exact_retrieval_seconds':retrieval_seconds,'retrieval_batch_users':64,
                  'retrieval_batch_p50_ms':float(np.percentile(batched_ms,50)),
                  'retrieval_batch_p95_ms':float(np.percentile(batched_ms,95)),
                  'single_user_samples':len(latency),'single_user_p50_ms':float(np.percentile(latency,50)),
                  'single_user_p95_ms':float(np.percentile(latency,95)),
                  'latency_scope':'warm local CPU exact retrieval only; 8 sampled users; no QPS/network/production claim',
                  'served_rrf_replays':replays,'single_vs_batch_top1000_symmetric_difference':set_differences,
                  'float_note':'float32 GEMM batch shapes may perturb near-tied boundary scores; single-user replay uses its actual list',
                  'files':{p.name:digest(p) for p in output.iterdir() if p.suffix in ('.npy','.npz')}}
        write_json(output/'metrics.json',result)
        return result
    finally:
        if 'ids' in locals(): ids._mmap.close()
        for value in locals().get('base',{}).values(): value._mmap.close()
        index.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--itemcf-index',type=Path,default=Path('artifacts/itemcf_support2_300'))
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    prepare_baseline(args.data,args.itemcf_index,args.baseline)
    print(json.dumps(evaluate(args.prepared,args.model,args.baseline,args.out,args.data),indent=2))
