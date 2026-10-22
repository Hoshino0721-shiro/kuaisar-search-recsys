"""Recompute direct-output metrics from archived candidate IDs and SQLite labels."""
import argparse
from collections import Counter, defaultdict
import json
from math import log2
from pathlib import Path
import numpy as np
from src.ctr_features import connection
from src.v2.recommend import DEFAULT_CONFIG
from src.v2.temporal_route_data import write_json
from src.v2.train_rankers import sha


def audit(report_path, cache):
    report=json.loads(report_path.read_text(encoding='utf-8'))
    meta=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    config=json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    assert report['config']==config
    assert report['candidate_sha256']==meta['candidate_sha256']
    assert report['itemcf_index_version']==meta['final_index_version']
    assert not report['test_used'] and not report['model_weights_loaded']
    assert sha('src/v2/recommend.py')==report['source_sha256']
    assert sha('src/v2/rrf_output.py')==report['rrf_source_sha256']
    users=np.load(cache/'pool_users.npy')
    items=np.load(cache/'pool_items.npy',mmap_mode='r')
    positives=defaultdict(Counter);observed_users=set()
    db=connection()
    try:
        for u,i,y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
            observed_users.add(u)
            if y:
                positives[u][i]+=1
    finally:
        db.close()
    assert sorted(observed_users)==users.tolist()
    hits=pool=total=0;ndcg=[]
    for j,u in enumerate(users):
        labels=positives[int(u)];selected=set(items[j,:20].tolist());available=set(items[j].tolist())
        total+=sum(labels.values())
        hits+=sum(n for i,n in labels.items() if i in selected)
        pool+=sum(n for i,n in labels.items() if i in available)
        ideal=sum(1/log2(r+2) for r in range(min(20,len(labels))))
        if ideal:
            ndcg.append(sum(1/log2(r+2) for r,i in enumerate(items[j,:20]) if i in labels)/ideal)
    assert report['users']==len(users)==8495
    assert report['hit_events_at20']==hits==1745
    assert report['pool_hit_events']==pool==34647
    assert report['positive_events']==total==354626
    assert abs(report['ndcg_at20']-np.mean(ndcg))<1e-14
    assert report['event_recall_at20']==hits/total
    result={'passed':True,'users':len(users),'hit_events_at20':hits,'pool_hit_events':pool,
            'positive_events':total,'ndcg_at20':float(np.mean(ndcg)),
            'candidate_sha256':report['candidate_sha256'],'same_ordered_pool_as_previous_experiments':True,
            'source_sha256':sha(__file__),'report_sha256':sha(report_path),
            'test_used':False,'scope':'full user metrics independently recomputed using frozen candidates and validation-only SQLite labels'}
    write_json(report_path.with_name('audit.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('artifacts/temporal_route_20261022/cache'))
    a=p.parse_args();audit(a.report,a.cache)
