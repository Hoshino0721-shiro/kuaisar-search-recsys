"""Compact local demonstration; optional tower is never the default policy."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.v2.recommend import Backend, DEFAULT_CONFIG


def demonstrate(user,with_tower=False,root=Path('artifacts/tower_closure_20261022')):
    cfg=json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    variants=[('default_recent_itemcf_rrf',cfg)]
    if with_tower:
        index=root/'models/history_s2026/index'
        if not (index/'metadata.json').exists(): raise ValueError('No trained/exported history tower; run the study first')
        experimental=json.loads(json.dumps(cfg))
        experimental['tower_index']=str(index)
        experimental['route_limits']['tower']=1000
        experimental['route_weights']['tower']=1.
        variants.append(('experimental_history_tower_rrf',experimental))
    result={'scope':'local offline frozen snapshot; not a deployed service or online CTR prediction',
            'user_id':user,'default_modified':False,'variants':{}}
    for name,config in variants:
        backend=Backend(config)
        try:
            routes=backend.retrieve(user)
            output=backend.run(user)
            rows=[]
            for iid in output['items']:
                ranks={c.source:c.rank for c in routes if c.item_id==iid}
                rows.append({'item_id':iid,'route_ranks':ranks,
                             'rrf_score':sum(config['route_weights'].get(s,1.)/(60+r) for s,r in ranks.items())})
            result['variants'][name]={'snapshot_ms':output['timestamp_ms'],'items':rows,
                'candidate_count':output['stages'][0]['output_count'],
                'itemcf_index_version':backend.itemcf_index.metadata['version'],
                'tower_index_version':backend.tower_index.metadata['version'] if backend.tower_index else None,
                'request_ms':output['request_elapsed_ms'],'timing_scope':'one local request, not benchmark/QPS'}
        finally: backend.close()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--user-id',type=int,default=20235)
    p.add_argument('--with-tower',action='store_true');p.add_argument('--root',type=Path,default=Path('artifacts/tower_closure_20261022'))
    a=p.parse_args();print(json.dumps(demonstrate(a.user_id,a.with_tower,a.root),ensure_ascii=False,indent=2))
