"""Replay sampled histories from train-only SQL, without training any model."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.itemcf_index import DATA,digest
from src.v2.tower_data import DAY,OFFSET,load_metadata,padded


def audit(data,prepared,output):
    if output.exists():
        raise FileExistsError('Preserve previous data audit')
    meta=load_metadata(prepared)
    if digest(data/'manifest.json')!=meta['manifest_sha256']:
        raise ValueError('Data manifest differs')
    arrays={name:np.load(prepared/(name+'.npy'),mmap_mode='r') for name in (
        'items','users','times','sample_contexts','context_users','context_rec','context_search',
        'query_users','query_rec','query_search')}
    items=arrays['items']
    db=sqlite3.connect((data/'dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)

    def history(uid,upper,search=False):
        scene='search' if search else 'rec'
        extra="AND timestamp_ms>=? AND result_type IN ('VIDEO','IMAGE_ATLAS')" if search else ''
        params=(uid,scene,upper,upper-DAY) if search else (uid,scene,upper)
        rows=db.execute("SELECT item_id FROM clean_events WHERE user_id=? AND scene=? AND split='train' "
                        "AND label=1 AND timestamp_ms<? "+extra+
                        ' ORDER BY timestamp_ms DESC,item_id DESC,id DESC',params)
        length=meta['search_length'] if search else meta['history_length']
        collected=[]
        seen=set()
        for (raw,) in rows:
            pos=int(np.searchsorted(items,raw))
            if pos<len(items) and items[pos]==raw and raw not in seen:
                collected.append(pos+2)
                seen.add(raw)
                if len(collected)==length:
                    break
        return padded(reversed(collected),length)

    try:
        rng=np.random.default_rng(2026)
        sampled=rng.choice(len(arrays['times']),min(64,len(arrays['times'])),replace=False)
        for sample in sampled:
            context=int(arrays['sample_contexts'][sample])
            uid=int(arrays['users'][int(arrays['context_users'][context])-2])
            ts=int(arrays['times'][sample])
            upper=((ts+OFFSET)//DAY)*DAY-OFFSET
            assert ts<meta['cutoff_ms']
            np.testing.assert_array_equal(arrays['context_rec'][context],history(uid,upper))
            np.testing.assert_array_equal(arrays['context_search'][context],history(uid,upper,True))
        queries=rng.choice(len(arrays['query_users']),min(32,len(arrays['query_users'])),replace=False)
        for row in queries:
            uid=int(arrays['query_users'][row])
            np.testing.assert_array_equal(arrays['query_rec'][row],history(uid,meta['cutoff_ms']))
            np.testing.assert_array_equal(arrays['query_search'][row],history(uid,meta['cutoff_ms'],True))
        result={'status':'passed','sampled_targets':len(sampled),'sampled_frozen_queries':len(queries),
                'checks':['prepared file checksums match','sampled training contexts equal independent SQL prefixes',
                          'sampled query histories equal train-end SQL snapshot','no validation/test history read'],
                'data_metadata_sha256':digest(prepared/'metadata.json'),'source_sha256':digest(Path(__file__)),
                'scope':'sampled data correctness, not model quality or exhaustive row audit',
                'real_model_training_run':False,'test_labels_used':False}
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
        return result
    finally:
        db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    audit(args.data,args.prepared,args.out)
