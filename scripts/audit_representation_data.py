"""Independent train-only counts, PAD, raw caption and SQL history checks."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from src.ctr_features import CACHE, connection
from src.v2.ranking_convergence import write_json
from src.v2.representation_data import verify, DAY
from src.v2.train_rankers import sha


def audit(cache, output):
    if output.exists():raise FileExistsError('Preserve audit')
    meta=verify(cache); v=json.loads((cache/'vocab.json').read_text())
    data={name:np.load(cache/(name+'.npy'),mmap_mode='r') for name in
          ('train_context','context_users','context_times','context_rec','context_search','query_rec','query_search',
           'item_fields','user_lookup','item_frequency','item_input_frequency','caption_raw','caption')}
    u=np.load(CACHE/'train_users.npy');i=np.load(CACHE/'train_items.npy'); cfg=meta['config'];minimum=cfg['minimum_frequency']
    np.testing.assert_array_equal(data['item_frequency'],np.bincount(i,minlength=len(data['item_fields'])))
    np.testing.assert_array_equal(data['context_users'][data['train_context']],u)
    weights=np.bincount(data['train_context'],minlength=len(data['context_users']))
    expected=np.array(data['item_frequency'])
    for key in ('context_rec','context_search'):
        for lo in range(0,len(weights),4096):
            a=data[key][lo:lo+4096]
            expected+=np.bincount(a.ravel(),weights=np.repeat(weights[lo:lo+len(a)],a.shape[1]),minlength=len(expected)).astype(np.int64)
    expected[0]=0
    np.testing.assert_array_equal(expected,data['item_input_frequency'])
    np.testing.assert_array_equal(v['item'],np.flatnonzero(expected>=minimum))
    np.testing.assert_array_equal(v['user'],np.flatnonzero(np.bincount(u,minlength=len(data['user_lookup']))>=minimum))
    if '0' in v['author'] or '0' in v['category']:raise ValueError('Raw missing attribute leaked into vocabulary')
    db=connection();rng=np.random.default_rng(2026);sql_checks=0
    try:
        for row in rng.choice(len(weights),64,replace=False):
            uid=int(data['context_users'][row]);cut=int(data['context_times'][row])
            for scene,key in [('rec','context_rec'),('search','context_search')]:
                extra=" AND timestamp_ms>=? AND result_type IN ('VIDEO','IMAGE_ATLAS')" if scene=='search' else ''
                args=[scene,uid,cut]+([cut-DAY] if scene=='search' else [])+[cfg['history_length']]
                rows=db.execute("SELECT item_id FROM clean_events WHERE scene=? AND split='train' AND label=1 AND user_id=? AND timestamp_ms<?"+extra+' ORDER BY timestamp_ms DESC,item_id DESC,id DESC LIMIT ?',args).fetchall()
                values=[r[0] for r in reversed(rows)];wanted=[0]*(cfg['history_length']-len(values))+values
                np.testing.assert_array_equal(data[key][row],wanted);sql_checks+=1
        for uid in rng.choice(len(data['query_rec']),32,replace=False):
            for scene,key in [('rec','query_rec'),('search','query_search')]:
                extra=" AND timestamp_ms>=? AND result_type IN ('VIDEO','IMAGE_ATLAS')" if scene=='search' else ''
                args=[scene,int(uid),meta['cutoff_ms']]+([meta['cutoff_ms']-DAY] if scene=='search' else [])+[cfg['history_length']]
                rows=db.execute("SELECT item_id FROM clean_events WHERE scene=? AND split='train' AND label=1 AND user_id=? AND timestamp_ms<?"+extra+' ORDER BY timestamp_ms DESC,item_id DESC,id DESC LIMIT ?',args).fetchall()
                values=[r[0] for r in reversed(rows)]
                np.testing.assert_array_equal(data[key][uid],[0]*(cfg['history_length']-len(values))+values);sql_checks+=1
        # All author raw0 rows, not just an arbitrary sample.
        for iid, in db.execute("SELECT item_id FROM items WHERE author_id='0'"):
            if data['item_fields'][iid,1]!=0:raise ValueError('Author raw0 not PAD')
    finally:db.close()
    # Word vocabulary frequency independently recomputed over every actual input.
    nwords=int(data['caption_raw'].max())+1; counts=np.zeros(nwords,np.int64)
    lookup=np.ones(nwords,np.int32)
    for j,raw in enumerate(v['word'],2):lookup[raw]=j
    for lo in range(0,len(expected),100000):
        a=np.asarray(data['caption_raw'][lo:lo+100000]);keep=a>=0
        count=np.bincount(a[keep],weights=np.repeat(expected[lo:lo+len(a)],a.shape[1]).reshape(a.shape)[keep],minlength=nwords)
        counts+=count.astype(np.int64)
        encoded=np.where(keep,lookup[np.maximum(a,0)],0)
        np.testing.assert_array_equal(encoded,data['caption'][lo:lo+len(a)])
    keys=np.flatnonzero(counts>=minimum)
    chosen=keys[np.lexsort((keys,-counts[keys]))[:cfg['word_limit']]]
    np.testing.assert_array_equal(sorted(chosen),v['word'])
    selected=set(rng.choice(np.flatnonzero(data['item_frequency']>0),64,replace=False).tolist());checked=0
    with zipfile.ZipFile('data/raw/KuaiSAR.zip') as z:
        with io.TextIOWrapper(z.open('KuaiSAR_final/item_features.csv'),encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                iid=int(row['item_id'])
                if iid in selected:
                    raw=json.loads(row['caption'])[:cfg['caption_length']]
                    np.testing.assert_array_equal(data['caption_raw'][iid],raw+[-1]*(cfg['caption_length']-len(raw)))
                    checked+=1
    if checked!=len(selected):raise ValueError('Raw caption sample missing')
    write_json(output,{'status':'passed','test_used':False,'all_train_input_counts_recomputed':True,
               'all_word_vocab_counts_and_encoding_recomputed':True,'all_author_raw0_is_pad':True,
               'sql_histories_checked':sql_checks,'raw_caption_records_checked':checked,
               'source_sha256':sha(__file__),'cache_sha256':sha(cache/'metadata.json')})
    for a in data.values():a._mmap.close()
    print('Representation data audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('artifacts/representation_20261022/cache'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.cache,a.out)
