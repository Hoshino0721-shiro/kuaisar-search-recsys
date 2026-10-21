"""Independent SQL history replay, exposure alignment, vocabulary/UNK audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.ctr_features import CACHE, connection
from src.v2.enhanced_data import EnhancedData, history_array, DAY
from src.v2.train_rankers import sha


def audit(cache, output):
    if output.exists():
        raise FileExistsError('Preserve audit output')
    with EnhancedData(cache) as data:
        tu = np.load(CACHE/'train_users.npy',mmap_mode='r')
        ti = np.load(CACHE/'train_items.npy',mmap_mode='r')
        vu = np.load(CACHE/'valid_users.npy')
        vi = np.load(CACHE/'valid_items.npy')
        alignment = True
        for lo in range(0,len(data.sample_indices),100000):
            pos = slice(lo,lo+100000)
            raw = data.sample_indices[pos]
            alignment &= bool(np.array_equal(data.train_fields[pos],data.fields(tu[raw],ti[raw])))
            alignment &= bool(np.array_equal(data.context_users[data.train_context[pos]],tu[raw]))
        alignment &= bool(np.array_equal(data.valid_fields,data.fields(vu,vi)))
        checks = []
        db = connection()
        def expected(uid, cutoff, scene):
            sql = "SELECT item_id FROM clean_events WHERE split='train' AND scene=? AND user_id=? AND label>0 AND timestamp_ms<?"
            params = [scene,uid,cutoff]
            if scene == 'search':
                sql += ' AND timestamp_ms>=?'
                params.append(cutoff-DAY)
            sql += ' ORDER BY timestamp_ms DESC,item_id DESC,id DESC LIMIT ?'
            params.append(data.meta['history_length'])
            raw = [r[0] for r in db.execute(sql,params)]
            tokens = [int(data.item_fields[i,0]) if 0 <= i < len(data.item_fields) else 1 for i in reversed(raw)]
            return history_array(tokens,data.meta['history_length'])
        try:
            offset = 0
            for day in data.meta['days']:
                lo,hi = np.searchsorted(data.sample_indices,[offset,offset+day['rows']])
                for p in np.linspace(lo,hi-1,min(3,hi-lo),dtype=int):
                    uid = int(tu[data.sample_indices[p]])
                    context = data.train_context[p]
                    checks.append({'scope':'train','user':uid,'cutoff':day['start_ms'],
                        'rec':bool(np.array_equal(expected(uid,day['start_ms'],'rec'),data.context_rec[context])),
                        'search':bool(np.array_equal(expected(uid,day['start_ms'],'search'),data.context_search[context]))})
                offset += day['rows']
            users = np.unique(vu)
            for uid in users[np.linspace(0,len(users)-1,8,dtype=int)]:
                checks.append({'scope':'query','user':int(uid),'cutoff':data.meta['cutoff_ms'],
                    'rec':bool(np.array_equal(expected(int(uid),data.meta['cutoff_ms'],'rec'),data.query_rec[uid])),
                    'search':bool(np.array_equal(expected(int(uid),data.meta['cutoff_ms'],'search'),data.query_search[uid]))})
        finally:
            db.close()
        vocabulary = {}
        for col,name in enumerate(('user','item','author','category')):
            counts = np.bincount(data.train_fields[:,col],minlength=data.meta['cardinalities'][name])
            if name == 'item':
                weights = np.bincount(data.train_context,minlength=len(data.context_users))
                for h in (data.context_rec,data.context_search):
                    counts += np.bincount(h.ravel(),weights=np.repeat(weights,h.shape[1]),minlength=len(counts)).astype(np.int64)
            vocabulary[name] = {'cardinality':len(counts),'allocated_but_unseen_in_fit':int((counts[2:]==0).sum()),
                                'minimum_known_fit_frequency':int(counts[2:].min()) if len(counts)>2 else None,
                                'validation_unk_fraction':float(np.mean(data.valid_fields[:,col]==1))}
        histories = {}
        for name,a in [('train_rec_contexts',data.context_rec),('train_search_contexts',data.context_search),
                       ('valid_rec_rows',data.query_rec[vu]),('valid_search_rows',data.query_search[vu])]:
            valid = a != 0
            histories[name] = {'rows':len(a),'nonempty_fraction':float(np.mean(valid.any(axis=1))),
                                'unk_fraction_of_nonpad':float((a==1).sum()/max(1,valid.sum()))}
        passed = bool(alignment and all(c['rec'] and c['search'] for c in checks)
                      and all(v['allocated_but_unseen_in_fit']==0 for v in vocabulary.values()))
        result = {'passed':passed,'test_used':False,'all_exposure_fields_and_context_users_aligned':alignment,
                  'history_sql_checks':checks,'vocabulary':vocabulary,'histories':histories,
                  'data_sha256':sha(cache/'metadata.json'),'source_sha256':sha(__file__),
                  'scope':'full field alignment, sampled SQL history replay; static attributes assumed, not proven time-available'}
    output.mkdir(parents=True)
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='history_sql_checks'},indent=2))
    if not passed:
        raise RuntimeError('Enhanced audit failed; inspect report')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    audit(a.cache,a.out)
