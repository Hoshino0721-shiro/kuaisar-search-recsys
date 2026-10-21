"""Train-only dual-tower data, day-lagged histories, frozen query snapshot.

PAD=0, UNK=1. Contexts are shared by user/day rather than copied per event.
No validation or test interactions are read during preparation.
"""
import argparse
from array import array
from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import sqlite3
import numpy as np
from src.itemcf_index import DATA, digest

DAY = 86400000
OFFSET = 8 * 3600000
MISSING = {None, '', '0', 'NULL', 'null', 'unknown', 'UNKNOWN', 'UNKNOW'}


def recent_add(state, user, item, length):
    history = state[user]
    history.pop(item, None)
    history[item] = None
    while len(history) > length:
        history.popitem(last=False)


def padded(history, length):
    values = list(history)[-length:]
    return [0] * (length - len(values)) + values


def prepare(data, output, history_length=30, search_length=20):
    if output.exists():
        raise FileExistsError('Tower data is immutable; choose a new directory')
    if min(history_length, search_length) < 1:
        raise ValueError('History lengths must be positive')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'ready':
        raise ValueError('Dataset must be ready')
    cutoff = manifest['cutoffs_ms'][0]
    if (cutoff + OFFSET) % DAY:
        raise ValueError('Day-lagged preparation requires a Beijing midnight cutoff')
    db = sqlite3.connect((data/'dataset.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        where = "scene='rec' AND split='train' AND label=1 AND timestamp_ms<?"
        items = np.fromiter((r[0] for r in db.execute(
            'SELECT DISTINCT item_id FROM clean_events WHERE '+where+' ORDER BY item_id', (cutoff,))), dtype=np.int64)
        users = np.fromiter((r[0] for r in db.execute(
            'SELECT DISTINCT user_id FROM clean_events WHERE '+where+' ORDER BY user_id', (cutoff,))), dtype=np.int64)
        if not len(items) or not len(users):
            raise ValueError('No recommendation training positives')
        item_map = {int(i): j+2 for j,i in enumerate(items)}
        user_map = {int(u): j+2 for j,u in enumerate(users)}
        fields = np.ones((len(items)+2,3), dtype=np.int32)
        fields[0] = 0
        vocabs = {name:{} for name in ('author','category','type')}
        for iid, author, category, typ in db.execute(
                'SELECT item_id,author_id,category_id,feature_type FROM items ORDER BY item_id'):
            if iid not in item_map:
                continue
            for column,(name,value) in enumerate(zip(vocabs,(author,category,typ))):
                if value not in MISSING:
                    key=str(value)
                    fields[item_map[iid],column] = vocabs[name].setdefault(key,len(vocabs[name])+2)
        rec = defaultdict(OrderedDict)
        search = defaultdict(OrderedDict)
        contexts, target_ids, sample_times = array('i'), array('i'), array('q')
        context_users, rec_flat, search_flat = array('i'), array('i'), array('i')
        first = db.execute("SELECT min(timestamp_ms) FROM clean_events WHERE scene IN ('rec','search') "
                           "AND split='train' AND label=1 AND timestamp_ms<?",(cutoff,)).fetchone()[0]
        day = ((first+OFFSET)//DAY)*DAY-OFFSET
        search_rows = search_known = 0
        while day < cutoff:
            end = min(day+DAY,cutoff)
            day_context = {}
            rows = db.execute(
                "SELECT user_id,item_id,timestamp_ms FROM clean_events WHERE scene='rec' AND split='train' "
                "AND label=1 AND timestamp_ms>=? AND timestamp_ms<? ORDER BY timestamp_ms,item_id,id",(day,end)).fetchall()
            # A day's targets all see the same previous-completed-day state.
            for uid,iid,ts in rows:
                if uid not in day_context:
                    day_context[uid] = len(context_users)
                    context_users.append(user_map[uid])
                    rec_flat.extend(padded(rec[uid],history_length))
                    search_flat.extend(padded(search[uid],search_length))
                contexts.append(day_context[uid])
                target_ids.append(item_map[iid])
                sample_times.append(ts)
            for uid,iid,_ in rows:
                recent_add(rec,uid,item_map[iid],history_length)
            search = defaultdict(OrderedDict)
            for uid,iid in db.execute(
                    "SELECT user_id,item_id FROM clean_events WHERE scene='search' AND split='train' "
                    "AND label=1 AND timestamp_ms>=? AND timestamp_ms<? "
                    "AND result_type IN ('VIDEO','IMAGE_ATLAS') ORDER BY timestamp_ms,item_id,id",(day,end)):
                search_rows += 1
                if iid in item_map:
                    search_known += 1
                    recent_add(search,uid,item_map[iid],search_length)
            print(f'Tower day {(day+OFFSET)//DAY}: {len(rows)} positive targets',flush=True)
            day += DAY
        query_users = np.asarray(sorted(set(rec)|set(search)),dtype=np.int64)
        values = {
            'items':items, 'users':users, 'item_fields':fields,
            'sample_contexts':np.asarray(contexts,dtype=np.int32),
            'targets':np.asarray(target_ids,dtype=np.int32),
            'times':np.asarray(sample_times,dtype=np.int64),
            'context_users':np.asarray(context_users,dtype=np.int32),
            'context_rec':np.asarray(rec_flat,dtype=np.int32).reshape(-1,history_length),
            'context_search':np.asarray(search_flat,dtype=np.int32).reshape(-1,search_length),
            'query_users':query_users,
            'query_user_tokens':np.asarray([user_map.get(int(u),1) for u in query_users],dtype=np.int32),
            'query_rec':np.asarray([padded(rec[int(u)],history_length) for u in query_users],dtype=np.int32),
            'query_search':np.asarray([padded(search[int(u)],search_length) for u in query_users],dtype=np.int32)}
        values['target_counts'] = np.bincount(values['targets'],minlength=len(items)+2)
        output.mkdir(parents=True,exist_ok=False)
        for name,value in values.items():
            np.save(output/(name+'.npy'),value)
        (output/'vocab.json').write_text(json.dumps(vocabs),encoding='utf-8')
        metadata = {
            'status':'ready','schema_version':1,'cutoff_ms':cutoff,
            'manifest_sha256':digest(data/'manifest.json'),'source_sha256':digest(Path(__file__)),
            'history_length':history_length,'search_length':search_length,
            'samples':len(contexts),'contexts':len(context_users),
            'cardinalities':{'user':len(users)+2,'item':len(items)+2,
                             **{name:len(v)+2 for name,v in vocabs.items()}},
            'catalog':'recommendation train clicked items; same catalog as ItemCF',
            'training_targets':'observed recommendation train click=1 events, repeats retained',
            'history_rule':'previous complete Beijing days; latest distinct recommendation clicks, previous day search clicks only',
            'query_rule':'frozen training-end history; no validation/test updates',
            'metadata_assumption':'author/category/type treated as static; no attribute snapshot timestamps',
            'search_seed_rows':search_rows,'search_known_seed_rows':search_known,
            'test_labels_used':False,
            'files':{p.name:digest(p) for p in sorted(output.iterdir())}}
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        return metadata
    finally:
        db.close()


def load_metadata(path, verify=True):
    meta = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('status')!='ready' or meta.get('schema_version')!=1:
        raise ValueError('Unsupported/incomplete tower data')
    if verify:
        for name,expected in meta['files'].items():
            if Path(name).name!=name or digest(path/name)!=expected:
                raise ValueError('Tower data checksum mismatch: '+name)
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--history-length',type=int,default=30)
    parser.add_argument('--search-length',type=int,default=20)
    args=parser.parse_args()
    print(json.dumps(prepare(args.data,args.out,args.history_length,args.search_length),indent=2))
