"""Audited nearby observed-exposure pairs, not inferred requests or negatives."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from src.ctr_features import CACHE, DATA, TZ, connection
from src.v2.train_rankers import sha


def build_pairs(users, items, times, labels, window_ms=1800000, seed=2026):
    users, items, times, labels = [np.asarray(v) for v in (users,items,times,labels)]
    if window_ms < 1 or not (len(users)==len(items)==len(times)==len(labels)) or not np.isin(labels,[0,1]).all():
        raise ValueError('Invalid exposure pairing inputs')
    bins = times//window_ms
    # Fixed deterministic temporal/item ordering; row index breaks equal-time ties.
    order = np.lexsort((np.arange(len(users)),times,items,bins,users))
    cuts = np.r_[0,np.flatnonzero((np.diff(users[order])!=0)|(np.diff(bins[order])!=0))+1,len(order)]
    rng = np.random.default_rng(seed)
    positives, negatives, lengths = [],[],[]
    usable, conflicts, collapsed = 0,0,0
    for a,b in zip(cuts[:-1],cuts[1:]):
        ix = order[a:b]
        if not len(ix):
            continue
        icuts = np.r_[0,np.flatnonzero(np.diff(items[ix]))+1,len(ix)]
        rows = []
        for lo,hi in zip(icuts[:-1],icuts[1:]):
            block = ix[lo:hi]
            if labels[block].min()!=labels[block].max():
                conflicts += len(block)
                continue
            rows.append(block[0])
            collapsed += len(block)-1
        rows = np.asarray(rows,dtype=np.int64)
        p,n = rows[labels[rows]==1],rows[labels[rows]==0]
        if not len(p) or not len(n):
            continue
        usable += 1
        negatives.extend(rng.choice(n,size=len(p),replace=True).tolist())
        positives.extend(p.tolist())
        lengths.extend([len(rows)]*len(p))
    p,n = np.asarray(positives,dtype=np.int64),np.asarray(negatives,dtype=np.int64)
    if len(p):
        assert np.all(users[p]==users[n]) and np.all(bins[p]==bins[n])
        assert np.all(items[p]!=items[n]) and np.all(labels[p]==1) and np.all(labels[n]==0)
    gaps = np.abs(times[p]-times[n])
    report = {'rows':len(users),'pseudo_contexts':max(0,len(cuts)-1) if len(users) else 0,'usable_contexts':usable,
        'pairs':len(p),'paired_users':len(np.unique(users[p])),
        'unique_endpoint_rows':len(np.unique(np.r_[p,n])), 'contradictory_item_rows_excluded':conflicts,
        'consistent_duplicate_rows_collapsed':collapsed,
        'gap_ms_p50_p95_max':np.percentile(gaps,[50,95,100]).tolist() if len(gaps) else None,
        'equal_timestamp_pairs':int(np.sum(gaps==0)), 'all_pair_invariants_pass':True,
        'negative_reuse_allowed':True, 'not_actual_requests':True}
    return p,n,report


def training_times():
    """Replay cache's day/id order and verify every user/item/label before use."""
    arrays={k:np.load(CACHE/f'train_{k}.npy',mmap_mode='r') for k in ('users','items','y')}
    times=np.empty(len(arrays['y']),dtype=np.int64)
    cutoff=json.loads((DATA/'manifest.json').read_text())['cutoffs_ms'][0]
    db=connection()
    cursor=0
    try:
        first=db.execute("SELECT min(timestamp_ms) FROM clean_events WHERE scene='rec' AND split='train'").fetchone()[0]
        day=datetime.fromtimestamp(first/1000,TZ).replace(hour=0,minute=0,second=0,microsecond=0)
        while int(day.timestamp()*1000)<cutoff:
            start=int(day.timestamp()*1000); end=min(start+86400000,cutoff)
            rows=np.asarray(db.execute("SELECT user_id,item_id,label,timestamp_ms FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<? ORDER BY id",(start,end)).fetchall(),dtype=np.int64).reshape(-1,4)
            sl=slice(cursor,cursor+len(rows))
            for col,k in enumerate(('users','items','y')):
                if not np.array_equal(rows[:,col],arrays[k][sl]):
                    raise ValueError('Database/cache order mismatch: '+k)
            times[sl]=rows[:,3];cursor+=len(rows);day+=timedelta(days=1)
        if cursor!=len(times):raise ValueError('Training row count differs')
        return times
    finally:
        db.close()
        for value in arrays.values():value._mmap.close()


def prepare(config, output):
    if output.exists():raise FileExistsError('Preserve existing pair cache')
    cfg=json.loads(config.read_text())
    if cfg['negative_per_positive']!=1:raise ValueError('This protocol supports one negative')
    enhanced=Path(cfg['enhanced_cache'])
    selected=np.load(enhanced/'sample_indices.npy')
    times=training_times()
    output.mkdir(parents=True)
    reports={}; hashes={}
    for split in ('train','valid'):
        raw={k:np.load(CACHE/f'{split}_{k}.npy',mmap_mode='r') for k in ('users','items','y')}
        if split=='train':
            u,i,y=[np.asarray(raw[k][selected]) for k in ('users','items','y')];t=times[selected]
        else:
            u,i,y=[np.asarray(raw[k]) for k in ('users','items','y')];t=np.load(CACHE/'valid_times.npy')
        p,n,r=build_pairs(u,i,t,y,cfg['window_ms'],cfg['pair_seed'])
        if not len(p):raise ValueError('No usable exposure pairs; do not train')
        for name,v in [('positive',p),('negative',n)]:
            path=output/f'{split}_{name}.npy';np.save(path,v);hashes[path.name]=sha(path)
        reports[split]=r
        for value in raw.values():value._mmap.close()
    meta={'config':cfg,'config_sha256':sha(config),'enhanced_data_sha256':sha(enhanced/'metadata.json'),
        'sample_sha256':sha(enhanced/'sample_indices.npy'),'state_sha256':sha(CACHE/'state.npz'),
        'source_sha256':sha(__file__),'test_used':False,'all_training_rows_aligned':True,'splits':reports,'files':hashes}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/pairwise_v1.json'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.config,a.out)
