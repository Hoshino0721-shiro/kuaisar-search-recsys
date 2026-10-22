"""Build strict daily-prefix recall features without reading any test rows."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from src.ctr_features import CACHE, DATA, FEATURES, FeatureState, connection
from src.itemcf_index import IndexConfig, ItemCFIndex, make_snapshot, build, DAY_MS
from src.v2.recall_fusion import select_recall
from src.v2.temporal_route_features import TEMPORAL, ROUTE, WIDTH, temporal_values, recent_order, route_values
from src.v2.train_rankers import Ranker, sha

OFFSET = 8*3600000


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def prefix_index(db, cutoff, config, output, manifest_sha, final_cutoff, final_path):
    if cutoff == final_cutoff:
        index = ItemCFIndex(final_path, manifest_sha)
    else:
        count = db.execute("SELECT count(*) FROM clean_events WHERE scene='rec' AND split='train' AND label=1 AND timestamp_ms<?", (cutoff,)).fetchone()[0]
        if not count:
            return None
        if not output.exists():
            ids = np.fromiter((r[0] for r in db.execute(
                "SELECT DISTINCT item_id FROM clean_events WHERE scene='rec' AND split='train' AND label=1 AND timestamp_ms<? ORDER BY item_id", (cutoff,))), dtype=np.int64)
            rows = db.execute("SELECT user_id,item_id,max(timestamp_ms) t FROM clean_events WHERE scene='rec' AND split='train' AND label=1 AND timestamp_ms<? GROUP BY user_id,item_id ORDER BY user_id,t DESC,item_id", (cutoff,))
            snap = make_snapshot(rows, ids, cutoff, config, manifest_sha)
            build(snap, config, output)
        index = ItemCFIndex(output, manifest_sha)
    if index.metadata['cutoff_ms'] != cutoff or index.metadata['config'] != asdict(config):
        index.close()
        raise ValueError('Prefix index parameters or cutoff differ')
    return index


def cf_query(index, uid, k):
    if index is None:
        return {'items': [], 'scores': []}
    return index.query(int(uid), k)


def fill_route_groups(users, items, target, recent, index, k):
    order = np.argsort(users, kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(np.asarray(users)[order]))+1, len(users)]
    for j, (a,b) in enumerate(zip(cuts[:-1], cuts[1:])):
        ix = order[a:b]
        if not len(ix):
            continue
        uid = int(users[ix[0]])
        result = cf_query(index, uid, k)
        target[ix] = route_values(items[ix], recent, result['items'], result['scores'])
        if j and j%5000 == 0:
            print(f'Route contexts {j}/{len(cuts)-1}', flush=True)


def prepare(config_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new preparation root; partial artifacts are preserved')
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    index_cfg = IndexConfig(**json.loads(Path(cfg['itemcf_config']).read_text(encoding='utf-8')))
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    cutoff = manifest['cutoffs_ms'][0]
    if manifest['status'] != 'ready' or (cutoff+OFFSET)%DAY_MS:
        raise ValueError('Expected completed dataset and Beijing midnight cutoff')
    output.mkdir(parents=True)
    started = perf_counter()
    protocol = {'config': cfg, 'config_sha256': sha(config_path), 'source_sha256': sha(__file__),
                'feature_source_sha256': sha(Path(__file__).with_name('temporal_route_features.py')),
                'manifest_sha256': sha(DATA/'manifest.json'), 'state_sha256': sha(CACHE/'state.npz'),
                'test_used': False, 'cutoff_ms': cutoff, 'feature_names': FEATURES+TEMPORAL+ROUTE}
    write_json(output/'protocol.json', protocol)
    tx = np.load(CACHE/'train_X.npy', mmap_mode='r')
    tu = np.load(CACHE/'train_users.npy', mmap_mode='r')
    ti = np.load(CACHE/'train_items.npy', mmap_mode='r')
    ty = np.load(CACHE/'train_y.npy', mmap_mode='r')
    sample = np.sort(np.random.default_rng(cfg['sample_seed']).choice(len(tx), min(cfg['sample_rows'], len(tx)), replace=False))
    np.save(output/'sample_indices.npy', sample)
    train_x = np.lib.format.open_memmap(output/'train_X.npy', mode='w+', dtype=np.float32, shape=(len(sample), WIDTH))
    train_x[:,:20] = tx[sample]
    np.save(output/'train_y.npy', np.asarray(ty[sample]))
    train_days = np.zeros(len(sample), dtype=np.int64)
    state = FeatureState.load(CACHE/'state.npz')
    size = len(state.ip)
    cumulative_p = np.zeros(size, dtype=np.float64)
    cumulative_n = 0
    rolling = deque(maxlen=3)
    zeros = np.zeros(size, dtype=np.float64)
    day_reports = []
    db = connection()
    try:
        first = db.execute("SELECT min(timestamp_ms) FROM clean_events WHERE scene='rec' AND split='train'").fetchone()[0]
        day = ((first+OFFSET)//DAY_MS)*DAY_MS-OFFSET
        cursor = 0
        while day < cutoff:
            rows = np.asarray(db.execute("SELECT user_id,item_id,timestamp_ms,label FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<? ORDER BY id", (day,day+DAY_MS)).fetchall(), dtype=np.int64).reshape(-1,4)
            count = len(rows)
            for col, values in [(0,tu),(1,ti),(3,ty)]:
                if not np.array_equal(rows[:,col], values[cursor:cursor+count]):
                    raise ValueError('Training source/cache order differs')
            lo,hi = np.searchsorted(sample,[cursor,cursor+count])
            selected = sample[lo:hi]-cursor
            u,i = rows[selected,0], rows[selected,1]
            n1,p1 = rolling[-1] if rolling else (zeros,zeros)
            n3 = sum((a for a,b in rolling), start=zeros.copy())
            p3 = sum((b for a,b in rolling), start=zeros.copy())
            prior = float(cumulative_p.sum()/max(1,cumulative_n)) if cumulative_n else .5
            train_x[lo:hi,20:28] = temporal_values(i,n1,p1,n3,p3,cumulative_p,prior)
            recent = recent_order(p1,cfg['recent_k'])
            index = prefix_index(db,day,index_cfg,output/'indexes'/str(day),protocol['manifest_sha256'],cutoff,cfg['final_index'])
            try:
                block = np.empty((hi-lo,len(ROUTE)), dtype=np.float32)
                fill_route_groups(u,i,block,recent,index,cfg['itemcf_k'])
                train_x[lo:hi,28:] = block
                train_days[lo:hi] = day
                day_reports.append({'cutoff_ms':day,'source_rows':count,'fitting_rows':int(hi-lo),
                                    'index_path':str(index.path) if index else None,
                                    'index_version':index.metadata['version'] if index else None,
                                    'recent_items':len(recent), 'route_hit_fractions':block[:,:2].mean(axis=0).tolist(),
                                    'source_max_timestamp_ms':int(rows[:,2].max()) if count else None})
            finally:
                if index:
                    index.close()
            # Target-day feedback enters state only after all target-day features.
            n = np.bincount(rows[:,1], minlength=size).astype(np.float64)
            p = np.bincount(rows[:,1], weights=rows[:,3], minlength=size)
            rolling.append((n,p))
            cumulative_p += p
            cumulative_n += count
            cursor += count
            day += DAY_MS
            train_x.flush()
            write_json(output/'day_progress.json',day_reports)
            print(f'Completed prior-day features {len(day_reports)}/7; {hi}/{len(sample)} fitting rows',flush=True)
        if cursor != len(tx) or not np.array_equal(cumulative_p,state.ip):
            raise ValueError('Full prior-prefix state does not match original features')
        np.save(output/'train_days.npy',train_days)
        np.save(output/'temporal_state.npy',np.stack([rolling[-1][0],rolling[-1][1],
                sum((a for a,b in rolling),start=zeros.copy()),sum((b for a,b in rolling),start=zeros.copy()),cumulative_p]))
        vusers = np.load(CACHE/'valid_users.npy')
        vitems = np.load(CACHE/'valid_items.npy')
        vx = np.load(CACHE/'valid_X.npy',mmap_mode='r')
        valid_x = np.lib.format.open_memmap(output/'valid_X.npy',mode='w+',dtype=np.float32,shape=(len(vx),WIDTH))
        valid_x[:,:20] = vx
        n1,p1 = rolling[-1]
        n3 = sum((a for a,b in rolling),start=zeros.copy())
        p3 = sum((b for a,b in rolling),start=zeros.copy())
        prior = float(cumulative_p.sum()/cumulative_n)
        valid_x[:,20:28] = temporal_values(vitems,n1,p1,n3,p3,cumulative_p,prior)
        recent = recent_order(p1,cfg['recent_k'])
        index = prefix_index(db,cutoff,index_cfg,output/'unused_final',protocol['manifest_sha256'],cutoff,cfg['final_index'])
        try:
            temp = np.empty((len(vusers),len(ROUTE)),dtype=np.float32)
            fill_route_groups(vusers,vitems,temp,recent,index,cfg['itemcf_k'])
            valid_x[:,28:] = temp
            valid_x.flush()
            users = np.unique(vusers)
            np.save(output/'pool_users.npy',users)
            pool_x = np.lib.format.open_memmap(output/'pool_X.npy',mode='w+',dtype=np.float32,shape=(len(users),cfg['budget']['recall'],WIDTH))
            pool_items = np.lib.format.open_memmap(output/'pool_items.npy',mode='w+',dtype=np.int32,shape=pool_x.shape[:2])
            coarse_positions = np.empty((len(users),cfg['budget']['coarse']),dtype=np.int16)
            ranker = Ranker(cfg['coarse_model'])
            if ranker.metadata['feature_state_sha256'] != protocol['state_sha256']:
                raise ValueError('Fixed coarse model uses a different state')
            with (output/'candidates.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
                for pos,uid in enumerate(users):
                    cf = cf_query(index,int(uid),cfg['itemcf_k'])
                    ranks = {(int(i),'recent'):r for r,i in enumerate(recent,1)}
                    ranks.update({(int(i),'itemcf'):r for r,i in enumerate(cf['items'],1)})
                    ids = select_recall(ranks,cfg['budget']['recall'],{'recent':1.,'itemcf':1.})
                    if len(ids) != cfg['budget']['recall']:
                        raise ValueError('Unexpected short candidate pool')
                    raw = np.asarray(ids,dtype=np.int64)
                    pool_items[pos] = raw
                    base = state.features(np.full(len(ids),int(uid)),raw,np.full(len(ids),cutoff,dtype=np.int64))
                    pool_x[pos,:,:20] = base
                    pool_x[pos,:,20:28] = temporal_values(raw,n1,p1,n3,p3,cumulative_p,prior)
                    pool_x[pos,:,28:] = route_values(raw,recent,cf['items'],cf['scores'])
                    pc = ranker.predict(base)
                    coarse_positions[pos] = np.lexsort((raw,-pc))[:cfg['budget']['coarse']]
                    stream.write(json.dumps({'user_id':int(uid),'timestamp_ms':cutoff,'items':ids},separators=(',',':'))+'\n')
                    if pos%1000 == 0:
                        pool_x.flush();pool_items.flush()
                        print(f'Frozen pool features {pos}/{len(users)}',flush=True)
            pool_x.flush();pool_items.flush()
            pool_x._mmap.close();pool_items._mmap.close()
            np.save(output/'coarse_positions.npy',coarse_positions)
            candidate_sha = sha(output/'candidates.jsonl')
            if candidate_sha != cfg['candidate_reference_sha256']:
                raise ValueError('Current candidates differ from the published optimized pool')
            final_version = index.metadata['version']
        finally:
            index.close()
        train_x.flush();valid_x.flush()
        train_x._mmap.close();valid_x._mmap.close()
        np.save(output/'mean.npy',np.asarray(np.load(output/'train_X.npy',mmap_mode='r')).mean(axis=0,dtype=np.float64).astype(np.float32))
        xx = np.load(output/'train_X.npy',mmap_mode='r')
        np.save(output/'std.npy',np.maximum(xx.std(axis=0,dtype=np.float64),1e-6).astype(np.float32))
        metadata = {**protocol,'days':day_reports,'sample_rows':len(sample),'validation_rows':len(vusers),
                    'validation_route_hit_fractions':temp[:,:2].mean(axis=0).tolist(),
                    'users':len(users),'candidate_sha256':candidate_sha,'final_index_version':final_version,
                    'coarse_checkpoint_sha256':sha(cfg['coarse_model']),'elapsed_seconds':perf_counter()-started,
                    'files':{p.name:sha(p) for p in sorted(output.glob('*.npy'))},'status':'ready'}
        write_json(output/'metadata.json',metadata)
        print(f'Preparation ready in {perf_counter()-started:.1f}s',flush=True)
        return metadata
    finally:
        db.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/temporal_route_factorial.json'))
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    prepare(args.config,args.out)
