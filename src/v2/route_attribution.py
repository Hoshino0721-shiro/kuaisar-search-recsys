"""Controlled route-feature attribution and frozen-pool failure diagnostics.

Retrain masked inputs; do not mistake inference masking for feature ablation.
Earlier holdouts refit normalization and models using only preceding samples.
No unexposed candidate receives an exposure label. No final test is read.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.ctr_features import CACHE, DATA, FeatureState, connection
from src.itemcf_index import DAY_MS, ItemCFIndex
from src.v2.cascade import observed_positive_metrics
from src.v2.models import FineDCN
from src.v2.recall_fusion import select_recall
from src.v2.temporal_route_data import write_json
from src.v2.temporal_route_evaluate import cascade_lists, paired_intervals
from src.v2.temporal_route_features import WIDTH, recent_order, route_values
from src.v2.temporal_route_train import verify_cache
from src.v2.train_rankers import probability_metrics, sha


def feature_mask(columns):
    if len(set(columns)) != len(columns) or any(c not in range(28, 34) for c in columns):
        raise ValueError('Expected distinct route columns')
    mask = np.zeros(WIDTH, dtype=np.float32)
    mask[:20] = 1
    mask[columns] = 1
    return mask


def normalize(x, mean, std, columns):
    return ((np.asarray(x, dtype=np.float32)-mean)/std)*feature_mask(columns)


class Scorer:
    def __init__(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.meta = checkpoint['metadata']
        self.mean = np.array(checkpoint['mean'], dtype=np.float32)
        self.std = np.array(checkpoint['std'], dtype=np.float32)
        self.model = FineDCN(WIDTH)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def predict(self, x):
        result = []
        with torch.inference_mode():
            for lo in range(0, len(x), 65536):
                z = normalize(x[lo:lo+65536], self.mean, self.std, self.meta['columns'])
                result.append(torch.sigmoid(self.model(torch.from_numpy(z))).numpy())
        return np.concatenate(result) if result else np.empty(0, dtype=np.float32)


def fit(x, y, xv, yv, mean, std, columns, seed, cfg, destination, context):
    destination.mkdir(parents=True, exist_ok=False)
    tx = torch.from_numpy(normalize(x, mean, std, columns))
    ty = torch.from_numpy(np.asarray(y, dtype=np.float32))
    torch.manual_seed(seed)
    model = FineDCN(WIDTH)
    initial = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    order_hash = hashlib.sha256()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history = []
    started = perf_counter()
    for epoch in range(cfg['epochs']):
        model.train()
        order = torch.randperm(len(tx))
        order_hash.update(order.numpy().tobytes())
        total = 0.
        for lo in range(0, len(tx), cfg['batch_size']):
            ix = order[lo:lo+cfg['batch_size']]
            loss = nn.functional.binary_cross_entropy_with_logits(model(tx[ix]), ty[ix])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(ix)
        history.append({'epoch': epoch+1, 'train_logloss': total/len(tx)})
    meta = {'kind': 'route_attribution_v1', 'seed': seed, 'columns': columns,
            'initial_parameters_sha256': initial, 'training_order_sha256': order_hash.hexdigest(),
            'checkpoint_rule': cfg['checkpoint_rule'], 'train_rows': len(y), 'context': context,
            'parameters': sum(p.numel() for p in model.parameters()), 'test_used': False}
    torch.save({'state_dict': model.state_dict(), 'mean': mean.tolist(), 'std': std.tolist(),
                'metadata': meta}, destination/'model.pt')
    scorer = Scorer(destination/'model.pt')
    p = scorer.predict(xv)
    with torch.inference_mode():
        model.eval()
        expected = torch.sigmoid(model(torch.from_numpy(normalize(xv[:256], mean, std, columns)))).numpy()
    np.testing.assert_allclose(p[:256], expected, rtol=1e-6, atol=1e-7)
    result = {'metadata': meta, 'history': history, 'exposure': probability_metrics(yv, p),
              'checkpoint_sha256': sha(destination/'model.pt'), 'elapsed_seconds': perf_counter()-started,
              'reload_check': True}
    write_json(destination/'metrics.json', result)
    print(f'Trained {destination.name}: AUC {result["exposure"]["auc"]:.6f}', flush=True)
    return scorer, result


def sorted_positions(ids, scores):
    return np.lexsort((np.asarray(ids), -np.asarray(scores)))


def metric_arrays(items, scores, positives):
    hits = np.zeros(len(items), dtype=np.int64)
    ndcg = np.full(len(items), np.nan)
    for j, (ids, p, labels) in enumerate(zip(items, scores, positives)):
        order = sorted_positions(ids, p)[:20]
        q = observed_positive_metrics(np.asarray(ids)[order].tolist(), labels)
        hits[j] = q['hit_events']
        if q['user_period_ndcg_at_20'] is not None:
            ndcg[j] = q['user_period_ndcg_at_20']
    return hits, ndcg


def summary(hits, ndcg, positive_events):
    return {'hit_events': int(np.sum(hits)), 'ndcg_at20': float(np.nanmean(ndcg)),
            'event_recall_at20': float(np.sum(hits)/positive_events)}


def retention_histogram(ranks, weights):
    ranks = np.asarray(ranks)
    weights = np.asarray(weights)
    return {name: int(weights[(ranks >= lo) & (ranks <= hi)].sum())
            for name, lo, hi in [('1_20', 1, 20), ('21_80', 21, 80),
                                 ('81_400', 81, 400), ('401_2000', 401, 2000)]}


def limited_orders(ids, scores, heads=(40, 100), reserve=10):
    ids = np.asarray(ids)
    result = {f'rrf_head{k}_dcn': ids[sorted_positions(ids[:k], scores[:k])[:20]].tolist() for k in heads}
    chosen = ids[:reserve].tolist()
    seen = set(chosen)
    for j in sorted_positions(ids, scores):
        if int(ids[j]) not in seen:
            chosen.append(int(ids[j])); seen.add(int(ids[j]))
        if len(chosen) == 20:
            break
    result[f'reserve_rrf{reserve}_then_dcn'] = chosen
    return result


def load_labels(users, items, split='valid', cutoff=None):
    db = connection()
    try:
        if cutoff is None:
            if split != 'valid':
                raise ValueError('Only validation or explicit train-day holdouts')
            rows = db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'")
        else:
            rows = db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<?", (cutoff, cutoff+DAY_MS))
        positives = defaultdict(Counter); exposures = defaultdict(Counter); actual_users = set()
        for u, i, y in rows:
            actual_users.add(u); exposures[u][i] += 1
            if y:
                positives[u][i] += 1
    finally:
        db.close()
    if list(users) != sorted(actual_users):
        raise ValueError('User population differs')
    labels = [positives[int(u)] for u in users]
    n = np.zeros(items.shape, dtype=np.int32); y = np.zeros_like(n)
    for j, uid in enumerate(users):
        n[j] = [exposures[int(uid)].get(int(i), 0) for i in items[j]]
        y[j] = [positives[int(uid)].get(int(i), 0) for i in items[j]]
    return labels, np.array([sum(p.values()) for p in labels]), n, y


def diagnose(cache, previous, output, cfg):
    """Use original scores; factual rank changes, not causal feature attributions."""
    items = np.load(cache/'pool_items.npy', mmap_mode='r')
    users = np.load(cache/'pool_users.npy')
    x = np.load(cache/'pool_X.npy', mmap_mode='r')
    labels, positive_counts, n, y = load_labels(users, items)
    np.save(output/'pool_exposure_counts.npy', n); np.save(output/'pool_click_counts.npy', y)
    report = {'runs': {}, 'test_used': False, 'scope': 'descriptive logged-feedback diagnosis, not counterfactual CTR'}
    total = int(positive_counts.sum())
    rrf_hits = int(y[:, :20].sum())
    if total != 354626 or rrf_hits != 1745:
        raise ValueError('Published labels failed to reproduce')
    coarse = np.load(cache/'coarse_positions.npy')
    coarse_retained = int(sum(y[j, np.intersect1d(np.arange(20), coarse[j])].sum() for j in range(len(users))))
    report['rrf_top20_positive_events'] = rrf_hits
    report['rrf_top20_positive_events_retained_by_lr400'] = coarse_retained
    report['rrf_top20_lost_before_fine'] = rrf_hits-coarse_retained
    for arm in ('base', 'R'):
        for seed in cfg['seeds']:
            name = f'{arm}_s{seed}'
            p = np.load(previous/(name+'_scores.npy'), mmap_mode='r')
            ranks = np.argsort(np.argsort(-p, axis=1, kind='stable'), axis=1, kind='stable')+1
            # Scores use item-ID tie breaking, not input-order tie breaking.
            for j in range(len(users)):
                ranks[j, sorted_positions(items[j], p[j])] = np.arange(1, items.shape[1]+1)
            hist = retention_histogram(ranks[:, :20].ravel(), y[:, :20].ravel())
            retained = int(y[:, :20][ranks[:, :20] <= 20].sum())
            selected = ranks <= 20
            selected_n = int(n[selected].sum()); selected_y = int(y[selected].sum())
            row = {'rrf_positive_event_destination': hist, 'rrf_hits_retained': retained,
                   'rrf_hits_lost': rrf_hits-retained, 'new_hits_outside_rrf_top20': selected_y-retained,
                   'direct_hits': selected_y,
                   'top20_logged_pair_fraction': float((n[selected] > 0).mean()),
                   'top20_logged_exposure_rows': selected_n,
                   'top20_observed_ctr_on_logged_subset': selected_y/selected_n if selected_n else None,
                   'top20_mean_predicted_ctr': float(p[selected].mean()),
                   'top20_median_train_exposures': float(np.median(np.expm1(x[:,:,6][selected]))),
                   'top20_mean_smoothed_item_ctr': float(x[:,:,2][selected].mean()),
                   'top20_recent_route_fraction': float(x[:,:,28][selected].mean()),
                   'top20_cf_route_fraction': float(x[:,:,29][selected].mean())}
            adjusted = defaultdict(lambda: {'hits': [], 'ndcg': []})
            for j, uid in enumerate(users):
                orders = limited_orders(items[j], p[j], cfg['limited_adjustment']['rerank_rrf_head'], cfg['limited_adjustment']['reserve_rrf'])
                for method, order in orders.items():
                    q = observed_positive_metrics(order, labels[j])
                    adjusted[method]['hits'].append(q['hit_events'])
                    adjusted[method]['ndcg'].append(q['user_period_ndcg_at_20'] if q['user_period_ndcg_at_20'] is not None else np.nan)
            row['limited_adjustment'] = {k: summary(v['hits'], v['ndcg'], total) for k,v in adjusted.items()}
            report['runs'][name] = row
    rrf_n = int(n[:,:20].sum())
    report['rrf_selection'] = {'top20_logged_pair_fraction': float((n[:,:20]>0).mean()),
        'top20_logged_exposure_rows': rrf_n, 'top20_observed_ctr_on_logged_subset': rrf_hits/rrf_n,
        'top20_median_train_exposures': float(np.median(np.expm1(x[:,:20,6]))),
        'top20_mean_smoothed_item_ctr': float(x[:,:20,2].mean())}
    write_json(output/'diagnosis.json', report)
    return labels, positive_counts


def ablate(cache, previous_models, previous_evaluation, output, cfg, labels, positive_counts):
    x = np.load(cache/'train_X.npy', mmap_mode='r'); y = np.load(cache/'train_y.npy')
    xv = np.load(cache/'valid_X.npy', mmap_mode='r'); yv = np.load(CACHE/'valid_y.npy')
    mean = np.load(cache/'mean.npy'); std = np.load(cache/'std.npy')
    items = np.load(cache/'pool_items.npy', mmap_mode='r'); pool_x = np.load(cache/'pool_X.npy', mmap_mode='r')
    coarse = np.load(cache/'coarse_positions.npy'); state = FeatureState.load(CACHE/'state.npz')
    source_training = json.loads((previous_models/'metrics.json').read_text(encoding='utf-8'))
    arrays = {}; runs = {}
    for arm, columns in cfg['route_columns'].items():
        for seed in cfg['seeds']:
            name = f'{arm}_s{seed}'
            scorer, train = fit(x,y,xv,yv,mean,std,columns,seed,cfg,output/'models'/name, {'cache_metadata_sha256':sha(cache/'metadata.json')})
            original = source_training['runs'][f'R_s{seed}']['metadata']
            for key in ('initial_parameters_sha256', 'training_order_sha256'):
                if train['metadata'][key] != original[key]:
                    raise ValueError('Matched-seed training differs: '+key)
            p = scorer.predict(pool_x.reshape(-1, WIDTH)).reshape(items.shape)
            np.save(output/(name+'_scores.npy'), p)
            h,d = metric_arrays(items,p,labels)
            ch = np.zeros(len(items),dtype=np.int64); cd = np.full(len(items),np.nan)
            for j in range(len(items)):
                _,_,order = cascade_lists(items[j],p[j],coarse[j],state.categories[items[j]],source_training['protocol']['config']['budget'])
                q = observed_positive_metrics(order,labels[j]); ch[j] = q['hit_events']
                if q['user_period_ndcg_at_20'] is not None:
                    cd[j] = q['user_period_ndcg_at_20']
            np.savez_compressed(output/(name+'_per_user.npz'), hits=h,ndcg=d,cascade_hits=ch,cascade_ndcg=cd)
            arrays[name] = (h,d,ch,cd)
            runs[name] = {'training':train,'direct':summary(h,d,int(positive_counts.sum())),
                          'cascade':summary(ch,cd,int(positive_counts.sum()))}
            write_json(output/'ablation_progress.json',runs)
            print(f'Evaluated {name}: direct {h.sum()}, cascade {ch.sum()}',flush=True)
    for arm in ('base','R'):
        for seed in cfg['seeds']:
            name = f'{arm}_s{seed}'
            a = np.load(previous_evaluation/(name+'_per_user.npz'))
            arrays[name] = tuple(a[k] for k in ('direct_hits','direct_ndcg','cascade_hits','cascade_ndcg'))
            runs[name] = {'reused':True,'direct':summary(arrays[name][0],arrays[name][1],int(positive_counts.sum())),
                          'cascade':summary(arrays[name][2],arrays[name][3],int(positive_counts.sum()))}
    means = {arm: [np.mean([arrays[f'{arm}_s{s}'][j] for s in cfg['seeds']],axis=0) for j in range(4)]
             for arm in list(cfg['route_columns'])+['base','R']}
    aggregate = {arm:{'direct':summary(v[0],v[1],int(positive_counts.sum())),
                     'cascade':summary(v[2],v[3],int(positive_counts.sum()))} for arm,v in means.items()}
    # summary(int hits) is for per-run integers. Preserve fractional seed means here.
    for arm,v in means.items():
        aggregate[arm]['direct']['hit_events'] = float(v[0].sum())
        aggregate[arm]['cascade']['hit_events'] = float(v[2].sum())
    contrasts = {}
    for arm in cfg['route_columns']:
        contrasts[arm+'-base'] = paired_intervals(means[arm][1]-means['base'][1],means[arm][0]-means['base'][0],positive_counts)
    for a,b in [('R','recent_only'),('R','cf_only'),('R','without_combined'),('ranks_only','flags_only')]:
        contrasts[a+'-'+b] = paired_intervals(means[a][1]-means[b][1],means[a][0]-means[b][0],positive_counts)
    result = {'runs':runs,'aggregate':aggregate,'paired_direct':contrasts,'test_used':False,
              'scope':cfg['scope'],'new_model_runs':len(cfg['route_columns'])*len(cfg['seeds'])}
    write_json(output/'ablation.json',result)
    return result


def prefix_state(cutoff):
    """Reconstruct original numeric state with strictly earlier complete days."""
    final = FeatureState.load(CACHE/'state.npz')
    state = FeatureState(final.categories, len(final.up))
    db = connection()
    first = db.execute("SELECT min(timestamp_ms) FROM clean_events WHERE scene='rec' AND split='train'").fetchone()[0]
    day = ((first+8*3600000)//DAY_MS)*DAY_MS-8*3600000
    last_clicks = np.zeros(len(state.ip))
    try:
        while day < cutoff:
            rows = np.asarray(db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<? ORDER BY id",(day,day+DAY_MS)).fetchall(),dtype=np.int64)
            state.update(rows[:,0],rows[:,1],rows[:,2])
            last_clicks = np.bincount(rows[:,1],weights=rows[:,2],minlength=len(state.ip))
            rows = np.asarray(db.execute("SELECT user_id,item_id FROM clean_events WHERE scene='search' AND split='train' AND label=1 AND timestamp_ms>=? AND timestamp_ms<? ORDER BY id",(day,day+DAY_MS)).fetchall(),dtype=np.int64).reshape(-1,2)
            state.update_search(rows[:,0],rows[:,1])
            day += DAY_MS
    finally:
        db.close()
    return state,last_clicks


def rolling(cache, output, cfg):
    x = np.load(cache/'train_X.npy',mmap_mode='r'); y = np.load(cache/'train_y.npy')
    days = np.load(cache/'train_days.npy'); meta = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    sample = np.load(cache/'sample_indices.npy')
    tu = np.load(CACHE/'train_users.npy',mmap_mode='r'); ti = np.load(CACHE/'train_items.npy',mmap_mode='r')
    state_final = FeatureState.load(CACHE/'state.npz')
    windows = {}
    for cutoff in cfg['rolling_cutoffs_ms']:
        dest = output/str(cutoff); dest.mkdir()
        train = days<cutoff; valid = days==cutoff
        if days[train].max() >= cutoff or not valid.any():
            raise ValueError('Invalid temporal holdout')
        xt = np.asarray(x[train]); xv = np.asarray(x[valid]); yt = y[train]; yv = y[valid]
        mean = xt.mean(axis=0,dtype=np.float64).astype(np.float32)
        std = np.maximum(xt.std(axis=0,dtype=np.float64),1e-6).astype(np.float32)
        state,last_clicks = prefix_state(cutoff)
        selected = sample[valid]
        # Reproduce all held-out numeric feature columns against independent state.
        db = connection()
        try:
            start = db.execute("SELECT count(*) FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms<?",(cutoff,)).fetchone()[0]
            times = np.fromiter((r[0] for r in db.execute("SELECT timestamp_ms FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<? ORDER BY id",(cutoff,cutoff+DAY_MS))),dtype=np.int64)
            users = np.fromiter((r[0] for r in db.execute("SELECT DISTINCT user_id FROM clean_events WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<? ORDER BY user_id",(cutoff,cutoff+DAY_MS))),dtype=np.int64)
        finally:
            db.close()
        expected = state.features(tu[selected],ti[selected],times[selected-start])
        max_error = float(np.abs(expected-xv[:,:20]).max())
        if max_error != 0:
            raise ValueError('Holdout prefix numeric features differ')
        np.savez_compressed(dest/'normalization.npz',mean=mean,std=std)
        models = {}; training = {}
        for arm,columns in cfg['rolling_arms'].items():
            for seed in cfg['seeds']:
                name = f'{arm}_s{seed}'
                scorer,record = fit(xt,yt,xv,yv,mean,std,columns,seed,cfg,dest/'models'/name,
                    {'cutoff_ms':cutoff,'train_max_day_ms':int(days[train].max()),'validation_rows_are_original_uniform_sample':True})
                models[name] = scorer; training[name] = record
        for seed in cfg['seeds']:
            for key in ('initial_parameters_sha256','training_order_sha256'):
                if training[f'base_s{seed}']['metadata'][key] != training[f'R_s{seed}']['metadata'][key]:
                    raise ValueError('Rolling paired seed mismatch')
        index_record = next(d for d in meta['days'] if d['cutoff_ms']==cutoff)
        index = ItemCFIndex(index_record['index_path'], meta['manifest_sha256'])
        if index.metadata['cutoff_ms'] != cutoff or np.any(index.seed_times>=cutoff):
            raise ValueError('Future graph or seeds')
        expected_ids = np.flatnonzero(state.ip>0)
        np.testing.assert_array_equal(index.items,expected_ids)
        recent = recent_order(last_clicks,2000)
        pool_items = np.lib.format.open_memmap(dest/'pool_items.npy',mode='w+',dtype=np.int32,shape=(len(users),2000))
        scores = {name:np.lib.format.open_memmap(dest/(name+'_scores.npy'),mode='w+',dtype=np.float32,shape=pool_items.shape) for name in models}
        np.save(dest/'users.npy',users)
        try:
            for lo in range(0,len(users),64):
                chunks = []
                for j in range(lo,min(lo+64,len(users))):
                    uid = int(users[j]); cf = index.query(uid,1000)
                    ranks = {(int(i),'recent'):r for r,i in enumerate(recent,1)}
                    ranks.update({(int(i),'itemcf'):r for r,i in enumerate(cf['items'],1)})
                    ids = select_recall(ranks,2000,{'recent':1.,'itemcf':1.})
                    if len(ids)!=2000:
                        raise ValueError('Short temporal pool')
                    pool_items[j] = ids
                    features = np.zeros((len(ids),WIDTH),dtype=np.float32)
                    features[:,:20] = state.features(np.full(len(ids),uid),np.asarray(ids),np.full(len(ids),cutoff))
                    features[:,28:] = route_values(ids,recent,cf['items'],cf['scores'])
                    chunks.append(features)
                batch = np.concatenate(chunks)
                for name,scorer in models.items():
                    scores[name][lo:lo+len(chunks)] = scorer.predict(batch).reshape(len(chunks),2000)
                if lo%1024==0:
                    print(f'Rolling {cutoff} pool {lo}/{len(users)}',flush=True)
        finally:
            index.close()
        pool_items.flush()
        for p in scores.values():
            p.flush()
        labels,positive,n,y_pool = load_labels(users,pool_items,cutoff=cutoff)
        rrf_h,rrf_d = metric_arrays(pool_items,np.broadcast_to(-np.arange(2000),pool_items.shape),labels)
        runs = {}; arrays = {}
        for name,p in scores.items():
            h,d = metric_arrays(pool_items,p,labels); arrays[name] = (h,d)
            np.savez_compressed(dest/(name+'_per_user.npz'),hits=h,ndcg=d,positive_events=positive)
            runs[name] = {'training':training[name],'direct':summary(h,d,int(positive.sum()))}
        means = {arm:[np.mean([arrays[f'{arm}_s{s}'][j] for s in cfg['seeds']],axis=0) for j in range(2)] for arm in cfg['rolling_arms']}
        aggregate = {}
        for arm,(h,d) in means.items():
            aggregate[arm] = summary(h,d,int(positive.sum())); aggregate[arm]['hit_events'] = float(h.sum())
        result = {'cutoff_ms':cutoff,'users':len(users),'positive_events':int(positive.sum()),
                  'pool_hits':int(y_pool.sum()),'train_rows':int(train.sum()),'sampled_validation_exposures':int(valid.sum()),
                  'numeric_feature_replay_max_error':max_error,'train_max_day_ms':int(days[train].max()),
                  'catalog_matches_strict_prefix':True,'candidate_items_sha256':sha(dest/'pool_items.npy'),
                  'runs':runs,'aggregate':aggregate,'rrf':summary(rrf_h,rrf_d,int(positive.sum())),
                  'paired_R_minus_base':paired_intervals(means['R'][1]-means['base'][1],means['R'][0]-means['base'][0],positive),
                  'test_used':False,'scope':'earlier dates from original train, refitted strict prefix; retrospective temporal check, not untouched independent test'}
        write_json(dest/'metrics.json',result); windows[str(cutoff)] = result
        write_json(output/'rolling.json',windows)
        print(f'Rolling {cutoff}: {aggregate}',flush=True)
    return windows


def run(cache, previous_models, previous_evaluation, output, config):
    if output.exists():
        raise FileExistsError('Choose a new output root')
    meta = verify_cache(cache)
    cfg = json.loads(config.read_text(encoding='utf-8'))
    torch.set_num_threads(cfg['threads'])
    output.mkdir(parents=True)
    protocol = {'config':cfg,'config_sha256':sha(config),'source_sha256':sha(__file__),
                'cache_metadata_sha256':sha(cache/'metadata.json'),'candidate_sha256':meta['candidate_sha256'],
                'previous_training_sha256':sha(previous_models/'metrics.json'),
                'previous_evaluation_sha256':sha(previous_evaluation/'metrics.json'),
                'test_used':False,'default_changed':False}
    write_json(output/'protocol.json',protocol)
    labels,positive = diagnose(cache,previous_evaluation,output,cfg)
    ablation = ablate(cache,previous_models,previous_evaluation,output,cfg,labels,positive)
    windows = rolling(cache,output,cfg)
    if sha(__file__)!=protocol['source_sha256']:
        raise ValueError('Implementation changed during execution')
    result = {'protocol':protocol,'ablation':ablation,'rolling':windows,
              'new_model_runs':len(cfg['seeds'])*(len(cfg['route_columns'])+len(cfg['rolling_arms'])*len(cfg['rolling_cutoffs_ms'])),
              'test_used':False,'default_changed':False,'status':'complete'}
    write_json(output/'metrics.json',result)
    print('Attribution and retrospective checks complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=Path('artifacts/temporal_route_20261022/cache'))
    p.add_argument('--previous-models',type=Path,default=Path('artifacts/temporal_route_20261022/models'))
    p.add_argument('--previous-evaluation',type=Path,default=Path('artifacts/temporal_route_20261022/evaluation'))
    p.add_argument('--config',type=Path,default=Path('configs/route_attribution.json'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.cache,a.previous_models,a.previous_evaluation,a.out,a.config)
