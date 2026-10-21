"""Recount allocation reports and replay selected policies independently."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.itemcf_index import DATA, ItemCFIndex, digest
from src.v2.recall_allocation import load_population


def reference(lists, spec, budget):
    supplied = {s: ids[:spec['route_limits'][s]] for s, ids in lists.items()}
    policy = spec.get('fusion', {})
    if policy.get('method', 'rrf') == 'rrf':
        values = defaultdict(float)
        for source, ids in supplied.items():
            w = spec.get('weights', {}).get(source, 1.)
            for rank, iid in enumerate(ids, 1):
                if w:
                    values[iid] += w/(60+rank)
        return sorted(values, key=lambda i: (-values[i],i))[:budget]
    chosen = []
    for source, quota in policy['allocations']:
        present = set(chosen)
        additions = [i for i in supplied.get(source, []) if i not in present]
        chosen += additions[:quota]
    for source in policy['fill_order']:
        present = set(chosen)
        additions = [i for i in supplied.get(source, []) if i not in present]
        chosen += additions[:max(0,budget-len(chosen))]
    return chosen[:budget]


def audit(directory, data):
    report = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    config = report['config']
    if digest(data/'manifest.json') != report['manifest_sha256']:
        raise ValueError('Data manifest changed')
    budget = config['budget']
    index = ItemCFIndex(config['itemcf_index'], report['manifest_sha256'])
    try:
        if index.metadata['version'] != report['index_version']:
            raise ValueError('ItemCF index changed')
        users, positive, popular, recent = load_population(data,index.metadata['cutoff_ms'],budget)
        with np.load(directory/'paired_hits.npz') as saved:
            np.testing.assert_array_equal(saved['users'],users)
            counts = np.array([sum(positive[u].values()) for u in users])
            np.testing.assert_array_equal(saved['positive_counts'],counts)
            masks = {'all':np.ones(len(users),dtype=bool),'selection':saved['selection'],
                     'check':~saved['selection'],'has_train_clicks':saved['warm'],
                     'no_train_clicks':~saved['warm']}
            for name, metrics in report['metrics'].items():
                for group, mask in masks.items():
                    h, n = int(saved[name][mask].sum()),int(counts[mask].sum())
                    assert h == metrics[group]['hit_events'], (name,group)
                    assert n == metrics[group]['positive_events'], (name,group)
                    assert (metrics[group]['gained_events_vs_recent']-metrics[group]['lost_events_vs_recent']
                            == h-int(saved['recent_only'][mask].sum())), (name,group)
            selected = report['selected_on_selection_users']
            names = list(dict.fromkeys(['recent_only','legacy_rrf','expanded_rrf',selected]))
            names = [name for name in names if name in config['variants']]
            for row, uid in enumerate(users):
                cf = index.query(uid,1000,neighbor_limit=config.get('itemcf_neighbor_limit'))['items']
                lists = {'popular':popular,'recent':recent,'itemcf':cf}
                for name in names:
                    chosen = reference(lists,config['variants'][name],budget)
                    assert len(chosen) == len(set(chosen)) <= budget
                    hit = sum(positive[uid].get(i,0) for i in chosen)
                    assert hit == saved[name][row], (uid,name,hit,int(saved[name][row]))
                if (row+1)%2000 == 0:
                    print(f'Audited {row+1}/{len(users)} users',flush=True)
        result = {'status':'passed','users':len(users),'policies_replayed':names,
                  'metrics_sha256':digest(directory/'metrics.json'),'source_sha256':digest(Path(__file__)),
                  'checks':['all per-user hit-vector totals and partitions match report',
                            'gained minus lost equals net hit difference for every variant/group',
                            'independent policy replay matches every user for controls and selected variant',
                            'candidate uniqueness and budget respected','data/index versions match'],
                  'test_labels_used':False}
        (directory/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
        return result
    finally:
        index.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=DATA)
    args=parser.parse_args()
    audit(args.report,args.data)
