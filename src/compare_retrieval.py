"""Paired user bootstrap on saved validation candidates; no training."""
import json
import sqlite3
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np


def main():
    runs=['r2_two_tower_valid','r2_history_only_valid','r2_history_attributes_valid']
    db=sqlite3.connect(Path('data/processed/kuaisar_small_v1/dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    positive=defaultdict(Counter);users=set()
    for u,i,y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
        users.add(u)
        if y:positive[u][i]+=1
    db.close();ordered=sorted(users);n=len(ordered)
    denominators=np.array([sum(positive[u].values()) for u in ordered])
    results={};per_user={};fingerprints=set()
    for run in runs:
        root=Path('experiments')/run
        m=json.loads((root/'metrics.json').read_text())
        fingerprints.add(m['config']['manifest_sha256'])
        candidates={}
        for line in (root/'candidates.jsonl').open():
            row=json.loads(line);assert row['user_id'] not in candidates
            candidates[row['user_id']]=set(row['items'][:100])
        assert set(candidates)==users
        values=np.array([sum(v for i,v in positive[u].items() if i in candidates[u]) for u in ordered])
        assert int(values.sum())==m['metrics']['100']['hits']
        per_user[run]=values
        results[run]={'hits_at_100':int(values.sum()),'recall_at_100':float(values.sum()/denominators.sum())}
    assert len(fingerprints)==1
    rng=np.random.default_rng(2026);comparisons=[]
    for base,new in [(runs[0],runs[2]),(runs[1],runs[2])]:
        delta=per_user[new]-per_user[base];samples=[]
        for _ in range(20):
            indices=rng.integers(0,n,size=(100,n))
            samples.extend((delta[indices].sum(axis=1)/denominators[indices].sum(axis=1)).tolist())
        comparisons.append({'base':base,'new':new,'delta_recall':float(delta.sum()/denominators.sum()),
                            'paired_user_bootstrap_95_interval':np.percentile(samples,[2.5,97.5]).tolist()})
    result={'split':'rec valid','test_used':False,'positive_events':int(denominators.sum()),'runs':results,
            'comparisons':comparisons,'bootstrap':{'unit':'user','replicates':2000,'seed':2026,
            'limitations':'exploratory single-seed validation comparison, not independent test significance; no multiplicity correction'}}
    out=Path('experiments/r2_comparison');out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
