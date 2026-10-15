"""Validation-only recommendation sanity baselines; SQLite is read-only."""
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from src.metrics import auc, gauc, logloss, brier


def evaluate(labels, scores, users):
    return dict(auc=auc(labels,scores),gauc=gauc(labels,scores,users),
                logloss=logloss(labels,scores),brier=brier(labels,scores))


def main():
    root=Path('data/processed/kuaisar_small_v1')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='ready'
    db=sqlite3.connect((root/'dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA cache_size=-131072')
    stats={}
    for iid,n,pos in db.execute("SELECT item_id,count(*),sum(label) FROM clean_events WHERE scene='rec' AND split='train' GROUP BY item_id"):
        stats[iid]=(n,pos)
    n=sum(v[0] for v in stats.values()); pos=sum(v[1] for v in stats.values())
    prior=pos/n
    print('Train statistics fitted',n,flush=True)
    popular=sorted((i for i,v in stats.items() if v[1]>0),key=lambda i:(-stats[i][1],i))[:100]
    labels=[]; scores=[]; users=[]; cold=[]; hits=Counter(); positive_total=0; warm_positive=0
    for uid,iid,y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid' ORDER BY id"):
        ni,pi=stats.get(iid,(0,0))
        labels.append(y); scores.append((pi+20*prior)/(ni+20)); users.append(uid); cold.append(ni==0)
        if y:
            positive_total+=1; warm_positive+=ni>0
            for k in (50,100): hits[k]+=iid in popular[:k]
    constants=[prior]*len(labels)
    report={'protocol':{'scene':'rec','evaluation_split':'valid','fit_split':'train',
                        'alpha':20,'alpha_selection':'fixed before evaluation; no sweep',
                        'retrieval':'global train positive-count top-K; ties item_id ascending; no history filtering',
                        'recall_denominator':'positive validation events, including repeated user-item events and unseen items',
                        'gauc':'user grouped, weighted by row count, single-class groups excluded',
                        'test_used':False},
            'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'train_rows':n,'train_item_count':len(stats),'train_positive_rate':prior,
            'valid_rows':len(labels),'valid_positive_rate':sum(labels)/len(labels),
            'constant_ctr':evaluate(labels,constants,users),
            'smoothed_item_ctr':evaluate(labels,scores,users),
            'recall':{str(k):{'hits':hits[k],'positive_events':positive_total,'recall':hits[k]/positive_total,
                             'warm_positive_events':warm_positive,'warm_recall':hits[k]/warm_positive} for k in (50,100)},
            'train_catalog_positive_event_ceiling':warm_positive/positive_total,
            'cold_item_rows':sum(cold),'groups':{}}
    for name,flag in [('cold',True),('warm',False)]:
        indices=[i for i,x in enumerate(cold) if x==flag]
        report['groups'][name]={'rows':len(indices),**evaluate([labels[i] for i in indices],[scores[i] for i in indices],[users[i] for i in indices])}
    model={'prior':prior,'alpha':20,'popular_top100':popular}
    out=Path('experiments/r0_p0_valid');out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'model_summary.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    db.close()
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
