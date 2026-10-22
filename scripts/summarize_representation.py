"""Aggregate only audited safe representation experiments; no seed selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.v2.ranking_convergence import write_json
from src.v2.route_attribution import load_labels
from src.v2.temporal_route_evaluate import paired_intervals
from src.v2.train_rankers import sha


def summarize(root,reference,baseline,output):
    if output.exists():raise FileExistsError('Preserve summary')
    audit=json.loads((root/'model_audit.json').read_text())
    if audit['status']!='passed':raise ValueError('Completed model audit required')
    targets=json.loads((root/'evaluation_target_safe/metrics.json').read_text())
    interests=json.loads((root/'evaluation_interest_safe/metrics.json').read_text())
    old=json.loads((baseline/'metrics.json').read_text())
    records={**targets['runs'],**interests['runs']};seeds=[2026,2027,2028]
    users=np.load(reference/'pool_users.npy');items=np.load(reference/'pool_items.npy',mmap_mode='r')
    _,positive,_,_=load_labels(users,items)
    aggregate={'numeric':old['aggregate']['R']};arrays={}
    for group in ['numeric','id_attrs','content','mean','din']:
        prefix='full_R' if group=='numeric' else group
        evaluation=baseline if group=='numeric' else root/('evaluation_target_safe' if group in ('id_attrs','content') else 'evaluation_interest_safe')
        values=[]
        for seed in seeds:
            with np.load(evaluation/f'{prefix}_s{seed}_per_user.npz') as a:
                values.append([a[k] for k in ('direct_hits','direct_ndcg','cascade_hits','cascade_ndcg')])
        arrays[group]=[np.mean([a[j] for a in values],axis=0) for j in range(4)]
        if group=='numeric':continue
        runs=[records[f'{group}_s{s}'] for s in seeds]
        aggregate[group]={'selected_epochs':[r['training']['selected_epoch'] for r in runs],
            'stopped_epochs':[r['training']['epochs_completed'] for r in runs],
            'validation':{k:float(np.mean([r['training']['selected_validation'][k] for r in runs])) for k in ('auc','gauc','logloss','brier')},
            'timestamp_ndcg':float(np.mean([r['training']['selected_validation']['same_timestamp']['ndcg_at20'] for r in runs])),
            'direct_hits':float(np.mean([r['direct']['hit_events'] for r in runs])),
            'direct_ndcg':float(np.mean([r['direct']['ndcg_at20'] for r in runs])),
            'cascade_hits':float(np.mean([r['cascade']['hit_events'] for r in runs])),
            'seed_direct_hits':[r['direct']['hit_events'] for r in runs],
            'seed_cascade_hits':[r['cascade']['hit_events'] for r in runs]}
    comparisons={}
    for a,b in [('id_attrs','numeric'),('content','id_attrs'),('content','numeric'),('mean','content'),('din','mean')]:
        x,y=arrays[a],arrays[b]
        comparisons[a+'-'+b]=paired_intervals(x[1]-y[1],x[0]-y[0],positive)
    result={'aggregate':aggregate,'comparisons':comparisons,'audit_sha256':sha(root/'model_audit.json'),
            'target_eval_sha256':sha(root/'evaluation_target_safe/metrics.json'),
            'interest_eval_sha256':sha(root/'evaluation_interest_safe/metrics.json'),
            'source_sha256':sha(__file__),'test_used':False,'default_changed':False,
            'scope':'reused development day; numeric baseline has different architecture/LR protocol, not pure representation-only causal attribution; content vs ID and mean vs attention are matched comparisons'}
    write_json(output,result);items._mmap.close()
    print(json.dumps(result,indent=2),flush=True)
    # Standalone scientific curves: epoch tables are primary; no external assets.
    from scripts.plot_ranking_convergence import chart
    parts=['<!doctype html><meta charset="utf-8"><title>Learned representation experiments</title>',
           '<style>body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:20px;background:#f5f6f8}'
           '.plots{display:grid;grid-template-columns:1fr 1fr;gap:12px}svg{width:100%}table{border-collapse:collapse;background:white}'
           'td,th{padding:7px;border:1px solid #ddd}h2{margin-top:35px}</style>',
           '<h1>Safe learned representations (v2)</h1><p>Validation LogLoss selects epochs. '
           'PAD/UNK and actual-input coverage enforced. Interest stage uses frozen content vectors. '
           'Reused development validation, not an independent final test.</p>']
    for name,r in records.items():
        t=r['training'];h=[row for row in t['history'] if row['epoch']>0]
        parts.append(f'<h2>{name}: best epoch {t["selected_epoch"]}</h2><div class="plots">')
        parts.append(chart(h,'valid.logloss','Validation LogLoss'))
        parts.append(chart(h,'train_probe_logloss','Fixed training probe LogLoss'))
        parts.append('</div>')
        if t['history'][0]['epoch']==0:parts.append(f'<p>Frozen content epoch 0 LL: {t["history"][0]["valid"]["logloss"]:.7f}</p>')
        parts.append('<details><summary>All epochs</summary><table><tr><th>Epoch</th><th>LR</th><th>LL</th><th>AUC</th><th>GAUC</th></tr>')
        for row in t['history']:
            parts.append(f'<tr><td>{row["epoch"]}</td><td>{row["learning_rate"]:.8f}</td>'
                f'<td>{row["valid"]["logloss"]:.7f}</td><td>{row["valid"]["auc"]:.7f}</td><td>{row["valid"]["gauc"]:.7f}</td></tr>')
        parts.append('</table></details>')
    (root/'curves_safe.html').write_text('\n'.join(parts),encoding='utf-8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/representation_20261022'))
    p.add_argument('--reference',type=Path,default=Path('artifacts/full_exposure_local_20261022/reference'))
    p.add_argument('--baseline',type=Path,default=Path('artifacts/ranking_convergence_20261022'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();summarize(a.root,a.reference,a.baseline,a.out)
