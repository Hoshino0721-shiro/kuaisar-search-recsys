"""Aggregate a bounded study without choosing seeds or rewriting old reports."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.itemcf_index import digest
from src.v2.ranking_convergence import write_json
from scripts.plot_ranking_convergence import chart


def interval(delta, counts):
    rng=np.random.default_rng(38291)
    values=[]
    for _ in range(1000):
        ix=rng.integers(0,len(counts),len(counts))
        values.append(float(delta[ix].sum()/counts[ix].sum()))
    return np.percentile(values,[2.5,97.5]).tolist()


def summarize(root):
    spec=json.loads((root/'study.json').read_text(encoding='utf-8'))
    records={}; training={}; per_user={}
    for job in spec['jobs']:
        name=job['name']; model=Path(job['model'])
        t=json.loads((model/'metrics.json').read_text(encoding='utf-8'))
        e=json.loads((root/'evaluation'/name/'metrics.json').read_text(encoding='utf-8'))
        if t['status']!='complete' or e['status']!='complete': raise ValueError('Study not complete')
        training[name]=t
        records[name]={'window':job['window'],'arm':job['arm'],'seed':job['seed'],
            'selected_epoch':t['selected_epoch'],'stopped_epoch':len(t['history']),
            'stop_reason':t['early_stopping_reason'],'optimizer_steps':t['optimizer_steps'],
            'elapsed_training_seconds':t['elapsed_seconds'],'pure_tower':e['pure_tower'],
            'fusion':e['fusion'],'baseline':e['baseline'],'index_array_bytes':e['index_array_bytes'],
            'gained_events':e['gained_events'],'lost_events':e['lost_events'],
            'tower_positive_events_outside_base_pool':e['tower_positive_events_outside_base_pool'],
            'catalog_items':e['catalog_items'],'catalog_event_coverage':e['catalog_event_coverage'],
            'single_user_p50_ms':e['single_user_p50_ms'],'single_user_p95_ms':e['single_user_p95_ms'],
            'checkpoint_sha256':e['checkpoint_sha256'],'index_version':e['index_version'],
            'training_report_sha256':digest(model/'metrics.json'),
            'evaluation_report_sha256':digest(root/'evaluation'/name/'metrics.json')}
        with np.load(root/'evaluation'/name/'per_user.npz') as a:
            per_user[name]={k:a[k] for k in ('counts','delta_hits','delta_ndcg','users')}
    aggregate={}
    for arm in ('attributes','history'):
        names=[f'{arm}_s{s}' for s in (2026,2027,2028)]
        rows=[records[n] for n in names]
        for name in names[1:]: np.testing.assert_array_equal(per_user[names[0]]['users'],per_user[name]['users'])
        delta=np.mean([per_user[n]['delta_hits'] for n in names],axis=0)
        counts=per_user[names[0]]['counts']
        aggregate[arm]={'seeds':[2026,2027,2028],
            'selected_epochs':[r['selected_epoch'] for r in rows],
            'stopped_epochs':[r['stopped_epoch'] for r in rows],
            'pure_recall_at_1000':float(np.mean([r['pure_tower']['recall_at_1000'] for r in rows])),
            'fused_recall_at_2000':float(np.mean([r['fusion']['all']['recall_at_2000'] for r in rows])),
            'fused_hits_at_2000':float(np.mean([r['fusion']['all']['hit_events_at_2000'] for r in rows])),
            'fused_hits_at_20':float(np.mean([r['fusion']['all']['hit_events_at_20'] for r in rows])),
            'fused_period_ndcg_at_20':float(np.mean([r['fusion']['all']['period_ndcg_at_20'] for r in rows])),
            'seed_delta_hits_at_2000':[r['fusion']['all']['net_hit_events_at_2000'] for r in rows],
            'seed_delta_hits_at_20':[r['fusion']['all']['net_hit_events_at_20'] for r in rows],
            'exploratory_user_bootstrap_recall_delta_95':interval(delta[:,0],counts)}
    primary=[records[f'history_s{s}'] for s in (2026,2027,2028)]+[records['earlier_history_s2026']]
    quality_pass=all(r['fusion']['all']['net_hit_events_at_2000']>0 and
                     r['fusion']['all']['net_hit_events_at_20']>=0 for r in primary)
    plateau_pass=all(r['stop_reason']=='validation_plateau' for r in primary)
    audit=json.loads((root/'audit.json').read_text(encoding='utf-8')) if (root/'audit.json').exists() else None
    result={'status':'audited_complete' if audit and audit['status']=='passed' else 'complete_pending_audit',
            'records':records,'aggregate':aggregate,'quality_adoption_gate_passed':quality_pass,
            'primary_plateau_gate_passed':plateau_pass,'default_changed':False,
            'decision':'eligible_for_manual_adoption_review' if quality_pass and plateau_pass else 'retain_existing_rrf',
            'audit_sha256':digest(root/'audit.json') if audit else None,
            'study_sha256':digest(root/'study.json'),'source_sha256':digest(Path(__file__)),
            'test_used':False,'scope':'reused development day and one retrospective day; no blind final test or online improvement',
            'bootstrap_scope':'seed-averaged paired users, reused development day; exploratory, no multiple-comparison correction'}
    write_json(root/'summary.json',result)
    parts=['<!doctype html><meta charset="utf-8"><title>Final dual-tower study</title>',
        '<style>body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:20px;background:#f5f6f8}'
        '.plots{display:grid;grid-template-columns:1fr 1fr;gap:12px}svg{width:100%}table{border-collapse:collapse}'
        'td,th{padding:7px;border:1px solid #ccc}</style><h1>Final dual-tower study</h1>',
        '<p>Best epoch: fixed 256 development users, exact full-catalog Recall@1000. '
        'All-user results are development / retrospective evidence, not a blind test. '
        'Each panel has its own scale. Training contrast loss is not CTR LogLoss.</p>']
    for name,t in training.items():
        parts.append(f'<h2>{name}: best {t["selected_epoch"]}, stop {len(t["history"])}</h2><div class="plots">')
        parts.append(chart(t['history'],'validation.event_recall','256-user full-catalog Recall@1000'))
        parts.append(chart(t['history'],'probe_contrast_loss','Fixed training probe contrast loss'))
        parts.append('</div><details><summary>Epoch data</summary><table><tr><th>Epoch</th><th>LR</th>'
                     '<th>Train loss</th><th>Probe loss</th><th>Valid Recall</th></tr>')
        for h in t['history']:
            parts.append(f'<tr><td>{h["epoch"]}</td><td>{h["learning_rates"][0]:.7f}</td>'
                f'<td>{h["train_contrast_loss"]:.6f}</td><td>{h["probe_contrast_loss"]:.6f}</td>'
                f'<td>{h["validation"]["event_recall"]:.6f}</td></tr>')
        parts.append('</table></details>')
    (root/'curves.html').write_text('\n'.join(parts),encoding='utf-8',newline='\n')
    print(json.dumps({'status':result['status'],'aggregate':aggregate,'decision':result['decision']},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('artifacts/tower_closure_20261022'))
    summarize(p.parse_args().root)
