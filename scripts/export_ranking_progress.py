"""Publish aggregate JSON evidence only; never copy arrays, logs or weights."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def export(output):
    if output.exists():raise FileExistsError('Preserve an existing published snapshot')
    # Explicit allowlist: these reports contain only aggregate metrics/provenance.
    sources={
        'ranking_root_cause.json':'artifacts/ranking_v3/root_cause_controls_valid_all/metrics.json',
        'ranking_sensitivity.json':'artifacts/ranking_v3/sensitivity_valid_all/metrics.json',
        'training_cache_audit.json':'artifacts/ranking_v3/training_cache_audit/metrics.json',
        'ranking_controlled_ablation.json':'artifacts/ranking_v3/controlled_ablation_v1/metrics.json',
        'dcn_aggregate.json':'artifacts/dcn_enhancement_v3/summary.json',
        'dcn_fixed_pool.json':'artifacts/dcn_enhancement_v3/evaluation/metrics.json',
        'dcn_input_audit.json':'artifacts/dcn_enhancement_v3/audit/metrics.json',
        'dcn_runtime_256.json':'artifacts/dcn_enhancement_v3/runtime_valid_256.json',
        'dcn_baseline_runtime_256.json':'artifacts/dcn_enhancement_v3/runtime_baseline_256.json',
        'pairwise_pair_contract.json':'artifacts/pairwise_v1/pairs/metadata.json',
        'pairwise_training.json':'artifacts/pairwise_final_v1/models/metrics.json',
        'pairwise_fixed_pool.json':'artifacts/pairwise_final_v1/evaluation/metrics.json',
        'pairwise_audit.json':'artifacts/pairwise_final_v1/audit.json',
    }
    reports={}
    for name,source in sources.items():
        value=json.loads(Path(source).read_text(encoding='utf-8'))
        if value.get('test_used') is not False:raise ValueError('Expected validation-only report: '+source)
        if name=='dcn_fixed_pool.json':
            value['candidate_metadata']={k:v for k,v in value['candidate_metadata'].items() if k!='users'}
        if name=='dcn_input_audit.json':
            checks=value.pop('history_sql_checks')
            value['history_sql_check_summary']={'checks':len(checks),
                'rec_all_match':all(v['rec'] for v in checks),'search_all_match':all(v['search'] for v in checks),
                'sample_user_ids_omitted':True}
        # Avoid accidentally publishing per-user rows in a future report schema.
        def check(v):
            if isinstance(v,dict):
                for k,item in v.items():
                    if k in ('user_id','per_user','per_user_rows') or (k=='users' and isinstance(item,list)):
                        raise ValueError('Per-user content is not publishable: '+name)
                    check(item)
            elif isinstance(v,list):
                for item in v:check(item)
        check(value)
        reports[name]=(value,source)
    output.mkdir(parents=True)
    files={}
    for name,(value,source) in reports.items():
        path=output/name
        path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
        files[name]={'source':source,'source_sha256':digest(source),
            'published_sha256':digest(path),'published_bytes':path.stat().st_size}
    manifest={'published_date':'2026-10-22','scope':'aggregate validation-development evidence; not independent test or online gains',
        'test_used':False,'default_models_changed':False,'raw_data_arrays_per_user_rows_and_checkpoints_excluded':True,
        'dcn_final_model_runs':24,'pairwise_final_model_runs':30,'tests_passed':99,
        'files':files,'exporter_sha256':digest(__file__)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Published aggregate reports:',len(files),'to',output)


def normalize_newlines(output):
    """Formatting-only migration, verifying existing content before any writes."""
    path=output/'manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for name,record in manifest['files'].items():
        if Path(name).name!=name or digest(output/name)!=record['published_sha256']:
            raise ValueError('Do not format a modified report: '+name)
    for name,record in manifest['files'].items():
        report=output/name
        report.write_bytes(report.read_bytes().replace(b'\r\n',b'\n'))
        record['published_sha256']=digest(report)
        record['published_bytes']=report.stat().st_size
    manifest['exporter_sha256']=digest(__file__)
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Normalized published JSON newlines and refreshed checksums')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('experiments/ranking_progress_2026_10_22'))
    p.add_argument('--normalize-newlines',action='store_true')
    a=p.parse_args()
    normalize_newlines(a.out) if a.normalize_newlines else export(a.out)
