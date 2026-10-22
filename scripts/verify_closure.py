"""Publish an audited local release, or check hashes without real-data training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.itemcf_index import digest
from src.v2.ranking_convergence import write_json
from src.v2.closure_integrity import verify_v1_archive

RELEASE=Path('experiments/closure_20261022')


def immutable_json(path,value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:
            raise ValueError('Preserve existing published file: '+str(path))
    else: write_json(path,value)


def freeze(root,release=RELEASE):
    if (release/'release.json').exists(): raise FileExistsError('Release is frozen; do not overwrite')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    tests=json.loads((root/'tests.json').read_text(encoding='utf-8'))
    if summary['status']!='audited_complete' or audit['status']!='passed' or tests['status']!='passed':
        raise ValueError('Completed study, independent audit and full tests required')
    if summary['audit_sha256']!=digest(root/'audit.json'):
        raise ValueError('Summary and audit differ')
    if audit['source_sha256']!=digest('scripts/audit_tower_closure.py'):
        raise ValueError('Audit code changed')
    if tests['source_sha256']!=digest('scripts/test_closure.py'):
        raise ValueError('Test runner changed')
    for group in (tests['test_sources'],tests['implementation_sources'],audit['checked_sources']):
        for name,expected in group.items():
            if digest(name)!=expected: raise ValueError('Code changed after test/audit: '+name)
    archive=verify_v1_archive()
    release.mkdir(parents=True,exist_ok=True)
    immutable_json(release/'summary.json',summary)
    immutable_json(release/'audit.json',audit)
    immutable_json(release/'tests.json',tests)
    curves=(root/'curves.html').read_text(encoding='utf-8')
    (release/'curves.html').write_text(curves,encoding='utf-8',newline='\n')
    # No per-user IDs, arrays, raw data, vocabularies or embedding weights published.
    local={root/'study.json',root/'summary.json',root/'audit.json',root/'tests.json'}
    spec=json.loads((root/'study.json').read_text(encoding='utf-8'))
    for prepared in ('data','earlier_data'):
        path=root/prepared; meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        local.add(path/'metadata.json')
        local.update(path/name for name in meta['files'])
    for job in spec['jobs']:
        model=Path(job['model']); evaluation=root/'evaluation'/job['name']
        local.update(model/name for name in ('model.pt','last.pt','metrics.json','protocol.json'))
        index=json.loads((model/'index/metadata.json').read_text(encoding='utf-8'))
        local.add(model/'index/metadata.json')
        local.update(model/'index'/name for name in index['files'])
        e=json.loads((evaluation/'metrics.json').read_text(encoding='utf-8'))
        local.add(evaluation/'metrics.json');local.update(evaluation/name for name in e['files'])
    for key in ('baseline_itemcf','earlier_itemcf'):
        path=Path(spec[key]); meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        local.add(path/'metadata.json');local.update(path/name for name in meta['files'])
    for base in ('baseline','earlier_baseline'):
        path=root/base;meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        local.add(path/'metadata.json');local.update(path/name for name in meta['files'])
    local.update([Path('data/processed/kuaisar_small_v1/manifest.json'),
                  Path('data/processed/kuaisar_small_v1/dataset.sqlite'),
                  Path('data/processed/kuaisar_small_v1/ctr_features_v1/state.npz')])
    if Path('data/raw/KuaiSAR.zip').exists(): local.add(Path('data/raw/KuaiSAR.zip'))
    public={Path('README.md'),Path('.gitignore'),Path('.gitattributes')}
    for directory,pattern in [('src','*.py'),('scripts','*.py'),('scripts','*.ps1'),
                              ('configs','*.json'),('docs','*.md'),('tests','*.py')]:
        public.update(Path(directory).rglob(pattern))
    public.update(release.glob('*'))
    public.update(Path('.').glob('requirements*'))
    record={'schema_version':1,'status':'frozen','test_used_for_new_selection':False,'default_changed':False,
            'decision':summary['decision'],'root':str(root),'frozen_v1_assets_verified':archive['verified_assets'],
            'v1_archive':archive,
            'scope':'v2 development and retrospective release, not a new blind test or production system',
            'source_sha256':digest(Path(__file__)),
            'public_assets':{p.as_posix():digest(p) for p in sorted(public)},
            'local_assets':{p.as_posix():digest(p) for p in sorted(local)}}
    write_json(release/'release.json',record)
    return {'status':'frozen','public_assets':len(public),'local_assets':len(local),'decision':summary['decision']}


def verify(release=RELEASE,public_only=False):
    record=json.loads((release/'release.json').read_text(encoding='utf-8'))
    if record['status']!='frozen' or record['schema_version']!=1: raise ValueError('Unsupported release')
    checked=0
    for group in ('public_assets',) if public_only else ('public_assets','local_assets'):
        for name,expected in record[group].items():
            if digest(name)!=expected: raise ValueError('Release asset changed: '+name)
            checked+=1
    if not public_only: verify_v1_archive()
    return {'status':'passed','checked_assets':checked,'public_only':public_only,
            'real_training_run':False,'test_labels_read':False,'default_changed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--public-only',action='store_true')
    p.add_argument('--root',type=Path,default=Path('artifacts/tower_closure_20261022'))
    p.add_argument('--release',type=Path,default=RELEASE);a=p.parse_args()
    print(json.dumps(freeze(a.root,a.release) if a.freeze else verify(a.release,a.public_only),indent=2))
