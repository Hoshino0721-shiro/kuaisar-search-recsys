"""Resume-safe bounded study, with capped parallel local training processes."""
import argparse
from contextlib import closing
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.itemcf_index import DATA, digest, IndexConfig, ItemCFIndex, read_snapshot, build
from src.v2.ranking_convergence import write_json
from src.v2.closure_integrity import verify_v1_archive
from src.v2.tower_data import prepare
from scripts.audit_tower_data import audit as audit_data


def ensure_controls(spec):
    config=IndexConfig(**json.loads(Path('configs/itemcf_index_support2.json').read_text(encoding='utf-8')))
    main=Path(spec['baseline_itemcf'])
    if not main.exists(): build(read_snapshot(DATA,config),config,main)
    from src.v2.temporal_route_data import prefix_index
    with closing(sqlite3.connect((DATA/'dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)) as db:
        prefix=prefix_index(db,1685203200000,config,Path(spec['earlier_itemcf']),
                            digest(DATA/'manifest.json'),1685289600000,main)
        if prefix is None: raise ValueError('No earlier-prefix graph')
        prefix.close()
    with_main=ItemCFIndex(main,digest(DATA/'manifest.json'))
    with_main.close()


def jobs(root):
    result=[]
    # History jobs first; attributes are faster and fill remaining slots later.
    for arm,seeds in [('history',[2026,2027,2028]),('attributes',[2027,2028,2026])]:
        for seed in seeds:
            result.append({'name':f'{arm}_s{seed}','arm':arm,'seed':seed,'window':'main',
                           'prepared':str(root/'data'),'model':str(root/'models'/f'{arm}_s{seed}')})
    for arm in ('history','attributes'):
        result.append({'name':f'earlier_{arm}_s2026','arm':arm,'seed':2026,'window':'earlier',
                       'prepared':str(root/'earlier_data'),'model':str(root/'models'/f'earlier_{arm}_s2026')})
    return result


@contextmanager
def study_lock(root):
    """OS-released lock; a crashed process cannot leave a stale held lock."""
    root.mkdir(parents=True,exist_ok=True)
    with (root/'study.lock').open('a+b') as handle:
        if handle.seek(0,2)==0:
            handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc: raise RuntimeError('This study is already running; do not launch a duplicate') from exc
        try: yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def run(root, workers, train_only=False):
    if not 1<=workers<=3: raise ValueError('Cap at three four-thread processes (12 logical CPU threads)')
    frozen=verify_v1_archive()['verified_assets']
    root.mkdir(parents=True,exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    spec={'config_sha256':digest('configs/tower_closure.json'),'jobs':jobs(root),
          'baseline_itemcf':'artifacts/itemcf_support2_300',
          'earlier_itemcf':'artifacts/full_exposure_local_20261022/reference/indexes/1685203200000',
          'primary':'history; attributes is a fixed diagnostic, not post-hoc winner selection',
          'adoption_rule':'history must add Recall@2000 and not lose Top20 hits in all three main seeds and the earlier seed; otherwise retain RRF',
          'test_used':False,'default_changed':False,'frozen_assets_verified':frozen}
    if (root/'study.json').exists():
        if json.loads((root/'study.json').read_text(encoding='utf-8'))!=spec:
            raise ValueError('Study protocol changed; preserve this run')
    else: write_json(root/'study.json',spec)
    for name,cutoff in [('data',None),('earlier_data',1685203200000)]:
        if not (root/name).exists(): prepare(DATA,root/name,cutoff_ms=cutoff)
        path=root/(name+'_audit.json')
        if not path.exists(): audit_data(DATA,root/name,path)
    pending=[]
    for job in spec['jobs']:
        path=Path(job['model'])/'metrics.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))['status']=='complete': continue
        pending.append(job)
    active=[]
    while pending or active:
        while pending and len(active)<workers:
            if digest('configs/tower_closure.json')!=spec['config_sha256']:
                raise ValueError('Training configuration changed while study was running')
            job=pending.pop(0)
            command=[sys.executable,'-m','src.v2.tower_closure','--prepared',job['prepared'],
                     '--out',job['model'],'--arm',job['arm'],'--seed',str(job['seed'])]
            if (Path(job['model'])/'last.pt').exists(): command.append('--resume')
            elif Path(job['model']).exists():
                raise ValueError('Partial first epoch retained without a checkpoint: '+job['model'])
            log=(root/'logs'/(job['name']+'.log')).open('a',encoding='utf-8')
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            active.append((job,proc,log))
            print('Started '+job['name'],flush=True)
        for job,proc,log in list(active):
            code=proc.poll()
            if code is None: continue
            log.close(); active.remove((job,proc,log))
            if code:
                # Do not kill healthy peer jobs or delete their partial state.
                raise RuntimeError(f'{job["name"]} exited {code}; inspect {root/"logs"/(job["name"]+".log")}')
            print('Completed '+job['name'],flush=True)
        progress={j['name']:(json.loads((Path(j['model'])/'metrics.json').read_text(encoding='utf-8'))
                            if (Path(j['model'])/'metrics.json').exists() else {'status':'pending'})
                  for j in spec['jobs']}
        write_json(root/'progress.json',{'stage':'training','models':{
            k:{p:v.get(p) for p in ('status','selected_epoch','elapsed_seconds','early_stopping_reason')}
            for k,v in progress.items()},'test_used':False})
        if active: time.sleep(5)
    if train_only: return
    from src.v2.tower_closure_evaluate import prepare_baseline,evaluate
    ensure_controls(spec)
    for window,cf in [('main',spec['baseline_itemcf']),('earlier',spec['earlier_itemcf'])]:
        baseline=root/('baseline' if window=='main' else 'earlier_baseline')
        prepare_baseline(DATA,Path(cf),baseline)
        for job in spec['jobs']:
            if job['window']!=window: continue
            print('Evaluating '+job['name'],flush=True)
            evaluate(Path(job['prepared']),Path(job['model']),baseline,root/'evaluation'/job['name'])
    from scripts.summarize_tower_closure import summarize
    summarize(root)
    print('All bounded training and evaluation jobs completed; inspect summary and run final audit.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('artifacts/tower_closure_20261022'))
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--train-only',action='store_true')
    args=parser.parse_args()
    with study_lock(args.root):
        run(args.root,args.workers,args.train_only)
