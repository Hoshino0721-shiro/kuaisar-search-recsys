"""Rebuild only affected attribute/paired arms; verify and preserve other runs."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from src.ctr_features import CACHE
from src.v2.enhanced_data import EnhancedData
from src.v2.enhanced_train import Banks
from src.v2.pairwise_train import train_one
from src.v2.train_rankers import sha


def repair(base,combined,output):
    if output.exists():raise FileExistsError('Preserve prior repaired run')
    original=json.loads((base/'models/metrics.json').read_text());previous=json.loads((combined/'models/metrics.json').read_text())
    cfg=previous['protocol']['config'];pair_hash=sha(base/'pairs/metadata.json')
    meta=json.loads((base/'pairs/metadata.json').read_text())
    if original['test_used'] or previous['test_used'] or original['protocol']['pairs_sha256']!=pair_hash or previous['protocol']['pairs_sha256']!=pair_hash:
        raise ValueError('Previous protocol/pairs mismatch')
    for name,digest in meta['files'].items():
        if Path(name).name!=name or sha(base/'pairs'/name)!=digest:raise ValueError('Pair arrays changed')
    torch.set_num_threads(4)
    with EnhancedData(cfg['enhanced_cache']) as data:
        if meta['enhanced_data_sha256']!=sha(data.path/'metadata.json') or meta['state_sha256']!=sha(CACHE/'state.npz'):
            raise ValueError('Feature data changed')
        bank=Banks(data);output.mkdir(parents=True)
        protocol={**previous['protocol'],'repair_source_sha256':sha(__file__),'current_pair_trainer_sha256':sha(Path(__file__).with_name('pairwise_train.py')),
            'reuse_report_sha256':sha(combined/'models/metrics.json'),
            'repair':'B paired_bce/pairwise/mixed coverage now counts actual repeated pair endpoints, not full exposure sample',
            'test_used':False}
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        pairs=[torch.from_numpy(np.load(base/'pairs'/f'train_{k}.npy')) for k in ('positive','negative')]
        pairs += [np.load(base/'pairs'/f'valid_{k}.npy') for k in ('positive','negative')]
        runs={}
        for name,r in previous['runs'].items():
            objective=r['metadata']['experiment'].split('_',1)[1]
            if name.startswith('B_') and objective in ('paired_bce','pairwise','mixed'):
                runs[name]=train_one('B',objective,r['metadata']['seed'],cfg,bank,pairs,output/name,pair_hash)
            else:
                source=combined/'models'/name
                if sha(source/'model.pt')!=r['checkpoint_sha256'] or r['metadata']['data_sha256']!=sha(data.path/'metadata.json'):
                    raise ValueError('Unsafe reuse: '+name)
                shutil.copytree(source,output/name);runs[name]={**r,'reused_from':str(source)}
            (output/'progress.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
        (output/'metrics.json').write_text(json.dumps({'protocol':protocol,'runs':runs,'test_used':False,'default_model_changed':False},indent=2),encoding='utf-8')
        bank.xv._mmap.close()
    print('Repaired final comparison complete:',len(runs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('artifacts/pairwise_v1'))
    p.add_argument('--combined',type=Path,default=Path('artifacts/pairwise_fullmix_v1'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();repair(a.base,a.combined,a.out)
