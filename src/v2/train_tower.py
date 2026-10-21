"""Train and export v2 dual towers; real training is an explicit separate CLI."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.itemcf_index import digest
from src.v2.tower_data import load_metadata
from src.v2.tower_model import FeatureTwoTower, in_batch_loss, model_source_digest


def arrays(path, names):
    return {name:np.load(path/(name+'.npy'),mmap_mode='r') for name in names}


class TowerSamples(Dataset):
    def __init__(self,path):
        self.data=arrays(path,['sample_contexts','targets','context_users','context_rec','context_search'])

    def __len__(self):
        return len(self.data['targets'])

    def __getitem__(self,index):
        row=int(self.data['sample_contexts'][index])
        return (int(self.data['context_users'][row]),int(self.data['targets'][index]),
                np.array(self.data['context_rec'][row],dtype=np.int64),
                np.array(self.data['context_search'][row],dtype=np.int64))


def make_device(name):
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; no automatic full-data CPU training fallback')
    return device


def train(data,config,output):
    if output.exists():
        raise FileExistsError('Preserve previous tower run')
    if config.get('epochs',0)<1 or config.get('batch_size',0)<2:
        raise ValueError('Invalid training budget')
    meta=load_metadata(data)
    device=make_device(config.get('device','cuda'))
    torch.manual_seed(config.get('seed',2026))
    torch.set_num_threads(4)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
    fields=np.load(data/'item_fields.npy')
    model=FeatureTwoTower(meta['cardinalities'],config,fields).to(device)
    sparse=model.sparse_parameters()
    sparse_ids={id(p) for p in sparse}
    optim_sparse=torch.optim.SparseAdam(sparse,lr=config.get('sparse_lr',.01))
    optim_dense=torch.optim.Adam([p for p in model.parameters() if id(p) not in sparse_ids],lr=config.get('dense_lr',.001))
    dataset=TowerSamples(data)
    generator=torch.Generator().manual_seed(config.get('seed',2026))
    loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,num_workers=0,generator=generator)
    frequency=torch.as_tensor(np.load(data/'target_counts.npy'),dtype=torch.float32,device=device)
    logq=(frequency/frequency.sum()).clamp_min(1e-12).log() if config.get('logq_correction',False) else None
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    start=perf_counter()
    history=[]
    for epoch in range(config['epochs']):
        model.train()
        total=rows=skipped=0
        for uid,target,rec,search in loader:
            uid,target,rec,search=(v.to(device) for v in (uid,target,rec,search))
            unique=torch.unique(target,sorted=True)
            if len(unique)<2:
                skipped+=len(target)
                continue
            loss=in_batch_loss(model.users(uid,rec,search),model.items(unique),target,rec,search,
                               config.get('temperature',.1),logq,config.get('mask_history_negatives',True))
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite retrieval loss')
            optim_sparse.zero_grad(set_to_none=True)
            optim_dense.zero_grad(set_to_none=True)
            loss.backward()
            optim_sparse.step()
            optim_dense.step()
            total+=float(loss.detach())*len(target)
            rows+=len(target)
        if not rows:
            raise ValueError('No usable batches with distinct candidate items')
        record={'epoch':epoch+1,'loss':total/rows,'trained_events':rows,'skipped_events':skipped}
        history.append(record)
        print(json.dumps(record),flush=True)
    checkpoint={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'config':config,'data_metadata_sha256':digest(data/'metadata.json'),
                'cardinalities':meta['cardinalities'],'model_source_sha256':model_source_digest(),
                'torch_version':str(torch.__version__)}
    torch.save(checkpoint,output/'model.pt')
    report={'status':'trained','config':config,'training':history,'train_seconds':perf_counter()-start,
            'selection':'fixed final epoch; no validation/test labels read',
            'data_metadata_sha256':checkpoint['data_metadata_sha256'],
            'manifest_sha256':meta['manifest_sha256'],'cutoff_ms':meta['cutoff_ms'],
            'checkpoint_sha256':digest(output/'model.pt'),
            'source_sha256':digest(Path(__file__)),'model_source_sha256':checkpoint['model_source_sha256'],
            'negative_semantics':'unique in-batch item classes; same-user targets not automatically positive; optional prior-history negative masking',
            'logq_semantics':'optional approximate log-frequency correction after unique candidate deduplication; not an unbiased estimator',
            'test_labels_used':False}
    (output/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def export(data,checkpoint,output,device_name='cpu'):
    if output.exists():
        raise FileExistsError('Tower index is immutable; choose a new path')
    meta=load_metadata(data)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['data_metadata_sha256']!=digest(data/'metadata.json'):
        raise ValueError('Checkpoint and prepared data differ')
    if saved.get('model_source_sha256')!=model_source_digest():
        raise ValueError('Model implementation changed since checkpoint training; export with matching source')
    config=saved['config']
    device=make_device(device_name)
    model=FeatureTwoTower(meta['cardinalities'],config,np.load(data/'item_fields.npy'))
    model.load_state_dict(saved['state_dict'],strict=True)
    model.to(device).eval()
    state=arrays(data,['items','query_users','query_user_tokens','query_rec','query_search'])
    dim=config.get('dimension',64)
    batch=config.get('export_batch_size',2048)
    if not isinstance(batch,int) or batch<1:
        raise ValueError('Invalid export batch size')
    output.mkdir(parents=True,exist_ok=False)
    vectors=np.lib.format.open_memmap(output/'vectors.npy',mode='w+',dtype='float32',shape=(len(state['items']),dim))
    queries=np.lib.format.open_memmap(output/'queries.npy',mode='w+',dtype='float32',shape=(len(state['query_users']),dim))
    signal=(state['query_user_tokens']>=2) & config.get('use_user_id',True)
    signal=signal | ((state['query_rec']!=0).any(1) & config.get('use_history',True))
    signal=signal | ((state['query_search']!=0).any(1) & config.get('use_search',False))
    with torch.inference_mode():
        for lo in range(0,len(vectors),batch):
            hi=min(lo+batch,len(vectors))
            vectors[lo:hi]=model.items(torch.arange(lo+2,hi+2,device=device)).cpu().numpy()
        for lo in range(0,len(queries),batch):
            values=[torch.tensor(np.array(state[name][lo:lo+batch]),dtype=torch.long,device=device)
                    for name in ('query_user_tokens','query_rec','query_search')]
            queries[lo:lo+batch]=model.users(*values).cpu().numpy()
    vectors.flush()
    queries.flush()
    vectors._mmap.close()
    queries._mmap.close()
    np.save(output/'items.npy',state['items'])
    np.save(output/'users.npy',state['query_users'])
    np.save(output/'query_signal.npy',signal)
    metadata={'status':'ready','schema_version':1,'dimension':dim,'cutoff_ms':meta['cutoff_ms'],
              'manifest_sha256':meta['manifest_sha256'],'data_metadata_sha256':digest(data/'metadata.json'),
              'checkpoint_sha256':digest(checkpoint),'config':config,
              'model_source_sha256':saved['model_source_sha256'],
              'source_sha256':digest(Path(__file__)),
              'score':'unit-normalized dot product, not CTR',
              'query_rule':meta['query_rule'],'catalog':meta['catalog'],
              'cold_user':'empty route when no trained ID or enabled history signal; no popularity hidden here',
              'files':{p.name:digest(p) for p in sorted(output.glob('*.npy'))}}
    metadata['version']=hashlib.sha256(json.dumps({'files':metadata['files'],
        'checkpoint':metadata['checkpoint_sha256'],'manifest':metadata['manifest_sha256'],
        'cutoff':metadata['cutoff_ms']},sort_keys=True).encode()).hexdigest()[:20]
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    t.add_argument('--data',type=Path,required=True)
    t.add_argument('--config',type=Path,default=Path('configs/two_tower_v2.json'))
    t.add_argument('--out',type=Path,required=True)
    e=sub.add_parser('export')
    e.add_argument('--data',type=Path,required=True)
    e.add_argument('--checkpoint',type=Path,required=True)
    e.add_argument('--out',type=Path,required=True)
    e.add_argument('--device',default='cpu')
    args=parser.parse_args()
    result=(train(args.data,json.loads(args.config.read_text(encoding='utf-8')),args.out) if args.command=='train'
            else export(args.data,args.checkpoint,args.out,args.device))
    print(json.dumps(result,indent=2))
