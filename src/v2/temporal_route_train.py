"""Matched-width 2x2 DCN training; fixed last epoch, all exposures retained."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from src.ctr_features import CACHE
from src.ranking import metrics
from src.v2.models import FineDCN
from src.v2.temporal_route_features import WIDTH, standardized
from src.v2.temporal_route_data import write_json
from src.v2.train_rankers import probability_metrics, sha


def verify_cache(path):
    meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('status')!='ready' or meta['test_used']:
        raise ValueError('Completed non-test cache required')
    if meta['state_sha256']!=sha(CACHE/'state.npz'):
        raise ValueError('Feature state changed')
    if meta['feature_source_sha256']!=sha(Path(__file__).with_name('temporal_route_features.py')):
        raise ValueError('Feature implementation changed after preparation')
    for name,digest in meta['files'].items():
        if Path(name).name!=name or sha(path/name)!=digest:
            raise ValueError('Cache file changed: '+name)
    if sha(path/'candidates.jsonl')!=meta['candidate_sha256']:
        raise ValueError('Frozen candidates changed')
    return meta


class FactorialRanker:
    def __init__(self,checkpoint):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.metadata=saved['metadata']
        if self.metadata['kind']!='temporal_route_factorial_v1':
            raise ValueError('Not a temporal-route experiment model')
        self.mean=np.asarray(saved['mean'],dtype=np.float32)
        self.std=np.asarray(saved['std'],dtype=np.float32)
        self.temporal,self.route=self.metadata['enabled']
        self.model=FineDCN(WIDTH)
        self.model.load_state_dict(saved['state_dict'],strict=True)
        self.model.eval()

    def predict(self,x,batch=65536):
        pieces=[]
        with torch.inference_mode():
            for lo in range(0,len(x),batch):
                z=standardized(x[lo:lo+batch],self.mean,self.std,self.temporal,self.route)
                pieces.append(torch.sigmoid(self.model(torch.from_numpy(z))).numpy())
        return np.concatenate(pieces) if pieces else np.empty(0,dtype=np.float32)


def run(config_path,cache,output):
    if output.exists():
        raise FileExistsError('Choose new training root')
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    meta=verify_cache(cache)
    if cfg!=meta['config'] or cfg['checkpoint_rule']!='fixed final epoch; no validation checkpoint selection':
        raise ValueError('Preparation/training protocol differs')
    output.mkdir(parents=True)
    protocol={'config':cfg,'cache':str(cache),'cache_metadata_sha256':sha(cache/'metadata.json'),
              'source_sha256':sha(__file__),'model_source_sha256':sha(Path(__file__).with_name('models.py')),
              'feature_source_sha256':meta['feature_source_sha256'],'test_used':False,'default_changed':False}
    write_json(output/'protocol.json',protocol)
    torch.set_num_threads(cfg['threads'])
    x=np.load(cache/'train_X.npy',mmap_mode='r');y=np.load(cache/'train_y.npy')
    xv=np.load(cache/'valid_X.npy',mmap_mode='r')
    yv=np.load(CACHE/'valid_y.npy');uv=np.load(CACHE/'valid_users.npy')
    mean=np.load(cache/'mean.npy');std=np.load(cache/'std.npy')
    ty=torch.from_numpy(y.astype(np.float32))
    runs={}
    for arm,enabled in cfg['arms'].items():
        tx=torch.from_numpy(standardized(x,mean,std,*enabled))
        for seed in cfg['seeds']:
            name=f'{arm}_s{seed}';dest=output/name;dest.mkdir()
            torch.manual_seed(seed)
            model=FineDCN(WIDTH)
            initial_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            order_hash=hashlib.sha256()
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            history=[];started=perf_counter()
            for epoch in range(cfg['epochs']):
                model.train();order=torch.randperm(len(tx));total=0.
                order_hash.update(order.numpy().tobytes())
                for lo in range(0,len(tx),cfg['batch_size']):
                    ix=order[lo:lo+cfg['batch_size']]
                    loss=nn.functional.binary_cross_entropy_with_logits(model(tx[ix]),ty[ix])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite factorial training loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                    total+=loss.item()*len(ix)
                # Curves are diagnostic only; checkpoint remains the fixed final epoch.
                model.eval();prediction=[]
                with torch.inference_mode():
                    for lo in range(0,len(xv),65536):
                        z=standardized(xv[lo:lo+65536],mean,std,*enabled)
                        prediction.append(torch.sigmoid(model(torch.from_numpy(z))).numpy())
                p=np.concatenate(prediction)
                record={'epoch':epoch+1,'train_logloss':total/len(tx),'valid':probability_metrics(yv,p)}
                history.append(record)
                print(f'{name} epoch {epoch+1}: trainLL={record["train_logloss"]:.6f}, validAUC={record["valid"]["auc"]:.6f}, validLL={record["valid"]["logloss"]:.6f}',flush=True)
            metadata={'kind':'temporal_route_factorial_v1','arm':arm,'seed':seed,'enabled':enabled,
                      'checkpoint_rule':cfg['checkpoint_rule'],'epochs':cfg['epochs'],
                      'cache_metadata_sha256':protocol['cache_metadata_sha256'],
                      'feature_state_sha256':meta['state_sha256'],'model_source_sha256':protocol['model_source_sha256'],
                      'feature_source_sha256':protocol['feature_source_sha256'],'trainer_source_sha256':protocol['source_sha256'],
                      'feature_names':meta['feature_names'],'test_used':False}
            metadata.update({'initial_parameters_sha256':initial_hash,'training_order_sha256':order_hash.hexdigest(),
                             'sample_rows':len(tx)})
            torch.save({'metadata':metadata,'mean':mean.tolist(),'std':std.tolist(),'state_dict':model.state_dict()},dest/'model.pt')
            ranker=FactorialRanker(dest/'model.pt')
            loaded=ranker.predict(xv)
            if not np.array_equal(p,loaded):
                raise ValueError('Reloaded prediction changed')
            np.save(dest/'valid_predictions.npy',loaded)
            route_hit=(np.asarray(xv[:,28:30])>0).any(axis=1)
            groups={'all':np.ones(len(yv),dtype=bool),'route_hit':route_hit,'route_missing':~route_hit}
            report={'metadata':metadata,'history':history,'parameters':sum(q.numel() for q in model.parameters()),
                    'exposure':{g:metrics(yv[m],loaded[m],uv[m]) for g,m in groups.items() if m.any()},
                    'checkpoint_sha256':sha(dest/'model.pt'),'valid_predictions_sha256':sha(dest/'valid_predictions.npy'),
                    'elapsed_seconds':perf_counter()-started,'test_used':False}
            write_json(dest/'metrics.json',report);runs[name]=report
            write_json(output/'progress.json',{'runs':runs,'test_used':False})
    result={'protocol':protocol,'runs':runs,'test_used':False,'default_changed':False}
    write_json(output/'metrics.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/temporal_route_factorial.json'))
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.config,args.cache,args.out)
