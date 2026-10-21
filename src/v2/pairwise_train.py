"""Pointwise vs nearby observed-pair training, fixed samples and validation pairs."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from src.ctr_features import CACHE
from src.ranking import metrics
from src.v2.enhanced_data import EnhancedData
from src.v2.enhanced_model import EnhancedDCN, EnhancedRanker, spec_for
from src.v2.enhanced_train import Banks
from src.v2.train_rankers import sha, probability_metrics


def objective_loss(positive, negative, objective, mixed_weight=.5):
    pair=F.softplus(negative-positive).mean()
    bce=(F.softplus(-positive).mean()+F.softplus(negative).mean())/2
    if objective=='paired_bce':return bce
    if objective=='pairwise':return pair
    if objective=='mixed':return (1-mixed_weight)*bce+mixed_weight*pair
    raise ValueError('Unknown paired objective')


def pair_metrics(logits, positive, negative):
    difference=np.asarray(logits[positive],dtype=np.float64)-np.asarray(logits[negative],dtype=np.float64)
    return {'pairs':len(difference),'logloss':float(np.logaddexp(0,-difference).mean()),
            'accuracy_half_ties':float(np.mean((difference>0)+.5*(difference==0)))}


def paired_coverage(bank,spec,positive,negative):
    if not spec['attrs']:return {}
    endpoints=torch.cat([positive,negative])
    return {name:torch.nonzero(torch.bincount(bank.fields[endpoints,column],
        minlength=bank.data.meta['cardinalities'][name])>=5).flatten()
        for name,column in [('author',2),('category',3)]}


def validation_logits(model, bank):
    model.eval();pieces=[]
    with torch.inference_mode():
        for lo in range(0,len(bank.yv),16384):
            sl=slice(lo,lo+16384)
            x=torch.from_numpy((np.asarray(bank.xv[sl])-bank.mean)/bank.std)
            fields=torch.tensor(np.asarray(bank.data.valid_fields[sl]),dtype=torch.long)
            pieces.append(model(x,fields).numpy())
    return np.concatenate(pieces)


def train_one(architecture, objective, seed, cfg, bank, pairs, output, pair_digest):
    if output.exists():raise FileExistsError('Preserve existing pair training')
    output.mkdir(parents=True)
    torch.manual_seed(seed)
    spec=spec_for(architecture)
    model=EnhancedDCN(spec,bank.data.meta['cardinalities'])
    tp,tn,vp,vn=pairs
    model.set_coverage(bank.coverage(spec,5) if objective=='full_bce' else paired_coverage(bank,spec,tp,tn))
    dense=[p for n,p in model.named_parameters() if not n.startswith('tables.')]
    optimizers=[torch.optim.Adam(dense,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])]
    if len(model.tables):optimizers.append(torch.optim.SparseAdam(model.tables.parameters(),lr=cfg['learning_rate']))
    metadata={'kind':'enhanced_dcn_v2','experiment':architecture+'_'+objective,'spec':spec,'seed':seed,
        'selection':cfg['selection'],'data_sha256':sha(bank.data.path/'metadata.json'),
        'state_sha256':sha(CACHE/'state.npz'),'sample_sha256':sha(bank.data.path/'sample_indices.npy'),
        'pair_metadata_sha256':pair_digest,'cardinalities':bank.data.meta['cardinalities'],'test_used':False,
        'model_source_sha256':sha(Path(__file__).with_name('enhanced_model.py')),'trainer_source_sha256':sha(__file__),
        'probability_semantics':'sigmoid of pure pairwise scores is not calibrated CTR' if objective=='pairwise' else 'exposure/endpoint sampling distribution; calibration assessed, not guaranteed'}
    history=[];best=float('inf');started=perf_counter()
    for epoch in range(cfg['epochs']):
        model.train();length=len(bank.y) if objective=='full_bce' else len(tp)
        order=torch.randperm(length);total=0.;steps=0
        for lo in range(0,length,cfg['batch_size']):
            idx=order[lo:lo+cfg['batch_size']]
            if objective=='full_bce':
                logits=model(bank.numeric[idx],bank.fields[idx]);loss=F.binary_cross_entropy_with_logits(logits,bank.y[idx])
            else:
                p,n=tp[idx],tn[idx]
                logits=model(torch.cat([bank.numeric[p],bank.numeric[n]]),torch.cat([bank.fields[p],bank.fields[n]]))
                positive,negative=logits.chunk(2)
                loss=objective_loss(positive,negative,objective,cfg['mixed_weight'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite pair loss')
            for opt in optimizers:opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers:opt.step()
            total+=loss.item()*len(idx);steps+=1
        logits=validation_logits(model,bank)
        probability=torch.sigmoid(torch.from_numpy(logits)).numpy()
        pair=pair_metrics(logits,vp,vn)
        history.append({'epoch':epoch+1,'train_objective':total/length,'optimizer_steps':steps,
            'valid_pair':pair,'valid_exposure':probability_metrics(bank.yv,probability)})
        print(f'{architecture}_{objective}/s{seed} epoch {epoch+1}: pairLL={pair["logloss"]:.6f}, AUC={history[-1]["valid_exposure"]["auc"]:.6f}',flush=True)
        if pair['logloss']<best:
            best=pair['logloss']
            torch.save({'metadata':{**metadata,'selected_epoch':epoch+1},'mean':bank.mean.tolist(),
                'std':bank.std.tolist(),'state_dict':model.state_dict()},output/'model.pt')
    ranker=EnhancedRanker(output/'model.pt',bank.data,metadata['state_sha256'])
    logits=validation_logits(ranker.model,bank);probability=torch.sigmoid(torch.from_numpy(logits)).numpy()
    np.save(output/'valid_logits.npy',logits);np.save(output/'valid_predictions.npy',probability)
    frequency=bank.data.item_train_frequency[bank.iv]
    groups={'all':np.ones(len(logits),dtype=bool),'unseen_item':frequency==0,
        'low_frequency_item':(frequency>0)&(frequency<5),'high_frequency_item':frequency>=20}
    report={'metadata':ranker.metadata,'history':history,'validation':{k:metrics(bank.yv[m],probability[m],bank.uv[m]) for k,m in groups.items() if m.any()},
        'valid_pair':pair_metrics(logits,vp,vn),'parameters':sum(p.numel() for p in model.parameters()),
        'checkpoint_sha256':sha(output/'model.pt'),'checkpoint_bytes':(output/'model.pt').stat().st_size,
        'elapsed_seconds':perf_counter()-started,'test_used':False}
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def run(config,pair_cache,output):
    if output.exists():raise FileExistsError('Choose a new model output')
    cfg=json.loads(config.read_text());meta=json.loads((pair_cache/'metadata.json').read_text())
    if meta['config_sha256']!=sha(config) or meta['test_used']:raise ValueError('Pair protocol mismatch')
    for name,digest in meta['files'].items():
        if Path(name).name!=name or sha(pair_cache/name)!=digest:raise ValueError('Pair checksum mismatch')
    torch.set_num_threads(4)
    with EnhancedData(cfg['enhanced_cache']) as data:
        if sha(data.path/'metadata.json')!=meta['enhanced_data_sha256'] or sha(CACHE/'state.npz')!=meta['state_sha256']:
            raise ValueError('Pair/enhanced/numeric state mismatch')
        bank=Banks(data)
        pairs=[torch.from_numpy(np.load(pair_cache/f'train_{k}.npy')) for k in ('positive','negative')]
        pairs += [np.load(pair_cache/f'valid_{k}.npy') for k in ('positive','negative')]
        output.mkdir(parents=True)
        protocol={'config':cfg,'source_sha256':sha(__file__),'pairs_sha256':sha(pair_cache/'metadata.json'),'test_used':False}
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        runs={}
        for architecture in cfg['models']:
            for objective in cfg['objectives']:
                for seed in cfg['seeds']:
                    key=f'{architecture}_{objective}_s{seed}'
                    runs[key]=train_one(architecture,objective,seed,cfg,bank,pairs,output/key,protocol['pairs_sha256'])
                    (output/'progress.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
        (output/'metrics.json').write_text(json.dumps({'protocol':protocol,'runs':runs,'test_used':False,'default_model_changed':False},indent=2),encoding='utf-8')
        bank.xv._mmap.close()
    print('Pairwise training complete:',len(runs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/pairwise_v1.json'))
    p.add_argument('--pairs',type=Path,default=Path('artifacts/pairwise_v1/pairs'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config,a.pairs,a.out)
