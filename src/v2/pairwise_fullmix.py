"""Declared follow-up: full original-exposure BCE plus nearby pair loss."""
import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from src.ctr_features import CACHE
from src.ranking import metrics
from src.v2.enhanced_data import EnhancedData
from src.v2.enhanced_model import EnhancedDCN, EnhancedRanker, spec_for
from src.v2.enhanced_train import Banks
from src.v2.pairwise_train import validation_logits, pair_metrics
from src.v2.train_rankers import sha, probability_metrics


def full_mixed_loss(exposure_logits,labels,positive,negative,weight):
    return (1-weight)*F.binary_cross_entropy_with_logits(exposure_logits,labels)+weight*F.softplus(negative-positive).mean()


def train(architecture,seed,cfg,extra,bank,pairs,output,pair_hash):
    if output.exists():raise FileExistsError('Preserve full mixture output')
    output.mkdir(parents=True);torch.manual_seed(seed)
    spec=spec_for(architecture);model=EnhancedDCN(spec,bank.data.meta['cardinalities']);model.set_coverage(bank.coverage(spec,5))
    optimizers=[torch.optim.Adam([p for n,p in model.named_parameters() if not n.startswith('tables.')],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])]
    if len(model.tables):optimizers.append(torch.optim.SparseAdam(model.tables.parameters(),lr=cfg['learning_rate']))
    tp,tn,vp,vn=pairs
    # Pair draws must not consume the RNG used by full_bce's epoch permutations.
    pair_generator=torch.Generator().manual_seed(seed+extra['pair_draw_seed_offset'])
    metadata={'kind':'enhanced_dcn_v2','experiment':architecture+'_full_mixed','spec':spec,'seed':seed,
        'selection':cfg['selection'],'data_sha256':sha(bank.data.path/'metadata.json'),'state_sha256':sha(CACHE/'state.npz'),
        'sample_sha256':sha(bank.data.path/'sample_indices.npy'),'pair_metadata_sha256':pair_hash,
        'cardinalities':bank.data.meta['cardinalities'],'test_used':False,'model_source_sha256':sha(Path(__file__).with_name('enhanced_model.py')),
        'trainer_source_sha256':sha(__file__),'followup':True,'probability_semantics':'full exposure BCE plus auxiliary pair loss; calibration assessed, not guaranteed'}
    history=[];best=float('inf');started=perf_counter()
    for epoch in range(cfg['epochs']):
        model.train();order=torch.randperm(len(bank.y));total=0.;steps=0
        for lo in range(0,len(order),cfg['batch_size']):
            idx=order[lo:lo+cfg['batch_size']];draw=torch.randint(len(tp),(len(idx),),generator=pair_generator)
            p,n=tp[draw],tn[draw];size=len(idx)
            rows=torch.cat([idx,p,n]);logits=model(bank.numeric[rows],bank.fields[rows])
            loss=full_mixed_loss(logits[:size],bank.y[idx],logits[size:2*size],logits[2*size:],extra['mixed_weight'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite full mixture')
            for opt in optimizers:opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers:opt.step()
            total+=loss.item()*size;steps+=1
        logits=validation_logits(model,bank);probability=torch.sigmoid(torch.from_numpy(logits)).numpy();pair=pair_metrics(logits,vp,vn)
        scores=probability_metrics(bank.yv,probability)
        history.append({'epoch':epoch+1,'train_objective':total/len(order),'optimizer_steps':steps,
            'valid_pair':pair,'valid_exposure':scores})
        print(f'{architecture}_full_mixed/s{seed} epoch {epoch+1}: pairLL={pair["logloss"]:.6f}, AUC={scores["auc"]:.6f}',flush=True)
        if pair['logloss']<best:
            best=pair['logloss'];torch.save({'metadata':{**metadata,'selected_epoch':epoch+1},'mean':bank.mean.tolist(),
                'std':bank.std.tolist(),'state_dict':model.state_dict()},output/'model.pt')
    selected=EnhancedRanker(output/'model.pt',bank.data,metadata['state_sha256']);logits=validation_logits(selected.model,bank)
    probability=torch.sigmoid(torch.from_numpy(logits)).numpy()
    np.save(output/'valid_logits.npy',logits);np.save(output/'valid_predictions.npy',probability)
    freq=bank.data.item_train_frequency[bank.iv]
    groups={'all':np.ones(len(logits),dtype=bool),'unseen_item':freq==0,'low_frequency_item':(freq>0)&(freq<5),'high_frequency_item':freq>=20}
    report={'metadata':selected.metadata,'history':history,'valid_pair':pair_metrics(logits,vp,vn),
        'validation':{k:metrics(bank.yv[m],probability[m],bank.uv[m]) for k,m in groups.items() if m.any()},
        'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_bytes':(output/'model.pt').stat().st_size,
        'checkpoint_sha256':sha(output/'model.pt'),'elapsed_seconds':perf_counter()-started,'test_used':False}
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report


def run(config,output):
    if output.exists():raise FileExistsError('Preserve combined experiment')
    extra=json.loads(config.read_text());base=Path(extra['base_run']);old=json.loads((base/'models/metrics.json').read_text())
    cfg=old['protocol']['config'];meta=json.loads((base/'pairs/metadata.json').read_text());pair_hash=sha(base/'pairs/metadata.json')
    if old['test_used'] or old['protocol']['pairs_sha256']!=pair_hash:raise ValueError('Base/pair mismatch')
    for name,digest in meta['files'].items():
        if Path(name).name!=name or sha(base/'pairs'/name)!=digest:raise ValueError('Pair arrays changed')
    torch.set_num_threads(4)
    with EnhancedData(cfg['enhanced_cache']) as data:
        if meta['enhanced_data_sha256']!=sha(data.path/'metadata.json') or meta['state_sha256']!=sha(CACHE/'state.npz'):
            raise ValueError('Base data changed')
        bank=Banks(data);output.mkdir(parents=True);runs={}
        for name,r in old['runs'].items():
            if sha(base/'models'/name/'model.pt')!=r['checkpoint_sha256']:raise ValueError('Base checkpoint changed')
            shutil.copytree(base/'models'/name,output/name);runs[name]={**r,'reused_from':str(base/'models'/name)}
        pairs=[torch.from_numpy(np.load(base/'pairs'/f'train_{k}.npy')) for k in ('positive','negative')]
        pairs += [np.load(base/'pairs'/f'valid_{k}.npy') for k in ('positive','negative')]
        combined={**cfg,'objectives':cfg['objectives']+['full_mixed'],
            'cascade_methods':cfg['cascade_methods']+['A_full_mixed_s2026','B_full_mixed_s2026']}
        protocol={'config':combined,'followup':extra,'followup_config_sha256':sha(config),'source_sha256':sha(__file__),
            'base_report_sha256':sha(base/'models/metrics.json'),'pairs_sha256':pair_hash,'test_used':False}
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        for architecture in cfg['models']:
            for seed in cfg['seeds']:
                name=f'{architecture}_full_mixed_s{seed}'
                runs[name]=train(architecture,seed,cfg,extra,bank,pairs,output/name,pair_hash)
                (output/'progress.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
        (output/'metrics.json').write_text(json.dumps({'protocol':protocol,'runs':runs,'test_used':False,'default_model_changed':False},indent=2),encoding='utf-8')
        bank.xv._mmap.close()
    print('Full-exposure mixture complete; combined models:',len(runs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/pairwise_fullmix_v1.json'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config,a.out)
