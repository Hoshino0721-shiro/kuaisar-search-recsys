"""Full identical-pool evaluation of learned representation/interest models."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.ctr_features import CACHE
from src.v2.full_exposure_control import evaluate_one
from src.v2.ranking_convergence import frozen_assets, write_json
from src.v2.representation_model import RepresentationScorer, InterestResidual
from src.v2.representation_interest import FrozenContent
from src.v2.representation_train import load_data
from src.v2.route_attribution import normalize, load_labels
from src.v2.temporal_route_train import verify_cache
from src.v2.train_rankers import sha


class InterestScorer:
    def __init__(self, checkpoint, content_checkpoint, vectors, data):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.meta = saved['metadata']
        self.model = InterestResidual(self.meta['dimension'], self.meta['attention'])
        self.model.load_state_dict(saved['state_dict']); self.model.eval()
        self.frozen = FrozenContent(content_checkpoint, data, vectors)
        self.data = data

    def logits(self, numeric, users, items):
        result = []
        base = self.frozen
        with torch.inference_mode():
            for lo in range(0, len(items), 8192):
                sl = slice(lo,lo+8192)
                x = torch.from_numpy(normalize(numeric[sl], base.scorer.mean, base.scorer.std, list(range(28,34))))
                u = torch.tensor(np.asarray(users[sl]),dtype=torch.long); i = torch.tensor(np.asarray(items[sl]),dtype=torch.long)
                histories = self.data.query_rec[u].long()
                result.append(self.model(x, base.logits(x,u,i), base.vectors[i],
                    base.vectors[histories], histories.ne(0)).numpy())
        return np.concatenate(result)

    def predict(self, numeric, users, items):
        return torch.sigmoid(torch.from_numpy(self.logits(numeric, users, items))).numpy()


def run(mode, cache, target, interest, reference, previous, output):
    if output.exists(): raise FileExistsError('Preserve evaluation')
    torch.set_num_threads(4)
    data = load_data(cache); meta = verify_cache(reference)
    models = target if mode == 'target' else interest
    report = json.loads((models/'metrics.json').read_text())
    if report['status'] != 'complete' or report['test_used']: raise ValueError('Complete non-test models required')
    protocol = {'mode': mode, 'cache_sha256': sha(cache/'metadata.json'), 'models_sha256': sha(models/'metrics.json'),
                'source_sha256': sha(__file__), 'candidate_sha256': meta['candidate_sha256'],
                'state_sha256': sha(CACHE/'state.npz'), 'test_used': False, 'default_changed': False}
    output.mkdir(parents=True); write_json(output/'protocol.json',protocol)
    np.save(output/'pool_item_seen_sample.npy', np.load(previous/'pool_item_seen_sample.npy'))
    users = np.load(reference/'pool_users.npy'); items = np.load(reference/'pool_items.npy',mmap_mode='r')
    labels,positive,_,_ = load_labels(users,items)
    flat_users = np.repeat(users,items.shape[1]); flat_items = np.asarray(items).reshape(-1)
    xv = np.load(reference/'valid_X.npy',mmap_mode='r')
    uv = np.load(CACHE/'valid_users.npy'); iv = np.load(CACHE/'valid_items.npy')
    records = {}
    for name, training in report['runs'].items():
        checkpoint=models/name/'model.pt'
        if sha(checkpoint)!=training['checkpoint_sha256']:raise ValueError('Model checkpoint changed')
        if mode=='target': scorer=RepresentationScorer(checkpoint,data)
        else:
            seed=training['metadata']['seed']
            vectors=torch.from_numpy(np.load(interest/'frozen'/f's{seed}'/'items.npy'))
            scorer=InterestScorer(checkpoint,target/f'content_s{seed}'/'model.pt',vectors,data)
        p=scorer.predict(xv,uv,iv); np.save(output/(name+'_valid.npy'),p)
        class BoundPool:
            def predict(self, numeric):
                if len(numeric)!=len(flat_items):raise ValueError('Expected complete frozen pool')
                return scorer.predict(numeric,flat_users,flat_items)
        records[name],_=evaluate_one(name,BoundPool(),training,reference,output,labels,positive,meta['config'])
        write_json(output/'progress.json',records)
    if sha(__file__)!=protocol['source_sha256']:raise ValueError('Evaluator changed during run')
    frozen_assets()
    write_json(output/'metrics.json',{'protocol':protocol,'runs':records,'status':'complete',
               'test_used':False,'default_changed':False})
    xv._mmap.close();items._mmap.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();base=Path('artifacts/representation_20261022')
    p.add_argument('--mode',choices=['target','interest'],required=True)
    p.add_argument('--cache',type=Path,default=base/'cache');p.add_argument('--target',type=Path,default=base/'models_safe')
    p.add_argument('--interest',type=Path,default=base/'interest_safe');p.add_argument('--reference',type=Path,default=Path('artifacts/full_exposure_local_20261022/reference'))
    p.add_argument('--previous',type=Path,default=Path('artifacts/full_exposure_local_20261022/study'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.mode,a.cache,a.target,a.interest,a.reference,a.previous,a.out)
