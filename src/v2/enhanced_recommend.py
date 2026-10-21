"""Optional enhanced-DCN cascade; default v2 recommendation entry is unchanged."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from src.v2.recommend import Backend, connection, evaluate_valid
from src.v2.cascade import Cascade
from src.v2.features import coarse_features
from src.v2.enhanced_data import EnhancedData
from src.v2.enhanced_model import EnhancedRanker
from src.v2.train_rankers import sha


def process_memory():
    if os.name != 'nt':
        return {'available':False,'scope':'Windows working-set probe unavailable'}
    try:
        command = f'Get-Process -Id {os.getpid()} | Select-Object WorkingSet64,PeakWorkingSet64 | ConvertTo-Json -Compress'
        result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
                                capture_output=True,text=True,check=True,timeout=15)
        return {'available':True,**json.loads(result.stdout),
                'scope':'whole inference process incl Python/Torch, feature state and mapped CF pages; peak is process lifetime, not isolated model RAM'}
    except (OSError,subprocess.SubprocessError,ValueError):
        return {'available':False,'scope':'working-set probe failed; no estimate substituted'}


class EnhancedBackend(Backend):
    def __init__(self, config):
        started = perf_counter()
        self.enhanced_data = EnhancedData(config['enhanced_data'])
        try:
            base = {**config, 'fine_model':config['baseline_fine_model']}
            super().__init__(base)
            self.enhanced = EnhancedRanker(config['fine_model'], self.enhanced_data,
                                          self.coarse.metadata['feature_state_sha256'])
            if self.enhanced_data.meta['cutoff_ms'] != self.snapshot:
                raise ValueError('History and recall snapshot cutoffs differ')
            self.config = config
            self.enhanced_sha = sha(config['fine_model'])
            self.startup_ms = (perf_counter()-started)*1000
        except Exception:
            self.enhanced_data.close()
            if hasattr(self, 'itemcf_index'):
                super().close()
            raise

    def predict_fine(self, uid, ids, timestamp_ms):
        raw = np.asarray(ids,dtype=np.int64)
        users = np.full(len(ids),uid)
        x = self.state.features(users,raw,np.full(len(ids),timestamp_ms))
        fields = self.enhanced_data.fields(users,raw)
        spec = self.enhanced.metadata['spec']
        rec = np.repeat(np.asarray(self.enhanced_data.query_rec[uid])[None,:],len(ids),axis=0) if spec['rec'] else None
        search = np.repeat(np.asarray(self.enhanced_data.query_search[uid])[None,:],len(ids),axis=0) if spec['search'] else None
        return self.enhanced.predict(x,fields,rec,search)

    def run(self, uid, timestamp_ms=None, positives=None, budget=None):
        if not 0 <= uid < len(self.state.un):
            raise ValueError('User outside frozen table')
        timestamp_ms = self.snapshot if timestamp_ms is None else timestamp_ms
        if timestamp_ms < self.snapshot:
            raise ValueError('Request predates frozen snapshot')
        tick = perf_counter()
        routes = self.retrieve(uid,timestamp_ms)
        active = budget or self.budget
        authors = {}
        if active.max_per_author is not None:
            db = connection()
            try:
                ids = sorted({c.item_id for c in routes})
                for lo in range(0,len(ids),500):
                    block = ids[lo:lo+500]
                    for i,a in db.execute('SELECT item_id,author_id FROM items WHERE item_id IN ('+
                                          ','.join('?' for _ in block)+')',block):
                        if a not in (None,'','NULL','null','unknown'):
                            authors[i] = a
            finally:
                db.close()
        retrieval_ms = (perf_counter()-tick)*1000
        def coarse(user,ids,ts):
            return self.coarse.predict_selected(coarse_features(self.state,np.full(len(ids),user),np.asarray(ids)))
        categories = {c.item_id:int(self.state.categories[c.item_id]) for c in routes}
        cascade = Cascade(coarse,self.predict_fine,active,categories=categories,authors=authors)
        result = cascade.run(uid,timestamp_ms,routes,route_weights=self.config.get('route_weights'),
                             fusion_policy=self.config.get('fusion'),positives=positives)
        result.update({'request_elapsed_ms':retrieval_ms+result['total_elapsed_ms'],
                       'retrieval_elapsed_ms':retrieval_ms,'feature_snapshot_ms':self.snapshot,
                       'mode':'offline_enhanced_dcn','checkpoint_sha256':self.enhanced_sha,
                       'history_frozen':True,'historical_catalog_only':True})
        if self.itemcf_index is not None:
            result['itemcf_index_version'] = self.itemcf_index.metadata['version']
        return result

    def close(self):
        super().close()
        self.enhanced_data.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--user-id',type=int)
    group.add_argument('--evaluate-valid',action='store_true')
    p.add_argument('--max-users',type=int,default=256)
    p.add_argument('--out',type=Path)
    a = p.parse_args()
    if a.evaluate_valid and a.out is None:
        p.error('Evaluation requires --out')
    backend = EnhancedBackend(json.loads(a.config.read_text(encoding='utf-8')))
    try:
        if a.evaluate_valid:
            report = evaluate_valid(backend,a.out,a.max_users,True)
            report['process_memory'] = process_memory()
            report['enhanced_entry_source_sha256'] = sha(__file__)
            report['enhanced_data_sha256'] = sha(Path(backend.config['enhanced_data'])/'metadata.json')
            a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps({'users':report['users'],'stages':report['stages'],'request_p95_ms':report['request_p95_ms']},indent=2))
        else:
            print(json.dumps(backend.run(a.user_id),indent=2))
    finally:
        backend.close()
