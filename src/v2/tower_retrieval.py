"""Versioned frozen-query dual-tower index and full-catalog exact evaluation.

No trained release is bundled. An absent index is an error, never a substitute
random model. Exact retrieval is a correctness/quality baseline, not ANN or an
industrial latency claim. Both matrix scores and query batches are bounded.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from src.itemcf_index import DATA, digest
from src.v2.recall_allocation import load_population
from src.v2.recall_fusion import select_recall


def best(ids,scores,k):
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite retrieval scores')
    if len(ids)>k:
        threshold=np.partition(scores,-k)[-k]
        keep=scores>=threshold
        ids,scores=ids[keep],scores[keep]
    order=np.lexsort((ids,-scores))[:k]
    return ids[order],scores[order]


class TowerIndex:
    def __init__(self,path,expected_manifest_sha256=None,block_items=32768,max_query_batch=64):
        self.path=Path(path)
        if not isinstance(block_items,int) or isinstance(block_items,bool) or block_items<1:
            raise ValueError('block_items must be a positive integer')
        self.block_items=block_items
        if not isinstance(max_query_batch,int) or isinstance(max_query_batch,bool) or max_query_batch<1:
            raise ValueError('max_query_batch must be a positive integer')
        self.max_query_batch=max_query_batch
        self.metadata=json.loads((self.path/'metadata.json').read_text(encoding='utf-8'))
        m=self.metadata
        if m.get('status')!='ready' or m.get('schema_version')!=1:
            raise ValueError('Unsupported/incomplete tower index')
        if expected_manifest_sha256 and m['manifest_sha256']!=expected_manifest_sha256:
            raise ValueError('Tower/data manifest mismatch')
        required={'items.npy','users.npy','vectors.npy','queries.npy','query_signal.npy'}
        if set(m['files'])!=required:
            raise ValueError('Unexpected tower index files')
        for name,expected in m['files'].items():
            if digest(self.path/name)!=expected:
                raise ValueError('Tower index checksum mismatch: '+name)
        self._arrays=[]
        try:
            for name in ('items','users','vectors','queries','query_signal'):
                value=np.load(self.path/(name+'.npy'),mmap_mode='r')
                self._arrays.append(value)
                setattr(self,name,value)
            if (self.vectors.shape!=(len(self.items),m['dimension'])
                    or self.queries.shape!=(len(self.users),m['dimension'])
                    or self.query_signal.shape!=(len(self.users),)
                    or self.query_signal.dtype!=np.bool_
                    or self.items.dtype!=np.int64 or self.users.dtype!=np.int64
                    or not len(self.items) or np.any(np.diff(self.items)<=0)
                    or np.any(np.diff(self.users)<=0)):
                raise ValueError('Tower index shape/ID contract violated')
            for data in (self.vectors,self.queries):
                if data.dtype!=np.float32:
                    raise ValueError('Expected float32 embeddings')
                for lo in range(0,len(data),block_items):
                    block=data[lo:lo+block_items]
                    if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block,axis=1),1.,atol=1e-4):
                        raise ValueError('Expected finite unit-normalized embeddings')
            self.user_map={int(u):j for j,u in enumerate(self.users)}
        except Exception:
            self.close()
            raise

    def query_batch(self,users,k=1000,timestamp_ms=None):
        if not isinstance(k,int) or isinstance(k,bool) or k<1:
            raise ValueError('k must be a positive integer')
        if timestamp_ms is not None and timestamp_ms<self.metadata['cutoff_ms']:
            raise ValueError('Request predates frozen tower index')
        users=list(users)
        if len(users)>self.max_query_batch:
            return [result for lo in range(0,len(users),self.max_query_batch)
                    for result in self.query_batch(users[lo:lo+self.max_query_batch],k,timestamp_ms)]
        result=[{'user_id':int(u),'items':[],'scores':[],'reason':'no_usable_query',
                 'version':self.metadata['version']} for u in users]
        known=[j for j,u in enumerate(users) if u in self.user_map and self.query_signal[self.user_map[u]]]
        if not known:
            return result
        vectors=np.asarray(self.queries[[self.user_map[users[j]] for j in known]])
        top_ids=[np.empty(0,dtype=np.int64) for _ in known]
        top_scores=[np.empty(0,dtype=np.float32) for _ in known]
        with threadpool_limits(limits=4):
            for lo in range(0,len(self.items),self.block_items):
                ids=self.items[lo:lo+self.block_items]
                scores=vectors@self.vectors[lo:lo+self.block_items].T
                for row in range(len(known)):
                    top_ids[row],top_scores[row]=best(np.r_[top_ids[row],ids],np.r_[top_scores[row],scores[row]],k)
        for row,pos in enumerate(known):
            result[pos].update(items=top_ids[row].tolist(),scores=top_scores[row].tolist(),reason='ok')
        return result

    def query(self,user,k=1000,timestamp_ms=None):
        return self.query_batch([user],k,timestamp_ms)[0]

    def close(self):
        for value in getattr(self,'_arrays',[]):
            value._mmap.close()
        self._arrays=[]


def evaluate(path,data,output,k=1000,batch_users=16,itemcf_path=None):
    if output.exists():
        raise FileExistsError('Preserve existing tower evaluation')
    if k<1 or batch_users<1:
        raise ValueError('Invalid retrieval evaluation budget')
    start=perf_counter()
    index=TowerIndex(path,digest(data/'manifest.json'))
    cf=None
    try:
        cutoff=json.loads((data/'manifest.json').read_text())['cutoffs_ms'][0]
        if cutoff!=index.metadata['cutoff_ms']:
            raise ValueError('Tower cutoff differs from training cutoff')
        if itemcf_path is not None:
            from src.itemcf_index import ItemCFIndex
            cf=ItemCFIndex(itemcf_path,index.metadata['manifest_sha256'])
            if cf.metadata['cutoff_ms']!=cutoff or not np.array_equal(cf.items,index.items):
                raise ValueError('CF/tower must share cutoff and item catalog')
        users,positives,_,recent=load_population(data,cutoff,2000)
        catalog=set(index.items.tolist())
        counts=np.asarray([sum(positives[u].values()) for u in users],dtype=np.int64)
        catalog_counts=np.asarray([sum(n for i,n in positives[u].items() if i in catalog) for u in users],dtype=np.int64)
        cuts=tuple(sorted({min(100,k),min(1000,k),k}))
        hit=np.zeros((len(users),len(cuts)),dtype=np.int32)
        current_hits=np.zeros(len(users),dtype=np.int32)
        added_hits=np.zeros(len(users),dtype=np.int32)
        size=np.zeros(len(users),dtype=np.int32)
        usable=np.zeros(len(users),dtype=bool)
        cfreq=Counter()
        fused_hits=0
        for lo in range(0,len(users),batch_users):
            results=index.query_batch(users[lo:lo+batch_users],k)
            for offset,result in enumerate(results):
                row=lo+offset
                uid=users[row]
                ids=result['items']
                usable[row]=bool(ids)
                size[row]=len(ids)
                cfreq.update(ids)
                for j,cut in enumerate(cuts):
                    hit[row,j]=sum(positives[uid].get(i,0) for i in set(ids[:cut]))
                ranks={(i,'recent'):rank for rank,i in enumerate(recent,1)}
                ranks.update({(i,'tower'):rank for rank,i in enumerate(ids,1)})
                fused=select_recall(ranks,2000,{'recent':1.,'tower':1.})
                fused_hits+=sum(positives[uid].get(i,0) for i in fused)
                if cf is not None:
                    cf_ids=cf.query(uid,1000)['items']
                    base={(i,'recent'):rank for rank,i in enumerate(recent,1)}
                    base.update({(i,'itemcf'):rank for rank,i in enumerate(cf_ids,1)})
                    current=select_recall(base,2000,{'recent':1.,'itemcf':1.})
                    current_hits[row]=sum(positives[uid].get(i,0) for i in current)
                    base.update({(i,'tower'):rank for rank,i in enumerate(ids,1)})
                    added=select_recall(base,2000,{'recent':1.,'itemcf':1.,'tower':1.})
                    added_hits[row]=sum(positives[uid].get(i,0) for i in added)
            if (lo//batch_users)%64==0:
                print(f'Exact tower retrieval {min(lo+batch_users,len(users))}/{len(users)} users',flush=True)
        groups={}
        for name,mask in [('all',np.ones(len(users),dtype=bool)),('has_query',usable),('no_query',~usable)]:
            denom=int(counts[mask].sum())
            catalog_denom=int(catalog_counts[mask].sum())
            groups[name]={'users':int(mask.sum()),'positive_events':denom,
                          'catalog_positive_events':catalog_denom,
                          'mean_candidates':float(size[mask].mean()) if mask.any() else 0.,
                          'at_k':{str(cut):{'hit_events':int(hit[mask,j].sum()),
                                           'event_recall':float(hit[mask,j].sum()/denom) if denom else None,
                                           'catalog_conditional_recall':float(hit[mask,j].sum()/catalog_denom) if catalog_denom else None}
                                  for j,cut in enumerate(cuts)}}
        denom=int(counts.sum())
        baseline=sum(sum(p.get(i,0) for i in recent) for p in positives.values())
        report={'status':'evaluated','split':'valid','test_labels_used':False,
                'scope':'exact full clicked-train catalog; frozen user-period observed-positive coverage',
                'index_version':index.metadata['version'],'checkpoint_sha256':index.metadata['checkpoint_sha256'],
                'manifest_sha256':index.metadata['manifest_sha256'],'candidate_items':len(index.items),
                'catalog_positive_events':int(catalog_counts.sum()),
                'query_and_catalog_reachable_positive_events':int(catalog_counts[usable].sum()),
                'groups':groups,'distinct_returned_items':len(cfreq),
                'recent_only_at_2000':{'hit_events':baseline,'recall':baseline/denom if denom else None},
                'recent_plus_tower_at_2000':{'hit_events':fused_hits,'recall':fused_hits/denom if denom else None},
                'itemcf_index_version':cf.metadata['version'] if cf is not None else None,
                'current_recent_cf_at_2000':({'hit_events':int(current_hits.sum()),
                                             'recall':float(current_hits.sum()/denom) if denom else None} if cf is not None else None),
                'current_plus_tower_at_2000':({'hit_events':int(added_hits.sum()),
                                              'recall':float(added_hits.sum()/denom) if denom else None,
                                              'net_hit_events':int((added_hits-current_hits).sum())} if cf is not None else None),
                'elapsed_seconds':perf_counter()-start,
                'timing_scope':'offline exact index loading, batch retrieval and evaluation; not serving latency',
                'source_sha256':digest(Path(__file__))}
        output.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(output/'hits.npz',users=np.asarray(users),counts=counts,hits=hit,has_query=usable,
                            current_hits=current_hits,added_hits=added_hits)
        (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return report
    finally:
        index.close()
        if cf is not None:
            cf.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--k',type=int,default=1000)
    parser.add_argument('--batch-users',type=int,default=16)
    parser.add_argument('--itemcf-index',type=Path)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.index,args.data,args.out,args.k,args.batch_users,args.itemcf_index),indent=2))
