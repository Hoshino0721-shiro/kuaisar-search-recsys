"""Small IVF-flat implementation: exact vectors, approximate coarse search.

Reports ANN overlap with exact retrieval, not recommendation task Recall.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
from src.history_tower import HistoryTower
from src.retrieval_pipeline import read_candidates


def main():
    out=Path('experiments/ann_valid')
    if out.exists():raise FileExistsError('Preserve existing ANN experiment')
    out.mkdir(parents=True);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved=torch.load('experiments/r2_history_attributes_valid/model.pt',map_location='cpu',weights_only=False)
    model=HistoryTower(len(saved['user_ids']),len(saved['item_ids']),saved['state_dict']['categories'],saved['state_dict']['types'])
    model.load_state_dict(saved['state_dict']);model.to(device).eval()
    mapping={u:i for i,u in enumerate(saved['user_ids'])}
    users=sorted(u for u in read_candidates('experiments/r2_history_attributes_valid/candidates.jsonl') if u in mapping)[:256]
    synchronize=lambda:torch.cuda.synchronize() if device.type=='cuda' else None
    with torch.inference_mode():
        vectors=torch.cat([model.items(torch.arange(lo,min(lo+32768,len(saved['item_ids'])+1),device=device)) for lo in range(1,len(saved['item_ids'])+1,32768)])
        idx=torch.tensor([mapping[u] for u in users],device=device)
        queries=model.users(idx,saved['final_histories'][torch.tensor([mapping[u] for u in users])].to(device))
        sample=np.random.default_rng(2026).choice(len(vectors),50000,replace=False)
        with threadpool_limits(limits=8):
            km=MiniBatchKMeans(n_clusters=64,batch_size=4096,max_iter=100,n_init=3,random_state=2026).fit(vectors[sample].cpu().numpy())
        centers=torch.tensor(km.cluster_centers_,device=device)
        def coarse(q):return q@centers.T-.5*(centers**2).sum(dim=1)
        labels=[]
        for lo in range(0,len(vectors),32768):labels.append(coarse(vectors[lo:lo+32768]).argmax(dim=1).cpu().numpy())
        labels=np.concatenate(labels);ordered=np.argsort(labels,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(labels,minlength=64))]
        np.savez_compressed(out/'ivf_index.npz',centers=km.cluster_centers_,order=ordered,offset=offset,item_ids=np.array(saved['item_ids']))
        exact=[];exact_times=[]
        # Warm up both dense and gather operations before timed queries.
        torch.topk(queries[:1]@vectors.T,100);synchronize()
        for q in queries:
            synchronize();tick=time.perf_counter();top=torch.topk(vectors@q,100).indices.cpu().numpy();synchronize()
            exact_times.append((time.perf_counter()-tick)*1000);exact.append(set(top.tolist()))
        results=[]
        for nprobe in (1,4,8,16,32):
            overlaps=[];times=[];sizes=[]
            for j,q in enumerate(queries):
                synchronize();tick=time.perf_counter()
                clusters=torch.topk(coarse(q),nprobe).indices.cpu().tolist()
                candidate=np.concatenate([ordered[offset[c]:offset[c+1]] for c in clusters])
                indices=torch.tensor(candidate,device=device)
                local=torch.topk(vectors[indices]@q,min(100,len(candidate))).indices
                returned=indices[local].cpu().tolist();synchronize()
                times.append((time.perf_counter()-tick)*1000);sizes.append(len(candidate));overlaps.append(len(exact[j]&set(returned))/100)
            results.append({'nprobe':nprobe,'ann_recall_at_100_vs_exact':float(np.mean(overlaps)),
                            'mean_scored_items':float(np.mean(sizes)),'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95))})
            print(results[-1],flush=True)
    acceptable=[r for r in results if r['ann_recall_at_100_vs_exact']>=.95 and r['p95_ms']<np.percentile(exact_times,95)]
    report={'protocol':'first 256 known validation users; frozen train embeddings; GPU query-by-query synchronized wall time; no task labels',
            'device':str(device),'queries':len(users),'items':len(vectors),'coarse_clusters':64,'numpy_seed':2026,
            'exact_p95_ms':float(np.percentile(exact_times,95)),'exact_p50_ms':float(np.percentile(exact_times,50)),
            'embedding_bytes':vectors.numel()*vectors.element_size(),'index_file_bytes':(out/'ivf_index.npz').stat().st_size,
            'variants':results,'selected_backend':min(acceptable,key=lambda r:r['p95_ms']) if acceptable else 'exact',
            'final_pipeline_backend':'exact; ANN remains an efficiency experiment and does not change final candidate protocol'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
