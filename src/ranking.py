"""Validation-selected CTR models on causally constructed exposure features."""
import hashlib
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from threadpoolctl import threadpool_limits
from src.ctr_features import CACHE,FEATURES
from src.metrics import auc as grouped_auc

OUT=Path('experiments/p_rank_valid')


def metrics(y,p,users):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    order=np.argsort(users,kind='stable'); cuts=np.r_[0,np.flatnonzero(np.diff(users[order]))+1,len(users)]
    numerator=0;denominator=0;valid_groups=0
    for a,b in zip(cuts[:-1],cuts[1:]):
        index=order[a:b];ys=y[index]
        if ys.min()!=ys.max():
            numerator+=(b-a)*grouped_auc(ys.tolist(),p[index].tolist());denominator+=b-a;valid_groups+=1
    curve=[];ece=0
    bins=np.minimum((p*10).astype(int),9)
    for j in range(10):
        mask=bins==j;n=int(mask.sum())
        pred=float(p[mask].mean()) if n else None;actual=float(y[mask].mean()) if n else None
        if n:ece+=n/len(y)*abs(pred-actual)
        curve.append({'bin':j,'rows':n,'predicted':pred,'observed':actual})
    return {'rows':len(y),'auc':float(roc_auc_score(y,p)) if y.min()!=y.max() else None,
            'gauc':float(numerator/denominator) if denominator else None,'gauc_effective_rows':int(denominator),'gauc_groups':valid_groups,
            'logloss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),
            'ece_10_bins':ece,'reliability':curve}


class GateRanker(nn.Module):
    def __init__(self,gated=True):
        super().__init__();self.gated=gated
        if gated:
            self.rec=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
            self.search=nn.Sequential(nn.Linear(4,32),nn.ReLU())
            self.gate=nn.Sequential(nn.Linear(64,32),nn.Sigmoid())
            self.head=nn.Linear(32,1)
        else:self.network=nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x,has_search):
        if not self.gated:return self.network(x).squeeze(-1)
        r=self.rec(x[:,:16]);s=self.search(x[:,16:])
        return self.head(r+self.gate(torch.cat([r,s],dim=-1))*s*has_search[:,None]).squeeze(-1)


def torch_predict(model,x,scaler,device):
    model.eval();pieces=[]
    with torch.inference_mode():
        for lo in range(0,len(x),16384):
            batch=np.asarray(x[lo:lo+16384]);mask=torch.tensor(batch[:,19]>0,device=device,dtype=torch.float32)
            tensor=torch.tensor(scaler.transform(batch),device=device,dtype=torch.float32)
            pieces.append(torch.sigmoid(model(tensor,mask)).cpu().numpy())
    return np.concatenate(pieces)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summarize',action='store_true',help='Recover metrics from already saved predictions; do not retrain')
    args=parser.parse_args()
    if args.summarize:
        summarize();return
    if OUT.exists():raise FileExistsError('Preserve existing ranking run')
    OUT.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(2026)
    x=np.load(CACHE/'train_X.npy',mmap_mode='r');y=np.load(CACHE/'train_y.npy',mmap_mode='r')
    xv=np.load(CACHE/'valid_X.npy',mmap_mode='r');yv=np.load(CACHE/'valid_y.npy');uv=np.load(CACHE/'valid_users.npy')
    rng=np.random.default_rng(2026);idx=np.sort(rng.choice(len(x),min(1000000,len(x)),replace=False))
    xt=np.asarray(x[idx]);yt=np.asarray(y[idx]);scaler=StandardScaler().fit(xt)
    joblib.dump(scaler,OUT/'scaler.joblib');np.save(OUT/'sample_indices.npy',idx)
    config={'seed':2026,'fit':'one million uniformly sampled rec train exposures; no class resampling',
            'sample_rows':len(idx),'feature_names':FEATURES,'features_rule':'previous complete calendar days; frozen train snapshot at evaluation',
            'selection':'minimum valid logloss, then maximum valid AUC','test_used':False,
            'hgb':{'max_iter':120,'max_leaf_nodes':15,'learning_rate':.06,'l2_regularization':10,'min_samples_leaf':100,'max_bins':128,'early_stopping':False},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    results={};predictions={};training_times={}
    predictions['constant']=np.full(len(xv),float(yt.mean()))
    predictions['item_ctr']=xv[:,2].copy()
    with threadpool_limits(limits=8):
        start=time.perf_counter()
        lr=SGDClassifier(loss='log_loss',alpha=1e-5,max_iter=15,tol=1e-4,random_state=2026)
        lr.fit(scaler.transform(xt[:,:20]),yt)
        joblib.dump(lr,OUT/'lr.joblib')
        predictions['lr']=lr.predict_proba(scaler.transform(xv))[:,1]
        training_times['lr']=time.perf_counter()-start
        for name,cols in [('hgb_rec',16),('hgb_search',20)]:
            start=time.perf_counter();model=HistGradientBoostingClassifier(**config['hgb'],random_state=2026)
            model.fit(xt[:,:cols],yt);joblib.dump(model,OUT/f'{name}.joblib')
            predictions[name]=model.predict_proba(xv[:,:cols])[:,1]
            training_times[name]=time.perf_counter()-start
            print(name,'trained',training_times[name],flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tx=torch.tensor(scaler.transform(xt),device=device,dtype=torch.float32);ty=torch.tensor(yt,device=device,dtype=torch.float32)
    mask=torch.tensor(xt[:,19]>0,device=device,dtype=torch.float32)
    for name,gated in [('mlp_concat',False),('mlp_gate',True)]:
        torch.manual_seed(2026);model=GateRanker(gated).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        start=time.perf_counter();losses=[]
        for epoch in range(5):
            model.train();order=torch.randperm(len(tx),device=device);total_loss=0
            for lo in range(0,len(tx),4096):
                index=order[lo:lo+4096];loss=nn.functional.binary_cross_entropy_with_logits(model(tx[index],mask[index]),ty[index])
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total_loss+=loss.item()*len(index)
            losses.append(total_loss/len(tx))
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'gated':gated,'losses':losses},OUT/f'{name}.pt')
        predictions[name]=torch_predict(model,xv,scaler,device);training_times[name]=time.perf_counter()-start
        print(name,'trained',losses,flush=True)
    for name,p in predictions.items():np.save(OUT/f'{name}_predictions.npy',p)
    summarize(training_times)


def summarize(training_times=None):
    xv=np.load(CACHE/'valid_X.npy',mmap_mode='r');yv=np.load(CACHE/'valid_y.npy');uv=np.load(CACHE/'valid_users.npy')
    config=json.loads((OUT/'config.json').read_text())
    results={}
    for name in ('constant','item_ctr','lr','hgb_rec','hgb_search','mlp_concat','mlp_gate'):
        p=np.load(OUT/f'{name}_predictions.npy')
        results[name]=metrics(yv,p,uv);np.save(OUT/f'{name}_predictions.npy',p)
        results[name]['groups']={}
        for group,flag in [('has_search',xv[:,19]>0),('no_search',xv[:,19]==0),('warm_item',xv[:,10]>0),('cold_item',xv[:,10]==0)]:
            if flag.any():results[name]['groups'][group]=metrics(yv[flag],p[flag],uv[flag])
        print(name,{k:results[name][k] for k in ('auc','gauc','logloss','brier','ece_10_bins')},flush=True)
    selected=min(results,key=lambda name:(results[name]['logloss'],-results[name]['auc']))
    report={'config':config,'models':results,'training_seconds':training_times,'selected':selected,
            'summary_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'search_ablation_delta_auc':results['hgb_search']['auc']-results['hgb_rec']['auc']}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Selected',selected,flush=True)


if __name__=='__main__':main()
