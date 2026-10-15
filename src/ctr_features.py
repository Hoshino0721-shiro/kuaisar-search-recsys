"""Daily lagged CTR features. Evaluation uses a frozen training snapshot."""
import json
import sqlite3
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np

DATA=Path('data/processed/kuaisar_small_v1')
CACHE=DATA/'ctr_features_v1'
FEATURES=['global_ctr','user_ctr','item_ctr','category_ctr','user_category_ctr',
          'log_user_count','log_item_count','log_category_count','log_user_category_count',
          'long_category_share','seen_item','seen_user','hour_sin','hour_cos','weekday_sin','weekday_cos',
          'log_search_user_clicks','log_search_category_clicks','short_search_category_share','has_search_clicks']
TZ=timezone(timedelta(hours=8))


def connection():
    return sqlite3.connect((DATA/'dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)


class FeatureState:
    def __init__(self,categories,user_size):
        self.categories=categories
        self.nc=int(categories.max())+1
        self.un=np.zeros(user_size);self.up=np.zeros(user_size)
        self.inum=np.zeros(len(categories));self.ip=np.zeros(len(categories))
        self.cn=np.zeros(self.nc);self.cp=np.zeros(self.nc)
        self.ucn=np.zeros(user_size*self.nc);self.ucp=np.zeros(user_size*self.nc)
        self.su=np.zeros(user_size);self.suc=np.zeros(user_size*self.nc)

    def features(self,u,i,t):
        c=self.categories[i];uc=u*self.nc+c
        prior=float(self.up.sum()/max(1,self.un.sum())) if self.un.sum() else .5
        smooth=lambda p,n,a:(p+a*prior)/(n+a)
        hour=((t//1000+8*3600)%86400)/3600
        weekday=((t//1000+8*3600)//86400+3)%7
        values=[np.full(len(u),prior),smooth(self.up[u],self.un[u],20),smooth(self.ip[i],self.inum[i],20),
                smooth(self.cp[c],self.cn[c],100),smooth(self.ucp[uc],self.ucn[uc],20),
                np.log1p(self.un[u]),np.log1p(self.inum[i]),np.log1p(self.cn[c]),np.log1p(self.ucn[uc]),
                (self.ucp[uc]+1)/(self.up[u]+self.nc),self.inum[i]>0,self.un[u]>0,
                np.sin(hour*np.pi/12),np.cos(hour*np.pi/12),np.sin(weekday*2*np.pi/7),np.cos(weekday*2*np.pi/7),
                np.log1p(self.su[u]),np.log1p(self.suc[uc]),(self.suc[uc]+1)/(self.su[u]+self.nc),self.su[u]>0]
        result=np.column_stack(values).astype(np.float32)
        if not np.isfinite(result).all():raise ValueError('Nonfinite features')
        return result

    def update(self,u,i,y):
        c=self.categories[i];uc=u*self.nc+c
        for counts,positive,keys in [(self.un,self.up,u),(self.inum,self.ip,i),(self.cn,self.cp,c),(self.ucn,self.ucp,uc)]:
            counts+=np.bincount(keys,minlength=len(counts))
            positive+=np.bincount(keys,weights=y,minlength=len(counts))

    def update_search(self,u,i):
        c=self.categories[i];known=c>0
        # Missing/unknown item categories are retained in source but not guessed as intent.
        self.su=np.bincount(u[known],minlength=len(self.su)).astype(float)
        self.suc=np.bincount(u[known]*self.nc+c[known],minlength=len(self.suc)).astype(float)

    def save(self,path):
        np.savez_compressed(path,**self.__dict__)

    @classmethod
    def load(cls,path):
        data=np.load(path)
        state=cls(data['categories'],len(data['un']))
        for key in data.files:
            setattr(state,key,int(data[key]) if key=='nc' else data[key])
        return state


def fetch(db,scene,start,end):
    rows=db.execute("SELECT user_id,item_id,timestamp_ms,label FROM events WHERE exclusion IS NULL AND scene=? AND timestamp_ms>=? AND timestamp_ms<? ORDER BY id",(scene,start,end)).fetchall()
    return np.array(rows,dtype=np.int64).reshape(-1,4)


def write_eval(split):
    if split not in ('valid','test'):raise ValueError(split)
    db=connection();state=FeatureState.load(CACHE/'state.npz')
    a,b=json.loads((DATA/'manifest.json').read_text())['cutoffs_ms']
    rows=fetch(db,'rec',a if split=='valid' else b,b if split=='valid' else 2**62)
    for name,values in [('X',state.features(rows[:,0],rows[:,1],rows[:,2])),('y',rows[:,3].astype(np.int8)),
                        ('users',rows[:,0].astype(np.int32)),('items',rows[:,1].astype(np.int32)),('times',rows[:,2])]:
        np.save(CACHE/f'{split}_{name}.npy',values)
    db.close();print('Prepared',split,len(rows),flush=True)


def main():
    if (CACHE/'ready.json').exists():
        print('Feature cache already complete');return
    CACHE.mkdir(parents=True,exist_ok=True)
    db=connection()
    max_item=db.execute('SELECT max(item_id) FROM events').fetchone()[0]
    max_user=db.execute('SELECT max(user_id) FROM users').fetchone()[0]
    raw=db.execute('SELECT DISTINCT category_id FROM items').fetchall()
    vocab={c:n+1 for n,(c,) in enumerate(sorted(raw)) if c not in (None,'','0')}
    categories=np.zeros(max_item+1,dtype=np.int16)
    for i,c in db.execute('SELECT item_id,category_id FROM items'):
        categories[i]=vocab.get(c,0)
    state=FeatureState(categories,max_user+1)
    n=db.execute("SELECT count(*) FROM events WHERE scene='rec' AND split='train' AND exclusion IS NULL").fetchone()[0]
    arrays={name:np.lib.format.open_memmap(CACHE/f'train_{name}.npy',mode='w+',dtype=dtype,shape=shape)
            for name,dtype,shape in [('X','float32',(n,len(FEATURES))),('y','int8',(n,)),('users','int32',(n,)),('items','int32',(n,))]}
    cursor=0
    # All observations of a day update counters only after that day's feature extraction.
    day=datetime(2023,5,22,tzinfo=TZ)
    while day<datetime(2023,5,29,tzinfo=TZ):
        start=int(day.timestamp()*1000);end=int((day+timedelta(days=1)).timestamp()*1000)
        rows=fetch(db,'rec',start,end);size=len(rows)
        arrays['X'][cursor:cursor+size]=state.features(rows[:,0],rows[:,1],rows[:,2])
        for name,col in [('users',0),('items',1),('y',3)]:arrays[name][cursor:cursor+size]=rows[:,col]
        state.update(rows[:,0],rows[:,1],rows[:,3])
        search=fetch(db,'search',start,end);positive=search[search[:,3]>0]
        state.update_search(positive[:,0],positive[:,1])
        cursor+=size;print(day.date(),size,'rows, cumulative',cursor,flush=True)
        day+=timedelta(days=1)
    assert cursor==n
    for array in arrays.values():array.flush()
    state.save(CACHE/'state.npz');db.close()
    write_eval('valid')
    info={'features':FEATURES,'train_rows':n,'train_stats_rule':'prior completed calendar days only; first day empty history',
          'search_rule':'previous completed Beijing calendar day clicked-category counts; unknown category excluded from intent',
          'evaluation_rule':'frozen train-end counters and May28 search snapshot; no valid/test updates','test_materialized':False}
    (CACHE/'ready.json').write_text(json.dumps(info,indent=2),encoding='utf-8')


if __name__=='__main__':main()
