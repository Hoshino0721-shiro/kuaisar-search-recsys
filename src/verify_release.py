"""Post-test verification only. Never changes selected model or parameters."""
import json
from collections import Counter,defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
from src.final_evaluate import OUT,freeze
from src.ctr_features import CACHE,connection
from src.retrieval_pipeline import CandidateScorer


def main():
    frozen=freeze();report=json.loads((OUT/'metrics.json').read_text())
    y=np.load(CACHE/'test_y.npy');p=np.load(OUT/'ctr_predictions.npy')
    selected=report['ctr_exposure_metrics'][frozen['ranker']]
    assert len(y)==1015181 and len(y)==len(p)
    assert abs(roc_auc_score(y,p)-selected['auc'])<1e-10
    assert abs(log_loss(y,p)-selected['logloss'])<1e-10
    x=np.load(CACHE/'test_X.npy',mmap_mode='r')
    reloaded=CandidateScorer(frozen['ranker']).predict(x[:4096])
    assert np.allclose(reloaded,p[:4096],atol=1e-6)
    positive=defaultdict(Counter);users=set();db=connection()
    for u,i,label in db.execute("SELECT user_id,item_id,label FROM events WHERE scene='rec' AND split='test' AND exclusion IS NULL"):
        users.add(u)
        if label:positive[u][i]+=1
    db.close()
    checked={}
    for name,value in report['recommendation_list_metrics'].items():
        observed=set();hit=0
        for line in (OUT/f'{name}_candidates.jsonl').open():
            row=json.loads(line);u=row['user_id'];items=row['items']
            assert u not in observed;observed.add(u);assert len(items)==len(set(items))==100
            pool=set(items);hit+=sum(n for i,n in positive[u].items() if i in pool)
        assert observed==users and hit==value['hits_at_100']
        checked[name]={'users':len(observed),'independent_hits_at_100':hit}
    audit={'status':'passed','frozen_asset_hashes':'verified','test_ctr_rows':len(y),'checkpoint_reloaded_predictions':'matched within 1e-6',
           'independent_ctr_auc':float(roc_auc_score(y,p)),'independent_ctr_logloss':float(log_loss(y,p)),
           'candidate_checks':checked,'test_used_to_change_configuration':False}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
