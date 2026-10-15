"""Local inference CLI for the frozen training snapshot."""
import argparse
import json
import numpy as np
from src.ctr_features import CACHE,FeatureState,connection
from src.final_evaluate import OUT,freeze
from src.retrieval_pipeline import train_lists,tower_for_users,make_sources,CandidateScorer,rrf,rerank


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--user-id',type=int,required=True)
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--mode',choices=['best_recall','full_pipeline'],default='best_recall')
    a=p.parse_args()
    if not 1<=a.top_k<=100:raise ValueError('top-k must be 1..100')
    if not (OUT/'metrics.json').exists():raise RuntimeError('Complete frozen final evaluation first')
    config=freeze();state=FeatureState.load(CACHE/'state.npz')
    if not 0<=a.user_id<len(state.un):raise ValueError('User ID outside anonymized dataset range')
    db=connection();popular,recent=train_lists(db,state);db.close()
    tower={}
    if a.mode=='full_pipeline' or config['retrieval'] not in ('popular','recent','rrf_equal','quota_recent'):
        tower=tower_for_users([a.user_id])
    if a.mode=='best_recall':
        ranked=make_sources([a.user_id],popular,recent,tower)[config['retrieval']][a.user_id]
    else:
        candidates=rrf([popular,recent,tower.get(a.user_id,popular[:100])],[1,1,1],300)
        features=state.features(np.full(len(candidates),a.user_id),np.array(candidates),np.full(len(candidates),config['inference_timestamp_ms']))
        scores=CandidateScorer(config['ranker']).predict(features)
        ranked=rerank(candidates,scores,state.categories,float(config['ctr_pipeline'].split('_')[-1]))
    print(json.dumps({'user_id':a.user_id,'mode':a.mode,'snapshot_ms':config['inference_timestamp_ms'],
                      'items':ranked[:a.top_k],'note':'offline frozen training snapshot'},indent=2))


if __name__=='__main__':main()
