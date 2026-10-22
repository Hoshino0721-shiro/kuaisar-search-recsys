from collections import Counter
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.ctr_features import FeatureState
from src.v2 import recommend
from src.v2.cascade import Budget, Candidate
from src.v2.rrf_output import recommend_rrf


class DirectRRFTests(unittest.TestCase):
    def test_fusion_deduplicates_and_filters_future_without_rankers(self):
        routes = [Candidate(10,'recent',1,0)]*5 + [Candidate(20,'recent',2,0),
                  Candidate(20,'itemcf',1,0),Candidate(5,'itemcf',2,100)]
        result = recommend_rrf(1,0,routes,Budget(4,3,2,2),positives=Counter({20:2,10:1,5:4}))
        self.assertEqual(result['items'],[20,10])
        self.assertEqual(result['stages'][-1]['hit_events'],3)
        self.assertEqual(result['rejected'],{'not_yet_available':1})
        self.assertEqual(result['candidate_sources']['20'],['itemcf','recent'])
        self.assertEqual([s['stage'] for s in result['stages']],['recall','output'])
        with self.assertRaises(ValueError):
            recommend_rrf(1,0,routes,Budget(4,3,2,2,category_penalty=.01))

    def test_real_backend_needs_no_checkpoint_and_evaluates_direct_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=root/'cache';cache.mkdir()
            state=FeatureState(np.array([0,1,2,2]),4)
            state.update(np.array([1,1,2]),np.array([1,2,2]),np.ones(3))
            state.save(cache/'state.npz')
            (root/'manifest.json').write_text(json.dumps({'cutoffs_ms':[100,200]}))
            dbpath=root/'dataset.sqlite'
            db=sqlite3.connect(dbpath)
            db.execute('CREATE TABLE clean_events(user_id,item_id,timestamp_ms,label,scene,split)')
            db.executemany('INSERT INTO clean_events VALUES (?,?,?,?,?,?)',[
                (1,1,10,1,'rec','train'),(1,2,20,1,'rec','train'),(2,2,30,1,'rec','train'),
                (1,2,110,1,'rec','valid'),(3,1,120,1,'rec','valid')])
            db.commit();db.close()
            cfg={'output_mode':'rrf','route_k':20,'route_limits':{'popular':0,'recent':20,'itemcf':0},
                 'budget':{'recall':20,'coarse':20,'fine':20,'output':20},'fusion':{'method':'rrf'}}
            with patch.multiple(recommend,CACHE=cache,DATA=root,connection=lambda:sqlite3.connect(dbpath)), \
                    patch.object(recommend,'Ranker',side_effect=AssertionError('Checkpoint loaded')):
                backend=recommend.Backend(cfg)
                try:
                    self.assertIsNone(backend.coarse);self.assertIsNone(backend.fine)
                    result=backend.run(3)
                    self.assertEqual(result['items'],[2,1])
                    self.assertEqual(result['mode'],'offline_v2_rrf')
                    self.assertFalse(result['model_weights_loaded'])
                    report=recommend.evaluate_valid(backend,root/'report.json',0)
                    self.assertEqual(report['hit_events_at20'],2)
                    self.assertEqual(report['users'],2)
                    self.assertEqual(report['event_recall_at20'],1.)
                    with self.assertRaises(FileExistsError):
                        recommend.evaluate_valid(backend,root/'report.json',0)
                    with self.assertRaises(ValueError):
                        backend.run(1,99)
                finally:
                    backend.close()
            with self.assertRaisesRegex(ValueError,'Direct RRF requires'):
                recommend.Backend({**cfg,'fusion':{'method':'quota'}})


if __name__=='__main__':
    unittest.main()
