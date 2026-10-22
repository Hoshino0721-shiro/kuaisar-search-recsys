import sqlite3
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.itemcf_index import IndexConfig, DAY_MS
from src.v2.temporal_route_data import prefix_index, fill_route_groups
from src.v2.temporal_route_features import WIDTH, temporal_values, route_values, standardized, recent_order
from src.v2.models import FineDCN
from src.v2.temporal_route_evaluate import cascade_lists, paired_intervals
from src.v2.cascade import Candidate, Cascade, Budget


class TemporalRouteTests(unittest.TestCase):
    def test_cached_cascade_matches_actual_executor_with_ties_and_unknown_categories(self):
        items=np.arange(10,22)
        coarse=np.array([2,7,3,6,0,5,10,1])
        p=np.array([.8,.5,.9,.9,.6,.8,.7,.8,.5,.4,.6,.4],dtype=np.float32)
        categories=np.array([0,1,1,1,2,2,3,3,4,4,5,5])
        budget=Budget(recall=12,coarse=8,fine=5,output=3,category_penalty=.1)
        c={int(i):-1000. for i in items}
        c.update({int(items[j]):-r for r,j in enumerate(coarse)})
        f=dict(zip(items,p))
        executor=Cascade(lambda u,ids,t:[c[i] for i in ids],lambda u,ids,t:[f[i] for i in ids],
                         budget,categories=dict(zip(items,categories)))
        actual=executor.run(1,0,[Candidate(int(i),'pool',r,0) for r,i in enumerate(items,1)])
        fine,without,rerank=cascade_lists(items,p,coarse,categories,budget.__dict__)
        self.assertEqual(fine,actual['stages'][2]['items'])
        self.assertEqual(rerank,actual['items'])
        self.assertEqual(without,fine[:3])

    def test_bootstrap_ignores_no_positive_users_for_ndcg_and_is_deterministic(self):
        a=paired_intervals([.1,.2,np.nan],[1,2,0],[10,10,0],draws=50)
        b=paired_intervals([.1,.2,np.nan],[1,2,0],[10,10,0],draws=50)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a['ndcg_difference'],.15)
        self.assertAlmostEqual(a['event_recall_difference'],.15)

    def test_temporal_counts_match_independent_formula_and_recent_ties(self):
        n1=np.array([0,10,20]);p1=np.array([0,4,4])
        n3=np.array([0,30,40]);p3=np.array([0,10,12]);total=np.array([0,100,50])
        x=temporal_values([1,2],n1,p1,n3,p3,total,.5)
        self.assertAlmostEqual(float(x[0,2]),14/30,places=6)
        self.assertAlmostEqual(float(x[0,5]),20/50,places=6)
        self.assertAlmostEqual(float(x[0,6]),4/101,places=6)
        self.assertAlmostEqual(float(x[0,7]),np.log(5)-np.log(4),places=6)
        self.assertEqual(recent_order(p1,2),[1,2])
        self.assertTrue(np.isfinite(temporal_values([0],n1,p1,n3,p3,total,.5)).all())
        with self.assertRaises(ValueError):
            temporal_values([1],n1,p1,n1,np.array([0,2,2]),total,.5)

    def test_route_features_are_actual_one_based_ranks_and_empty_is_not_negative_label(self):
        x=route_values([10,20,30,40],[20,10],[30,10],[2.,1.])
        np.testing.assert_array_equal(x[:,:2],[[1,1],[1,0],[0,1],[0,0]])
        self.assertAlmostEqual(float(x[0,2]),60/62,places=6)
        self.assertAlmostEqual(float(x[0,3]),60/62,places=6)
        self.assertAlmostEqual(float(x[0,4]),np.log(2),places=6)
        np.testing.assert_array_equal(x[:,5],x[:,2]+x[:,3])
        target=np.empty((2,6),dtype=np.float32)
        fill_route_groups(np.array([99,99]),np.array([10,40]),target,[10],None,1000)
        self.assertEqual(target[0,0],1)
        self.assertEqual(target[1].sum(),0)

    def test_prefix_graph_excludes_current_and_future_clicks_and_catalog_ids(self):
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE clean_events(user_id,item_id,timestamp_ms,label,scene,split)')
        rows=[(1,10,DAY_MS-2,1,'rec','train'),(1,20,DAY_MS-1,1,'rec','train'),
              (2,10,DAY_MS-2,1,'rec','train'),(2,20,DAY_MS-1,1,'rec','train'),
              (1,30,DAY_MS,1,'rec','train'),(2,40,DAY_MS+1,1,'rec','train'),
              (3,50,DAY_MS-1,1,'rec','valid')]
        db.executemany('INSERT INTO clean_events VALUES (?,?,?,?,?,?)',rows)
        cfg=IndexConfig(neighbors=2,max_user_items=10,seed_items=2,min_cooccurrence=2,
                        user_iuf=False,popularity_alpha=.25,query_half_life_days=None)
        with tempfile.TemporaryDirectory() as temp:
            a=prefix_index(db,DAY_MS,cfg,Path(temp)/'a','fixture',3*DAY_MS,Path(temp)/'unused')
            try:
                self.assertEqual(a.items.tolist(),[10,20])
                self.assertTrue(np.all(a.seed_times<DAY_MS))
                before=a.query(1,10)
            finally:
                a.close()
            db.execute("UPDATE clean_events SET item_id=999,label=0 WHERE timestamp_ms>=?",(DAY_MS,))
            b=prefix_index(db,DAY_MS,cfg,Path(temp)/'b','fixture',3*DAY_MS,Path(temp)/'unused')
            try:
                after=b.query(1,10)
                self.assertEqual(before['items'],after['items'])
                self.assertEqual(before['scores'],after['scores'])
            finally:
                b.close()
        db.close()

    def test_disabled_groups_do_not_change_shared_model_predictions(self):
        rng=np.random.default_rng(2026)
        x=rng.normal(size=(8,WIDTH)).astype(np.float32)
        mean=np.zeros(WIDTH,dtype=np.float32);std=np.ones(WIDTH,dtype=np.float32)
        changed=x.copy();changed[:,20:]+=100
        torch.manual_seed(2026);model=FineDCN(WIDTH)
        with torch.inference_mode():
            a=model(torch.from_numpy(standardized(x,mean,std,False,False)))
            b=model(torch.from_numpy(standardized(changed,mean,std,False,False)))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(standardized(x,mean,std,True,False)[:,28:].sum(),0)
        self.assertEqual(standardized(x,mean,std,False,True)[:,20:28].sum(),0)
