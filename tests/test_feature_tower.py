from contextlib import redirect_stdout
from datetime import datetime,timezone,timedelta
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
import torch
from src.itemcf_index import digest,IndexConfig,build,read_snapshot
from src.v2.tower_data import prepare,load_metadata,DAY
from src.v2.tower_model import FeatureTwoTower,in_batch_loss
from src.v2.train_tower import train,export
from src.v2.tower_retrieval import TowerIndex,evaluate


def source_fixture(root):
    day=int(datetime(2023,5,22,tzinfo=timezone(timedelta(hours=8))).timestamp()*1000)
    cutoff=day+2*DAY
    db=sqlite3.connect(root/'dataset.sqlite')
    db.executescript('''CREATE TABLE clean_events(id INTEGER PRIMARY KEY,user_id,item_id,timestamp_ms,
       label,scene,split,result_type); CREATE TABLE items(item_id,author_id,category_id,feature_type);''')
    rows=[(1,10,day+1000,1,'rec','train','VIDEO'),(1,20,day+1000,1,'rec','train','VIDEO'),
          (2,20,day+2000,1,'rec','train','VIDEO'),(2,30,day+3000,1,'rec','train','VIDEO'),
          (1,30,day+4000,1,'search','train','VIDEO'),(99,20,day+4500,1,'search','train','VIDEO'),
          (1,999,day+5000,1,'search','train','VIDEO'),(1,40,day+6000,0,'rec','train','VIDEO'),
          (1,30,day+DAY+1000,1,'rec','train','VIDEO'),(2,10,day+DAY+2000,1,'rec','train','VIDEO'),
          (99,20,day+DAY+3000,1,'search','train','VIDEO'),(1,10,day+DAY+4000,1,'search','train','VIDEO'),
          (1,999,cutoff+1000,1,'rec','valid','VIDEO'),(99,10,cutoff+2000,1,'rec','valid','VIDEO'),
          (100,20,cutoff+3000,0,'rec','valid','VIDEO'),(1,888,cutoff+4000,1,'rec','test','VIDEO')]
    db.executemany('INSERT INTO clean_events(user_id,item_id,timestamp_ms,label,scene,split,result_type) VALUES (?,?,?,?,?,?,?)',rows)
    db.executemany('INSERT INTO items VALUES (?,?,?,?)',[(10,'100','1','NORMAL'),(20,'200','2','NORMAL'),
        (30,'100','2','NORMAL'),(40,'future','future','NORMAL'),(999,'future','future','NORMAL')])
    db.commit()
    db.close()
    (root/'manifest.json').write_text(json.dumps({'status':'ready','cutoffs_ms':[cutoff,cutoff+DAY]}))
    return cutoff


class FeatureTowerTests(unittest.TestCase):
    def test_day_causal_context_sharing_search_window_and_train_only_vocab(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source_fixture(root)
            out=root/'prepared'
            with redirect_stdout(io.StringIO()):
                meta=prepare(root,out,3,2)
            self.assertEqual(meta['samples'],6)
            self.assertEqual(meta['contexts'],4)
            np.testing.assert_array_equal(np.load(out/'items.npy'),[10,20,30])
            contexts=np.load(out/'sample_contexts.npy')
            self.assertEqual(contexts[0],contexts[1])
            rec=np.load(out/'context_rec.npy')
            search=np.load(out/'context_search.npy')
            np.testing.assert_array_equal(rec[:2],np.zeros((2,3)))
            np.testing.assert_array_equal(rec[contexts[4]],[0,2,3])
            np.testing.assert_array_equal(search[contexts[4]],[0,4])
            self.assertNotIn('future',(out/'vocab.json').read_text())
            self.assertEqual(meta,load_metadata(out))
            with self.assertRaises(FileExistsError):
                prepare(root,out)

    def test_losses_deduplicate_targets_and_only_mask_available_history(self):
        q=torch.eye(2,requires_grad=True)
        c=torch.tensor([[1.,3.],[2.,4.]],requires_grad=True)
        target=torch.tensor([2,3])
        empty=torch.zeros((2,1),dtype=torch.long)
        plain=in_batch_loss(q,c,target,empty,empty,temperature=1.)
        expected=torch.nn.functional.cross_entropy(q@c.T,torch.tensor([0,1]))
        self.assertTrue(torch.allclose(plain,expected))
        # Another target of the same user is not a positive unless it is already in history.
        prior=torch.tensor([[3],[0]])
        masked=in_batch_loss(q,c,target,empty,prior,temperature=1.)
        self.assertAlmostEqual(masked.item(),torch.nn.functional.cross_entropy((q@c.T)[1:],torch.tensor([1])).item()/2,places=6)
        masked.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        repeated=torch.tensor([2,2,3])
        qq=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        zeros=torch.zeros((3,1),dtype=torch.long)
        loss=in_batch_loss(qq,c.detach(),repeated,zeros,zeros,temperature=1.)
        self.assertTrue(torch.allclose(loss,torch.nn.functional.cross_entropy(qq@c.detach().T,torch.tensor([0,0,1]))))

    def test_padding_unit_norm_attributes_and_unknown_id_content_path(self):
        torch.manual_seed(2026)
        model=FeatureTwoTower({'user':4,'item':5,'author':4,'category':4,'type':3},
                              {'dimension':8,'use_search':True},[[0,0,0],[1,1,1],[2,2,2],[3,3,2],[2,3,2]])
        model.eval()
        a=model.users(torch.tensor([2]),torch.tensor([[0,0,2]]),torch.tensor([[0,3]]))
        b=model.users(torch.tensor([2]),torch.tensor([[0,2]]),torch.tensor([[3]]))
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        self.assertTrue(torch.allclose(a.norm(dim=-1),torch.ones(1),atol=1e-6))
        self.assertTrue(torch.equal(model.items(torch.tensor([0])),torch.zeros((1,8))))
        cold=model.items(torch.tensor([1,1]),torch.tensor([[2,2,2],[3,3,2]]))
        self.assertFalse(torch.allclose(cold[0],cold[1]))

    def test_train_export_full_retrieval_and_current_cf_comparison_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cutoff=source_fixture(root)
            cfg={'dimension':8,'epochs':1,'batch_size':4,'seed':2026,'device':'cpu',
                 'use_search':True,'id_dropout':.1,'export_batch_size':2,'logq_correction':True}
            with redirect_stdout(io.StringIO()):
                prepare(root,root/'data',3,2)
                trained=train(root/'data',cfg,root/'run')
                export(root/'data',root/'run/model.pt',root/'index')
                cfconfig=IndexConfig(neighbors=3,max_user_items=3,seed_items=3)
                build(read_snapshot(root,cfconfig),cfconfig,root/'cf')
                report=evaluate(root/'index',root,root/'eval',3,2,root/'cf')
            self.assertFalse(trained['test_labels_used'])
            self.assertEqual(report['candidate_items'],3)
            self.assertEqual(report['groups']['all']['positive_events'],2)
            self.assertIsNotNone(report['current_plus_tower_at_2000'])
            index=TowerIndex(root/'index',digest(root/'manifest.json'),block_items=1,max_query_batch=1)
            try:
                result=index.query(1,3)
                self.assertEqual(set(result['items']),{10,20,30})
                self.assertEqual(len(index.query(99,3)['items']),3)  # search-only user
                self.assertEqual(index.query(100,3)['items'],[])  # no hidden popular fallback
                batched=index.query_batch([1,99,100],3)
                self.assertEqual([row['items'] for row in batched],
                                 [index.query(uid,3)['items'] for uid in [1,99,100]])
                expected=index.queries[index.user_map[1]]@index.vectors.T
                order=np.lexsort((index.items,-expected))
                self.assertEqual(result['items'],index.items[order].tolist())
                with self.assertRaises(ValueError):
                    index.query(1,1,cutoff-1)
            finally:
                index.close()
            with self.assertRaises(FileExistsError):
                export(root/'data',root/'run/model.pt',root/'index')
            np.save(root/'index/users.npy',np.asarray([1,2],dtype=np.int64))
            with self.assertRaises(ValueError):
                TowerIndex(root/'index')

    def test_exact_index_id_ties_and_negative_scores_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            values={'items':np.asarray([10,20,30],dtype=np.int64),
                    'users':np.asarray([1,2],dtype=np.int64),
                    'vectors':np.asarray([[1,0],[1,0],[-1,0]],dtype=np.float32),
                    'queries':np.asarray([[1,0],[1,0]],dtype=np.float32),
                    'query_signal':np.asarray([True,False])}
            for name,value in values.items():
                np.save(root/(name+'.npy'),value)
            meta={'status':'ready','schema_version':1,'dimension':2,'cutoff_ms':100,'version':'fixture',
                  'manifest_sha256':'fixture','files':{p.name:digest(p) for p in root.glob('*.npy')}}
            (root/'metadata.json').write_text(json.dumps(meta))
            index=TowerIndex(root,block_items=1)
            try:
                self.assertEqual(index.query(1,2)['items'],[10,20])
                self.assertEqual(index.query(1,3)['scores'],[1.,1.,-1.])
                self.assertEqual(index.query(2,3)['items'],[])
            finally:
                index.close()


if __name__=='__main__':
    unittest.main()
