import copy
from contextlib import redirect_stdout
from contextlib import closing
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.v2.tower_closure import fit, population, retrieve, load_training, query_arrays
from src.v2.tower_data import prepare, DAY
from src.v2.tower_model import FeatureTwoTower
from test_feature_tower import source_fixture


class TowerClosureTests(unittest.TestCase):
    def test_missing_attributes_and_temporal_prefix(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cutoff = source_fixture(root)
            with closing(sqlite3.connect(root/'dataset.sqlite')) as db:
                db.execute("UPDATE items SET author_id='0' WHERE item_id=10")
                db.execute("INSERT INTO clean_events(user_id,item_id,timestamp_ms,label,scene,split,result_type) "
                           "VALUES (1,40,?,1,'rec','train','VIDEO')", (cutoff-1,))
                db.commit()
            with redirect_stdout(io.StringIO()):
                meta = prepare(root, root/'prefix', 3, 2, cutoff-DAY)
            self.assertEqual(meta['cutoff_ms'], cutoff-DAY)
            items = np.load(root/'prefix/items.npy')
            self.assertNotIn(40, items)
            self.assertEqual(np.load(root/'prefix/item_fields.npy')[np.searchsorted(items,10)+2,0],0)
            self.assertLess(np.load(root/'prefix/times.npy').max(),cutoff-DAY)
            users, positive = population(root,cutoff-DAY)
            self.assertEqual(positive[1][40],1)
            with self.assertRaises(ValueError): population(root,cutoff+DAY)

    def test_dedup_preserves_occurrence_dropout_and_gradients(self):
        torch.manual_seed(3)
        card = {'user':5,'item':8,'author':6,'category':4,'type':3}
        fields = torch.tensor([[0,0,0],[1,1,1]]+[[2,2,2],[3,2,2]]*3)
        cfg = {'dimension':8,'safe_embeddings':True,'id_dropout':.5}
        model = FeatureTwoTower(card,cfg,fields)
        fast = copy.deepcopy(model); fast.config = dict(cfg,deduplicate_history_compute=True)
        history = torch.tensor([[0,2,3,2,4],[2,2,3,5,0]])
        torch.manual_seed(71); a,_ = model._pool(history,True)
        torch.manual_seed(71); b,_ = fast._pool(history,True)
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
        a.sum().backward(); b.sum().backward()
        for pa,pb in zip(model.parameters(),fast.parameters()):
            if pa.grad is not None:
                ga = pa.grad.to_dense() if pa.grad.is_sparse else pa.grad
                gb = pb.grad.to_dense() if pb.grad.is_sparse else pb.grad
                torch.testing.assert_close(ga,gb,atol=2e-6,rtol=2e-5)

    def test_safe_input_support_no_eval_mutation_or_random_unseen_ids(self):
        card = {'user':5,'item':6,'author':5,'category':4,'type':3}
        fields = [[0,0,0],[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
        model = FeatureTwoTower(card,{'dimension':8,'safe_embeddings':True,'id_dropout':0.},fields)
        model.items(torch.tensor([2]))
        self.assertTrue(model.seen_item_embedding[2]); self.assertFalse(model.seen_item_embedding[3])
        model.eval(); before = model.seen_item_embedding.clone()
        out = model.items(torch.tensor([3,4]))
        torch.testing.assert_close(out[0],out[1],atol=1e-7,rtol=1e-6)
        self.assertTrue(torch.equal(before,model.seen_item_embedding))
        for table in model.embeddings.values(): self.assertEqual(table.weight[0].abs().sum().item(),0)

    def test_epoch_resume_is_bitwise_and_selector_uses_exact_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source_fixture(root)
            with redirect_stdout(io.StringIO()): prepare(root,root/'prepared',3,2)
            cfg = json.loads(Path('configs/tower_closure.json').read_text())
            cfg.update(seed=2026,use_history=True,batch_size=4,dimension=8,validation_k=3,
                       max_epochs=3,min_epochs=1,probe_events=4,export_batch_size=2)
            with redirect_stdout(io.StringIO()):
                fit(root/'prepared',cfg,root/'straight',root)
                fit(root/'prepared',cfg,root/'resumed',root,stop_after=1)
                fit(root/'prepared',cfg,root/'resumed',root,resume=True)
            a = torch.load(root/'straight/last.pt',weights_only=True)
            b = torch.load(root/'resumed/last.pt',weights_only=True)
            for key in a['state_dict']:
                self.assertTrue(torch.equal(a['state_dict'][key],b['state_dict'][key]),key)
                self.assertTrue(torch.equal(a['best_state'][key],b['best_state'][key]),key)
            self.assertEqual([r['validation']['hit_events'] for r in a['history']],
                             [r['validation']['hit_events'] for r in b['history']])
            cfg['temperature'] = .2
            with self.assertRaises(ValueError): fit(root/'prepared',cfg,root/'resumed',root,resume=True)
            _, arrays = load_training(root/'prepared')
            self.assertEqual(query_arrays(np.array([12345]),arrays)[0].item(),1)
        vectors = np.array([[1,0],[1,0],[-1,0]],np.float32)
        ids = retrieve(vectors,np.array([[1,0]],np.float32),np.array([10,20,30]),3,block=1)
        self.assertEqual(ids.tolist(),[[10,20,30]])

    def test_full_evaluation_and_independent_audit_on_fixture(self):
        import sqlite3
        from src.itemcf_index import IndexConfig,read_snapshot,build
        from src.v2.train_tower import export
        from src.v2.tower_closure_evaluate import prepare_baseline,evaluate
        from scripts.audit_tower_closure import check_training,check_metrics,aligned_training
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cutoff=source_fixture(root)
            # More than 1000 catalog items exercise real Top1000 truncation.
            with closing(sqlite3.connect(root/'dataset.sqlite')) as db:
                db.execute("UPDATE clean_events SET timestamp_ms=? WHERE split='test'",(cutoff+DAY+1000,))
                db.executemany("INSERT INTO items VALUES (?,?,?,?)",[(i,'100','1','NORMAL') for i in range(1000,2000)])
                db.executemany("INSERT INTO clean_events(user_id,item_id,timestamp_ms,label,scene,split,result_type) "
                               "VALUES (2,?,?,1,'rec','train','VIDEO')",[(i,cutoff-2*DAY+6000+i) for i in range(1000,2000)])
                db.commit()
            cfg=json.loads(Path('configs/tower_closure.json').read_text())
            cfg.update(seed=2026,use_history=True,dimension=8,max_epochs=1,min_epochs=1,probe_events=16)
            model=root/'models/fixture'
            with redirect_stdout(io.StringIO()):
                prepare(root,root/'data',3,2)
                fit(root/'data',cfg,model,root)
                export(root/'data',model/'model.pt',model/'index','cpu')
                icfg=IndexConfig(neighbors=300,max_user_items=30,seed_items=30)
                build(read_snapshot(root,icfg),icfg,root/'cf')
                prepare_baseline(root,root/'cf',root/'baseline')
                result=evaluate(root/'data',model,root/'baseline',root/'evaluation/fixture',root)
                self.assertEqual(result,evaluate(root/'data',model,root/'baseline',root/'evaluation/fixture',root))
                checked=check_training(model,root/'data',root)
                metrics=check_metrics(root,{'name':'fixture','window':'main'},root)
                aligned=aligned_training(root,root/'data')
            self.assertEqual(checked['epochs'],1)
            self.assertEqual(metrics['all_users_recomputed'],3)
            self.assertEqual(aligned['all_training_rows_aligned'],1006)
            self.assertEqual(result['baseline']['hit_events_at_20'],1)

    def test_public_verifier_checks_tampering_without_requiring_private_assets(self):
        from scripts.verify_closure import verify
        from src.itemcf_index import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); public=root/'public.txt'
            public.write_text('fixture',encoding='utf-8')
            (root/'release.json').write_text(json.dumps({'status':'frozen','schema_version':1,
                'public_assets':{str(public):digest(public)},'local_assets':{str(root/'missing.npy'):'private'}}))
            self.assertEqual(verify(root,public_only=True)['checked_assets'],1)
            public.write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError): verify(root,public_only=True)

    def test_study_lock_rejects_duplicate_and_releases(self):
        from scripts.run_tower_closure import study_lock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with study_lock(root):
                with self.assertRaises(RuntimeError):
                    with study_lock(root): pass
            with study_lock(root): pass

    def test_independent_v2_reproduction_does_not_require_v1_private_weights(self):
        from src.v2.closure_integrity import verify_v1_archive
        from src.itemcf_index import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'archived.py';source.write_text('# fixture\n')
            freeze=root/'freeze.json'
            freeze.write_text(json.dumps({'asset_hashes':{str(source):digest(source),str(root/'private.pt'):'missing'}}))
            result=verify_v1_archive(freeze)
            self.assertFalse(result['complete']);self.assertEqual(result['verified_assets'],1)
            source.write_text('# changed\n')
            with self.assertRaises(ValueError): verify_v1_archive(freeze)


if __name__ == '__main__': unittest.main()
