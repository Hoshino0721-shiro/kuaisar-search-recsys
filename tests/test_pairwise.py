import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import torch
from src.v2.pairwise_data import build_pairs
from src.v2.pairwise_train import objective_loss, pair_metrics, train_one
from src.v2.train_rankers import sha


class PairwiseTests(TestCase):
    def test_pairs_do_not_cross_users_bins_or_same_items(self):
        u=np.array([1,1,1,1,2,2,1]);i=np.array([10,11,10,12,10,13,14])
        t=np.array([1,2,3,101,1,2,100]);y=np.array([1,0,1,0,0,1,1])
        p,n,r=build_pairs(u,i,t,y,100)
        np.testing.assert_array_equal(u[p],u[n]);np.testing.assert_array_equal(t[p]//100,t[n]//100)
        self.assertTrue(np.all(i[p]!=i[n]));self.assertTrue(np.all(y[p]==1));self.assertTrue(np.all(y[n]==0))
        self.assertEqual(r['pairs'],3);self.assertEqual(r['consistent_duplicate_rows_collapsed'],1)
        p2,n2,_=build_pairs(u,i,t,y,100)
        np.testing.assert_array_equal(p,p2);np.testing.assert_array_equal(n,n2)

    def test_conflicts_single_label_groups_and_bad_labels(self):
        p,n,r=build_pairs([1]*4,[2,2,3,4],[1,2,3,4],[0,1,1,0],10)
        self.assertEqual(r['contradictory_item_rows_excluded'],2)
        self.assertEqual(p.tolist(),[2]);self.assertEqual(n.tolist(),[3])
        self.assertEqual(len(build_pairs([1,2],[2,3],[1,1],[1,0],10)[0]),0)
        empty=build_pairs([],[],[],[],10)
        self.assertEqual(len(empty[0]),0);self.assertEqual(empty[2]['pseudo_contexts'],0)
        with self.assertRaises(ValueError):build_pairs([1],[2],[1],[2])
        with self.assertRaises(ValueError):build_pairs([1],[2],[1],[1],0)

    def test_loss_gradient_offset_and_mixture(self):
        p=torch.tensor([.2,.4],requires_grad=True);n=torch.tensor([-.1,.1],requires_grad=True)
        loss=objective_loss(p,n,'pairwise');loss.backward()
        self.assertTrue((p.grad<0).all());self.assertTrue((n.grad>0).all())
        torch.testing.assert_close(loss,objective_loss(p+10,n+10,'pairwise'))
        bce=objective_loss(p,n,'paired_bce')
        torch.testing.assert_close(objective_loss(p,n,'mixed'),.5*(loss+bce))
        q=pair_metrics(np.array([1.,1.,0.]),np.array([0,0]),np.array([1,2]))
        self.assertEqual(q['accuracy_half_ties'],.75)
        with self.assertRaises(ValueError):objective_loss(p,n,'unknown')

    def test_full_mixture_keeps_exposure_bce_and_independent_rng(self):
        from src.v2.pairwise_fullmix import full_mixed_loss
        x=torch.tensor([.2,-.3]);y=torch.tensor([1.,0.]);p=torch.tensor([.4]);n=torch.tensor([.1])
        torch.testing.assert_close(full_mixed_loss(x,y,p,n,.5),
            .5*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.5*torch.nn.functional.softplus(n-p).mean())
        torch.manual_seed(123);expected=torch.randperm(10)
        torch.manual_seed(123);generator=torch.Generator().manual_seed(999)
        torch.randint(5,(20,),generator=generator)
        torch.testing.assert_close(torch.randperm(10),expected)

    def test_pair_coverage_uses_only_repeated_endpoints_not_all_rows(self):
        from src.v2.pairwise_train import paired_coverage
        bank=SimpleNamespace(fields=torch.tensor([[0,0,2,2],[0,0,3,2],[0,0,4,3]]),
            data=SimpleNamespace(meta={'cardinalities':{'author':5,'category':4}}))
        coverage=paired_coverage(bank,{'attrs':True},torch.tensor([0]*5),torch.tensor([1]*5))
        self.assertEqual(coverage['author'].tolist(),[2,3])
        self.assertEqual(coverage['category'].tolist(),[2])
        self.assertEqual(paired_coverage(bank,{'attrs':False},torch.tensor([0]),torch.tensor([1])),{})

    def test_every_objective_trains_reloads_and_refuses_overwrite(self):
        from src.v2 import pairwise_train
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'state.npz').write_bytes(b'fixture-state')
            state_hash=sha(root/'state.npz')
            data=SimpleNamespace(path=root,meta={'cardinalities':dict(user=3,item=4,author=3,category=3),
                'feature_state_sha256':state_hash},valid_fields=np.full((4,4),2),item_train_frequency=np.array([0,0,5,0]))
            (root/'metadata.json').write_text(json.dumps(data.meta));np.save(root/'sample_indices.npy',np.arange(4))
            rng=np.random.default_rng(3);x=rng.normal(size=(4,20)).astype(np.float32)
            bank=SimpleNamespace(data=data,mean=np.zeros(20,dtype=np.float32),std=np.ones(20,dtype=np.float32),
                numeric=torch.from_numpy(x),fields=torch.full((4,4),2,dtype=torch.long),y=torch.tensor([1.,0.,1.,0.]),
                xv=x,yv=np.array([1,0,1,0]),uv=np.array([1,1,2,2]),iv=np.array([2]*4),
                coverage=lambda spec,minimum:{'author':[2],'category':[2]})
            pairs=[torch.tensor([0,2]),torch.tensor([1,3]),np.array([0,2]),np.array([1,3])]
            cfg={'epochs':1,'batch_size':2,'mixed_weight':.5,'learning_rate':.001,'weight_decay':1e-5,'selection':'fixed pair loss'}
            with patch.object(pairwise_train,'CACHE',root),redirect_stdout(io.StringIO()):
                for architecture in ('A','B'):
                    for objective in ('full_bce','paired_bce','pairwise','mixed'):
                        out=root/(architecture+objective)
                        report=train_one(architecture,objective,2026,cfg,bank,pairs,out,'fixture-pairs')
                        self.assertEqual(report['metadata']['selected_epoch'],1)
                        self.assertTrue(np.isfinite(np.load(out/'valid_logits.npy')).all())
                        self.assertEqual(report['checkpoint_sha256'],sha(out/'model.pt'))
                        with self.assertRaises(FileExistsError):train_one(architecture,objective,2026,cfg,bank,pairs,out,'fixture-pairs')
                from src.v2 import pairwise_fullmix
                with patch.object(pairwise_fullmix,'CACHE',root):
                    for architecture in ('A','B'):
                        report=pairwise_fullmix.train(architecture,2026,cfg,{'mixed_weight':.5,'pair_draw_seed_offset':10000},
                            bank,pairs,root/(architecture+'fullmix'),'fixture-pairs')
                        self.assertEqual(report['valid_pair']['pairs'],2)
                        self.assertEqual(report['metadata']['selected_epoch'],1)
