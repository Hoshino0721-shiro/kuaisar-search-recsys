import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.v2.representation_data import parse_caption, padded, vocab
from src.v2.representation_model import RepresentationDCN, InterestResidual
from src.v2.representation_train import fit
from src.v2.ranking_convergence import ExposureDiagnostics


class RepresentationTests(unittest.TestCase):
    def test_word_zero_is_not_pad_and_training_frequency_vocab(self):
        self.assertEqual(parse_caption('[0, 9, 10]', 2), [0, 9])
        self.assertEqual(parse_caption('[]', 8), [])
        self.assertEqual(vocab({0: 20, 9: 10, 10: 1}, 5), {0: 2, 9: 3})
        with self.assertRaises(ValueError): parse_caption('[-1]', 8)
        with self.assertRaises(ValueError): parse_caption('[true]', 8)
        np.testing.assert_array_equal(padded([2, 3], 4), [0, 0, 2, 3])

    def test_sparse_embeddings_and_padding(self):
        cards = dict(user=8, item=9, author=8, category=5, word=10)
        for content in (False, True):
            torch.manual_seed(1)
            model = RepresentationDCN(cards, content, 16, id_dropout=0)
            p = model(torch.zeros(3, 34), torch.tensor([2, 3, 1]),
                      torch.tensor([[2, 0, 3], [3, 4, 2], [1, 0, 0]]), torch.tensor([[2, 0], [1, 3], [0, 0]]))
            p.sum().backward()
            for name, table in model.tables.items():
                torch.testing.assert_close(table.weight[0], torch.zeros_like(table.weight[0]))
                if table.weight.grad is not None: self.assertTrue(table.weight.grad.is_sparse)

    def test_attention_empty_history_and_zero_residual_initialization(self):
        torch.manual_seed(7)
        candidate = torch.randn(3, 16)
        history = torch.randn(3, 4, 16)
        mask = torch.tensor([[0,0,0,0], [0,1,1,0], [1,1,1,1]], dtype=torch.bool)
        for attention in (False, True):
            model = InterestResidual(16, attention)
            vector = model.interest(candidate, history, mask)
            torch.testing.assert_close(vector[0], torch.zeros(16))
            base = torch.randn(3)
            torch.testing.assert_close(model(torch.randn(3,34), base, candidate, history, mask), base)
            self.assertTrue(torch.isfinite(vector).all())

    def test_unfitted_history_identity_backoff_and_frozen_coverage(self):
        model=RepresentationDCN(dict(user=8,item=9,author=8,category=5,word=10),True,16,id_dropout=0)
        model.train()
        model(torch.zeros(1,34),torch.tensor([2]),torch.tensor([[2,3,2]]),torch.tensor([[4,0]]))
        model.eval()
        self.assertTrue(model.seen_item[2]);self.assertFalse(model.seen_item[7])
        before=model.seen_item.clone()
        torch.testing.assert_close(model.embed('item',torch.tensor([7])),model.tables['item'](torch.tensor([1])))
        torch.testing.assert_close(model.embed('word',torch.tensor([8])),model.tables['word'](torch.tensor([1])))
        torch.testing.assert_close(model.seen_item,before)

    def test_dense_sparse_resume_matches_uninterrupted(self):
        torch.set_num_threads(1)
        cfg = json.loads(Path('configs/representation_v1.json').read_text())
        cfg.update(max_epochs=5, min_epochs=3, batch_size=8, early_patience=99)
        rng = np.random.default_rng(3)
        x = torch.from_numpy(rng.normal(size=(32, 34)).astype(np.float32))
        y = np.tile([0,1], 16).astype(np.int8)
        users = torch.tensor(np.repeat(np.arange(2,6), 8))
        items = torch.tensor(np.tile(np.arange(2,10), 4))
        diag = ExposureDiagnostics(y, users.numpy(), np.repeat(np.arange(16), 2), items.numpy())
        data = SimpleNamespace(meta={'cardinalities':dict(user=8,item=12,author=6,category=6,word=12)},
            user_lookup=torch.arange(8), item_fields=torch.tensor([[0,0,0], [1,1,1]]+[[j,2+j%3,2+j%3] for j in range(2,12)]),
            caption=torch.tensor([[0,0], [1,0]]+[[j,2+j%3] for j in range(2,12)]))
        def train(dest, **kwargs):
            with redirect_stdout(io.StringIO()):
                return fit(True, 2026, cfg, data, x, y, users, items, x, users, items, diag,
                           np.zeros(34,np.float32), np.ones(34,np.float32), dest, {'fixture':True}, **kwargs)
        with TemporaryDirectory(dir='artifacts') as folder:
            root = Path(folder)
            full = train(root/'full')
            train(root/'resumed', stop_after=2)
            resumed = train(root/'resumed', resume=True)
            self.assertEqual(full['history'], resumed['history'])
            for name in ('last.pt','model.pt'):
                a = torch.load(root/'full'/name, weights_only=True)
                b = torch.load(root/'resumed'/name, weights_only=True)
                for k in a['state_dict']: self.assertTrue(torch.equal(a['state_dict'][k], b['state_dict'][k]))
            self.assertEqual(full['selected_epoch'], int(np.argmin([r['valid']['logloss'] for r in full['history']]))+1)

    def test_interest_training_resume_and_epoch_zero_selection(self):
        from src.v2.representation_interest import fit as interest_fit
        torch.set_num_threads(1)
        rng = np.random.default_rng(24)
        cfg = json.loads(Path('configs/representation_v1.json').read_text())
        cfg.update(max_epochs=5, min_epochs=3, batch_size=8, early_patience=99)
        x = torch.from_numpy(rng.normal(size=(24,34)).astype(np.float32))
        y = np.tile([0,1],12).astype(np.int8)
        c = torch.from_numpy(rng.normal(size=(24,16)).astype(np.float32))
        h = torch.from_numpy(rng.normal(size=(6,10,16)).astype(np.float32))
        mask = torch.ones(6,10,dtype=torch.bool); mask[0] = False
        contexts = torch.arange(24)%6
        base = torch.zeros(24)
        diag = ExposureDiagnostics(y,np.repeat(np.arange(6),4),np.repeat(np.arange(12),2),np.arange(24))
        def train(dest, **kwargs):
            with redirect_stdout(io.StringIO()):
                return interest_fit(True,2026,cfg,x,y,c,h,mask,contexts,base,
                    x,c,h,mask,contexts,base,diag,dest,{'fixture':True},**kwargs)
        with TemporaryDirectory(dir='artifacts') as folder:
            root=Path(folder); a=train(root/'full');train(root/'resume',stop_after=2)
            b=train(root/'resume',resume=True)
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(a['selected_epoch'],int(np.argmin([r['valid']['logloss'] for r in a['history']])))
            p=torch.load(root/'full'/'model.pt',weights_only=True)
            q=torch.load(root/'resume'/'model.pt',weights_only=True)
            for k in p['state_dict']:self.assertTrue(torch.equal(p['state_dict'][k],q['state_dict'][k]))


if __name__ == '__main__': unittest.main()
