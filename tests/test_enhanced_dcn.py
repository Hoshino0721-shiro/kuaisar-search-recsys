from contextlib import redirect_stdout
from collections import deque
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.ctr_features import FeatureState, FEATURES
from src.v2.enhanced_data import prepare, EnhancedData, history_array
from src.v2.enhanced_model import EnhancedDCN, masked_mean, spec_for


class EnhancedDCNTests(unittest.TestCase):
    def test_masked_mean_empty_and_padding(self):
        values = torch.tensor([[[99., 99.], [2., 4.], [4., 8.]], [[99., 99.]] * 3])
        ids = torch.tensor([[0, 2, 3], [0, 0, 0]])
        result, counts = masked_mean(values, ids)
        torch.testing.assert_close(result, torch.tensor([[3., 6.], [0., 0.]]))
        self.assertEqual(counts.tolist(), [2, 0])

    def test_repeat_events_and_all_variant_gradients(self):
        self.assertEqual(history_array(deque([2, 2, 3]), 4).tolist(), [0, 2, 2, 3])
        cardinalities = dict(user=8, item=8, author=8, category=8)
        for name in ['A', 'A_capacity', 'B', 'C', 'user_only', 'item_only', 'D', 'E']:
            spec = spec_for(name, spec_for('C')) if name in ('D', 'E') else spec_for(name)
            model = EnhancedDCN(spec, cardinalities)
            fields = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
            tokens = torch.tensor([[0, 2, 2], [0, 0, 0]])
            output = model(torch.zeros(2, 20), fields, tokens, tokens)
            self.assertEqual(output.shape, (2,))
            output.sum().backward()
            for table in model.tables.values():
                if table.weight.grad is not None:
                    self.assertTrue(table.weight.grad.is_sparse)
                    self.assertNotIn(0, table.weight.grad.coalesce().indices()[0].tolist())

    def test_train_only_vocab_day_lag_repeats_and_frozen_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'numeric'
            cache.mkdir()
            start, day = 1684684800000, 86400000
            (root/'manifest.json').write_text(json.dumps({'status': 'ready', 'cutoffs_ms': [start+2*day]}))
            categories = np.zeros(100,dtype=np.int16)
            categories[1:4] = [1,1,2]
            state = FeatureState(categories,3)
            state.update(np.array([1,1,1]), np.array([1,1,2]), np.array([1,1,0]))
            state.save(cache/'state.npz')
            for name, a in [('train_users', [1, 1, 1]), ('train_items', [1, 1, 2]),
                            ('valid_users', [1, 2]), ('valid_items', [3, 3])]:
                np.save(cache/(name+'.npy'), a)
            db = sqlite3.connect(root/'dataset.sqlite')
            db.executescript('CREATE TABLE items(item_id PRIMARY KEY,author_id,category_id); '
                             'CREATE TABLE clean_events(id PRIMARY KEY,user_id,item_id,timestamp_ms,label,scene,split);')
            db.executemany('INSERT INTO items VALUES (?,?,?)', [(1, 'a', '6'), (2, 'a', '6'), (3, 'new', '9')])
            db.executemany('INSERT INTO clean_events VALUES (?,?,?,?,?,?,?)', [
                (1, 1, 1, start+1, 1, 'rec', 'train'), (2, 1, 1, start+2, 1, 'rec', 'train'),
                (3, 1, 99, start+3, 1, 'search', 'train'),
                (6, 1, 99, start+4, 1, 'search', 'train'),
                (4, 1, 2, start+day+1, 0, 'rec', 'train'),
                (5, 1, 3, start+2*day+1, 1, 'search', 'valid')])
            db.commit()
            db.close()
            with redirect_stdout(io.StringIO()):
                prepare(root/'enhanced', sample_rows=99, minimum=2, length=3, cache=cache, data=root)
            data = EnhancedData(root/'enhanced')
            self.assertEqual(data.context_rec[data.train_context[0]].tolist(), [0, 0, 0])
            self.assertEqual(data.context_rec[data.train_context[2]].tolist(), [0, 2, 2])
            self.assertEqual(data.context_search[data.train_context[2]].tolist(), [0, 3, 3])
            self.assertEqual(data.query_search[1].tolist(), [0, 0, 0])
            self.assertEqual(data.valid_fields[1].tolist(), [1, 1, 1, 1])
            self.assertEqual(data.fields([1], [1])[0].tolist(), data.train_fields[0].tolist())
            # Full synthetic exposure training -> checkpoint reload -> live cascade.
            rng = np.random.default_rng(8)
            for name, value in [('train_X',rng.normal(size=(3,20)).astype(np.float32)),
                                ('train_y',np.array([1,1,0],dtype=np.int8)),
                                ('valid_X',rng.normal(size=(2,20)).astype(np.float32)),
                                ('valid_y',np.array([1,0],dtype=np.int8))]:
                np.save(cache/(name+'.npy'),value)
            (cache/'ready.json').write_text(json.dumps({'features':FEATURES}))
            from src.v2 import enhanced_train, recommend
            from src.v2.train_rankers import train as baseline_train
            from src.v2.enhanced_recommend import EnhancedBackend
            config = {'epochs':1,'batch_size':2,'learning_rate':.001,'weight_decay':1e-5,
                      'selection':'minimum validation LogLoss','minimum_frequency':2}
            with patch.object(enhanced_train,'CACHE',cache), redirect_stdout(io.StringIO()):
                banks = enhanced_train.Banks(data,cache)
                trained = enhanced_train.train('E',spec_for('E',spec_for('C')),2026,config,banks,root/'new_model')
                for kind in ('coarse_lr','fine_dcn'):
                    baseline_train(kind,cache,root/kind,3,1,2,2026)
            cfg = {'coarse_model':str(root/'coarse_lr/model.pt'),'fine_model':str(root/'new_model/model.pt'),
                   'baseline_fine_model':str(root/'fine_dcn/model.pt'),'enhanced_data':str(root/'enhanced'),
                   'route_k':3,'budget':{'recall':20,'coarse':20,'fine':20,'output':20}}
            with patch.multiple(recommend,CACHE=cache,DATA=root,connection=lambda:sqlite3.connect(root/'dataset.sqlite')):
                backend = EnhancedBackend(cfg)
                try:
                    result = backend.run(1,positives={1:1})
                    self.assertEqual(result['mode'],'offline_enhanced_dcn')
                    self.assertEqual(len(result['stages']),4)
                    self.assertEqual(result['items'],[1])
                    with self.assertRaises(ValueError):
                        backend.run(1,timestamp_ms=start)
                finally:
                    backend.close()
            banks.xv._mmap.close()
            data.close()
            with redirect_stdout(io.StringIO()):
                small = prepare(root/'small', sample_rows=1, minimum=2, length=3, cache=cache, data=root)
            self.assertEqual({k:small['cardinalities'][k] for k in ('user','author','category')},dict(user=2,author=2,category=2))
            with EnhancedData(root/'small') as small_data:
                used = set(small_data.train_fields[:,1].tolist()) | set(small_data.context_rec.ravel().tolist()) | set(small_data.context_search.ravel().tolist())
                self.assertTrue(set(range(2,small['cardinalities']['item'])).issubset(used))
            with self.assertRaises(FileExistsError):
                prepare(root/'enhanced', cache=cache, data=root)
            (root/'enhanced/vocab.json').write_text('{}')
            with self.assertRaises(ValueError):
                EnhancedData(root/'enhanced')

    def test_model_specific_untrained_token_falls_back_to_unk(self):
        model = EnhancedDCN(spec_for('C'), dict(user=8,item=8,author=8,category=8))
        model.set_coverage({name:[0,1,2] for name in model.tables})
        fields = torch.tensor([[7,7,7,7]])
        fallback = torch.ones_like(fields)
        torch.testing.assert_close(model(torch.zeros(1,20),fields), model(torch.zeros(1,20),fallback))


if __name__ == '__main__':
    unittest.main()
