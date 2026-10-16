"""Small generated SQLite fixtures verify integration, never business quality."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.ctr_features import FEATURES, FeatureState
from src.v2 import recommend
from src.v2.train_rankers import train
from src.itemcf_index import IndexConfig, build, read_snapshot


class BackendTests(unittest.TestCase):
    def test_real_models_sparse_retrieval_and_validation_report_connect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            cache.mkdir()
            dbpath = root / 'dataset.sqlite'
            db = sqlite3.connect(dbpath)
            db.executescript('''
                CREATE TABLE clean_events(id INTEGER PRIMARY KEY,user_id INTEGER,item_id INTEGER,
                  timestamp_ms INTEGER,label INTEGER,scene TEXT,split TEXT);
                CREATE TABLE items(item_id INTEGER PRIMARY KEY,author_id TEXT);
            ''')
            events = [(1, 1, 10, 1, 'train'), (1, 2, 20, 1, 'train'),
                      (2, 2, 30, 1, 'train'), (2, 3, 40, 1, 'train'),
                      (1, 3, 110, 1, 'valid'), (4, 1, 120, 0, 'valid')]
            db.executemany('INSERT INTO clean_events(user_id,item_id,timestamp_ms,label,scene,split) VALUES (?,?,?,?,\'rec\',?)', events)
            db.executemany('INSERT INTO items VALUES (?,?)', [(1, 'a'), (2, 'a'), (3, 'b')])
            db.commit()
            db.close()
            (root / 'manifest.json').write_text(json.dumps({'status': 'ready', 'cutoffs_ms': [100, 200]}), encoding='utf-8')
            state = FeatureState(np.array([0, 1, 1, 2]), 5)
            state.update(np.array([1, 1, 2, 2]), np.array([1, 2, 2, 3]), np.ones(4))
            state.save(cache / 'state.npz')
            (cache / 'ready.json').write_text(json.dumps({'features': FEATURES}), encoding='utf-8')
            rng = np.random.default_rng(9)
            for split in ('train', 'valid'):
                np.save(cache / f'{split}_X.npy', rng.normal(size=(16, 20)).astype(np.float32))
                np.save(cache / f'{split}_y.npy', np.tile([0, 1], 8).astype(np.int8))
            with redirect_stdout(io.StringIO()):
                for kind in ('coarse_lr', 'fine_dcn'):
                    train(kind, cache, root / kind, 16, 1, 16, 2026)
            config = {'coarse_model': str(root / 'coarse_lr/model.pt'),
                      'fine_model': str(root / 'fine_dcn/model.pt'), 'route_k': 3,
                      'budget': {'recall': 20, 'coarse': 20, 'fine': 20, 'output': 20,
                                 'max_per_author': 1}}
            connect = lambda: sqlite3.connect(dbpath)
            with patch.multiple(recommend, CACHE=cache, DATA=root, connection=connect):
                backend = recommend.Backend(config)
                self.assertEqual(backend.retrieve(4)[-1].source, 'recent')
                result = backend.run(1)
                self.assertLessEqual(len(result['items']), 2)
                self.assertEqual([s['stage'] for s in result['stages']], ['recall', 'coarse', 'fine', 'rerank'])
                with self.assertRaises(ValueError):
                    backend.run(1, timestamp_ms=99)
                report = recommend.evaluate_valid(backend, root / 'report.json', 0, True)
                self.assertEqual(report['users'], 2)
                self.assertIn('without_coarse', report['baselines'])
                self.assertFalse(report['test_used'])
                cfg = IndexConfig(neighbors=3, max_user_items=3, seed_items=3)
                with redirect_stdout(io.StringIO()):
                    build(read_snapshot(root, cfg), cfg, root / 'itemcf')
                indexed = recommend.Backend({**config, 'itemcf_index': str(root / 'itemcf')})
                try:
                    # Proves index-backed startup skips rebuilding the user-item matrix.
                    self.assertFalse(hasattr(indexed, 'normalized'))
                    self.assertTrue(indexed.itemcf_index.has_user(1))
                    self.assertEqual([c for c in indexed.retrieve(4) if c.source == 'itemcf'], [])
                    output = indexed.run(1)
                    self.assertEqual(output['itemcf_index_version'], indexed.itemcf_index.metadata['version'])
                    self.assertTrue(output['items'])
                finally:
                    indexed.close()


if __name__ == '__main__':
    unittest.main()
