from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
from src.itemcf_index import IndexConfig, build, read_snapshot
from src.itemcf_evaluate import evaluate, merged
from src.v2.cascade import Budget, Candidate, Cascade


class ItemCFEvaluationTests(unittest.TestCase):
    def test_fusion_comparator_matches_actual_cascade_recall(self):
        popular, recent, cf = [1, 2], [2, 3], [3, 4]
        weights = {'popular': 1., 'recent': 2., 'itemcf': 1.}
        base = {}
        routes = []
        for name, items in [('popular', popular), ('recent', recent), ('itemcf', cf)]:
            for rank, iid in enumerate(items, 1):
                routes.append(Candidate(iid, name, rank, 0))
                if name != 'itemcf':
                    base[iid] = base.get(iid, 0) + weights[name] / (60 + rank)
        dummy = lambda u, ids, t: [0] * len(ids)
        reference = Cascade(dummy, dummy, Budget(4, 4, 4, 1)).run(1, 100, routes,
                    route_weights=weights)['stages'][0]['items']
        self.assertEqual(merged(cf, base, weights, 4), reference)

    def test_standalone_evaluation_handles_empty_positive_group_and_saves_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = sqlite3.connect(root / 'dataset.sqlite')
            db.execute('CREATE TABLE clean_events(user_id INTEGER,item_id INTEGER,timestamp_ms INTEGER,label INTEGER,scene TEXT,split TEXT)')
            db.executemany('INSERT INTO clean_events VALUES (?,?,?,?,?,?)',
                [(1, 10, 90, 1, 'rec', 'train'), (1, 20, 80, 1, 'rec', 'train'),
                 (2, 20, 90, 1, 'rec', 'train'), (2, 30, 80, 1, 'rec', 'train'),
                 (1, 30, 110, 1, 'rec', 'valid'), (9, 10, 120, 0, 'rec', 'valid'),
                 (1, 999, 210, 1, 'rec', 'test')])
            db.commit()
            db.close()
            (root / 'manifest.json').write_text(json.dumps({'status': 'ready', 'cutoffs_ms': [100, 200]}))
            cfg = IndexConfig(neighbors=3, max_user_items=3, seed_items=3)
            specs = {'indexes': {'fixture': str(root / 'index')},
                     'variants': {'flat': {'index': 'fixture'},
                                  'decay': {'index': 'fixture', 'half_life_days': 1}},
                     'contrasts': [['decay', 'flat']]}
            with redirect_stdout(io.StringIO()):
                build(read_snapshot(root, cfg), cfg, root / 'index')
                report = evaluate(None, None, root, root / 'report', 0, 1, specs, None)
            self.assertEqual(report['groups']['all']['positive_events'], 1)
            self.assertEqual(report['metrics']['all/flat']['at_k']['1000']['hit_events'], 1)
            self.assertIsNone(report['metrics']['no_train_clicks/flat']['at_k']['1000']['event_recall'])
            self.assertFalse(report['test_used'])
            self.assertIsNone(report['legacy_report'])
            with np.load(root / 'report/paired_hits.npz') as audit:
                self.assertEqual(int(audit['flat'][:, 1].sum()), 1)


if __name__ == '__main__':
    unittest.main()
