import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.ctr_features import FeatureState
from src.v2.cascade import Budget, Candidate
from src.v2 import ranking_diagnostics as diagnostics
from src.v2.ranking_diagnostics import compare_pool


class RankingDiagnosticsTests(unittest.TestCase):
    def test_same_pool_and_cutoff_make_coarse_loss_visible(self):
        ids = list(range(50))
        # LR prefers low IDs; DCN prefers high IDs.
        metrics, retention = compare_pool(ids, [-i for i in ids], ids, {}, {},
                                          {49: 3, 48: 1}, budgets=(20, 30, 40))
        self.assertEqual(metrics['dcn_direct']['hit_events'], 4)
        self.assertEqual(metrics['lr20_dcn']['hit_events'], 0)
        self.assertEqual(retention['20']['dcn_top20_retained'], 0)
        self.assertEqual(retention['40']['dcn_top20_retained'], 10)
        self.assertEqual(metrics['oracle_binary_in_pool']['user_period_ndcg_at_20'], 1.)

    def test_missing_route_ids_are_not_invented(self):
        metrics, _ = compare_pool([1, 2], [0., 0.], [0., 0.], {99: 0}, {}, {99: 2})
        self.assertTrue(all(m['hit_events'] == 0 for m in metrics.values()))
        self.assertTrue(all(m['positive_events'] == 2 for m in metrics.values()))

    def test_empty_pool_and_no_positives(self):
        metrics, retention = compare_pool([], [], [], {}, {}, {})
        self.assertIsNone(metrics['dcn_direct']['observed_event_recall'])
        self.assertIsNone(metrics['dcn_direct']['user_period_ndcg_at_20'])
        self.assertEqual(retention['400']['dcn_top20_count'], 0)

    def test_duplicate_pool_and_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            compare_pool([1, 1], [0., 1.], [0., 1.], {}, {}, {})
        with self.assertRaises(ValueError):
            compare_pool([1], [float('nan')], [0.], {}, {}, {})
        with self.assertRaises(ValueError):
            compare_pool([1], [], [0.], {}, {}, {})

    def test_report_freezes_pool_and_does_not_read_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dbpath = root / 'fixture.sqlite'
            db = sqlite3.connect(dbpath)
            db.execute('CREATE TABLE clean_events(user_id,item_id,label,scene,split)')
            db.executemany('INSERT INTO clean_events VALUES (?,?,?,?,?)', [
                (1, 1, 1, 'rec', 'valid'), (1, 1, 1, 'rec', 'valid'),
                (2, 2, 0, 'rec', 'valid'), (1, 2, 1, 'rec', 'test')])
            db.commit()
            db.close()
            state = FeatureState(np.array([0, 1, 1]), 3)
            state.update(np.array([1]), np.array([1]), np.array([1]))
            backend = SimpleNamespace(
                state=state, snapshot=100, recent=[2, 1], popular=[1, 2],
                budget=Budget(recall=20, coarse=20, fine=20, output=20),
                config={'coarse_model': 'fake', 'fine_model': 'fake'},
                coarse=SimpleNamespace(predict_selected=lambda x: np.arange(len(x))),
                fine=SimpleNamespace(predict=lambda x: -np.arange(len(x))),
                retrieve=lambda u, t: [Candidate(1, 'recent', 2, 100),
                                       Candidate(2, 'recent', 1, 100),
                                       Candidate(999, 'recent', 1, 101)])
            with patch.object(diagnostics, 'connection', lambda: sqlite3.connect(dbpath)), \
                    patch.object(diagnostics, 'sha', return_value='fixture'):
                output = root / 'report'
                report = diagnostics.evaluate(backend, output, 0)
                self.assertEqual(report['groups']['all']['positive_events'], 2)
                self.assertEqual(report['groups']['warm']['users'], 1)
                self.assertEqual(report['groups']['cold']['users'], 1)
                self.assertFalse(report['test_used'])
                contents = (output / 'candidates.jsonl').read_bytes()
                self.assertEqual(report['candidate_sha256'], hashlib.sha256(contents).hexdigest())
                self.assertEqual(json.loads(contents.splitlines()[0])['items'], [2, 1])
                with self.assertRaises(FileExistsError):
                    diagnostics.evaluate(backend, output, 0)


if __name__ == '__main__':
    unittest.main()
