from contextlib import redirect_stdout
from datetime import datetime
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from src.ctr_features import TZ, FeatureState
from src.v2.training_cache_audit import replay


def fixture(root):
    start = int(datetime(2023, 5, 22, tzinfo=TZ).timestamp() * 1000)
    day = 86400000
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE clean_events(id INTEGER PRIMARY KEY,user_id,item_id,timestamp_ms,label,scene,split)')
    db.executemany('INSERT INTO clean_events VALUES (?,?,?,?,?,?,?)', [
        (1, 1, 1, start + 1000, 1, 'rec', 'train'),
        (2, 1, 2, start + 2000, 1, 'search', 'train'),
        (3, 1, 2, start + day + 1000, 0, 'rec', 'train'),
        (4, 1, 1, start + 2 * day + 1000, 1, 'rec', 'test')])
    state = FeatureState(np.array([0, 1, 2]), 3)
    u, first, second = np.array([1]), np.array([1]), np.array([2])
    x1 = state.features(u, first, np.array([start + 1000]))
    state.update(u, first, np.array([1]))
    state.update_search(u, second)
    x2 = state.features(u, second, np.array([start + day + 1000]))
    state.update(u, second, np.array([0]))
    state.update_search(np.array([], dtype=int), np.array([], dtype=int))
    state.save(root / 'state.npz')
    for name, values in [('X', np.vstack([x1, x2])), ('users', [1, 1]),
                          ('items', [1, 2]), ('y', [1, 0])]:
        np.save(root / f'train_{name}.npy', values)
    return db, start + 2 * day, state


class TrainingCacheAuditTests(unittest.TestCase):
    def test_daily_replay_and_search_reset_ignore_test_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, cutoff, _ = fixture(root)
            try:
                with redirect_stdout(io.StringIO()):
                    report = replay(db, root, cutoff)
                self.assertTrue(report['verified'])
                self.assertEqual(report['training_rows'], 2)
                self.assertEqual(report['days'][0]['has_search_fraction'], 0.)
                self.assertEqual(report['days'][1]['has_search_fraction'], 1.)
                self.assertTrue(all(report['final_state_exact_match'].values()))
            finally:
                db.close()

    def test_wrong_label_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, cutoff, _ = fixture(root)
            np.save(root / 'train_y.npy', [0, 1])
            try:
                with redirect_stdout(io.StringIO()):
                    report = replay(db, root, cutoff)
                self.assertFalse(report['verified'])
                self.assertFalse(report['days'][0]['alignment']['y'])
            finally:
                db.close()

    def test_current_day_feature_contamination_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, cutoff, state = fixture(root)
            x = np.load(root / 'train_X.npy')
            # Deliberately replace the first day's empty history with final state.
            x[0] = state.features(np.array([1]), np.array([1]), np.array([cutoff - 2 * 86400000 + 1000]))[0]
            np.save(root / 'train_X.npy', x)
            try:
                with redirect_stdout(io.StringIO()):
                    report = replay(db, root, cutoff)
                self.assertFalse(report['verified'])
                self.assertGreater(report['days'][0]['max_feature_abs_error'], 0.)
            finally:
                db.close()


if __name__ == '__main__':
    unittest.main()
