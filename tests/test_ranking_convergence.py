import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.v2.ranking_convergence import ExposureDiagnostics, fit


class ConvergenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = json.loads(Path('configs/ranking_convergence.json').read_text())
        self.cfg.update(max_epochs=10, min_epochs=10, batch_size=16, early_patience=99)
        rng = np.random.default_rng(12)
        self.x = rng.normal(size=(64, 34)).astype(np.float32)
        self.y = np.tile([0, 1, 1, 0], 16).astype(np.int8)
        self.diag = ExposureDiagnostics(self.y, np.repeat(np.arange(8), 8),
                                        np.repeat(np.arange(16), 4), np.arange(64))
        self.mean = self.x.mean(0)
        self.std = self.x.std(0)

    def train(self, dest, **kwargs):
        with redirect_stdout(io.StringIO()):
            return fit(self.x, self.y, self.x, self.diag, self.mean, self.std,
                       [], 2026, self.cfg, dest, {'data': 'fixture'}, **kwargs)

    def test_metrics_ties_and_single_class_users(self):
        y = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])
        users = np.repeat(np.arange(5), 2)
        p = np.array([.2, .2, .8, .4, .1, .9, .1, .2, .3, .4])
        diag = ExposureDiagnostics(y, users, np.zeros(10), np.arange(10))
        result = diag.metrics(p)
        expected = np.mean([roc_auc_score(y[j:j+2], p[j:j+2]) for j in (0, 2, 4)])
        self.assertAlmostEqual(result['gauc'], expected)
        self.assertEqual(result['gauc_groups'], 3)
        self.assertEqual(result['same_timestamp']['mixed_groups'], 3)
        # First two groups put the negative first (item tie break for group 0).
        self.assertAlmostEqual(result['same_timestamp']['ndcg_at20'], (2/np.log2(3)+1)/3)

    def test_vector_metrics_match_independent_weighted_user_auc(self):
        rng = np.random.default_rng(55)
        users = rng.integers(0, 31, 1000)
        y = rng.integers(0, 2, 1000)
        p = rng.integers(0, 8, 1000)/8
        result = ExposureDiagnostics(y, users, np.zeros(1000), np.arange(1000)).metrics(p)
        total = weighted = 0
        for u in np.unique(users):
            mask = users == u
            if len(np.unique(y[mask])) > 1:
                weighted += mask.sum()*roc_auc_score(y[mask], p[mask])
                total += mask.sum()
        self.assertAlmostEqual(result['gauc'], weighted/total)
        self.assertAlmostEqual(result['auc'], roc_auc_score(y, p))

    def test_exact_epoch_resume_and_best_reload(self):
        with TemporaryDirectory(dir='artifacts') as folder:
            root = Path(folder)
            full = self.train(root/'full')
            partial = self.train(root/'resumed', stop_after=3)
            self.assertEqual(partial['status'], 'running')
            resumed = self.train(root/'resumed', resume=True)
            self.assertEqual(full['history'], resumed['history'])
            self.assertEqual(full['selected_epoch'], resumed['selected_epoch'])
            for name in ('last.pt', 'model.pt', 'epoch5.pt'):
                a = torch.load(root/'full'/name, weights_only=True)
                b = torch.load(root/'resumed'/name, weights_only=True)
                for key in a['state_dict']:
                    self.assertTrue(torch.equal(a['state_dict'][key], b['state_dict'][key]))
            scores = [r['valid']['logloss'] for r in full['history']]
            self.assertEqual(full['selected_epoch'], int(np.argmin(scores))+1)

    def test_resume_rejects_changed_protocol_and_preserves_directory(self):
        with TemporaryDirectory(dir='artifacts') as folder:
            dest = Path(folder)/'run'
            self.train(dest, stop_after=2)
            self.cfg['learning_rate'] *= .5
            with self.assertRaisesRegex(ValueError, 'protocol'):
                self.train(dest, resume=True)
            with self.assertRaises(FileExistsError):
                self.train(dest)

    def test_plateau_decays_lr_then_stops_and_keeps_first_best(self):
        self.cfg.update(early_patience=6, max_epochs=30)
        constant = self.diag.metrics(np.full(64, .5))
        with TemporaryDirectory(dir='artifacts') as folder, patch.object(self.diag, 'metrics', return_value=constant):
            result = self.train(Path(folder)/'run')
            self.assertEqual(result['selected_epoch'], 1)
            self.assertEqual(result['epochs_completed'], 10)
            self.assertGreaterEqual(result['lr_reductions'], 2)
            self.assertEqual(result['early_stopping_reason'], 'validation_plateau_after_lr_reductions')
            self.assertTrue(all(h['learning_rate'] == .001 for h in result['history'][:5]))

    def test_invalid_alignment_or_labels_rejected(self):
        with self.assertRaises(ValueError):
            ExposureDiagnostics([0, 2], [1, 1], [1, 1], [1, 2])
        with self.assertRaises(ValueError):
            ExposureDiagnostics([0, 1], [1], [1, 1], [1, 2])

    def test_corrupted_selected_checkpoint_rejected_on_resume(self):
        with TemporaryDirectory(dir='artifacts') as folder:
            dest = Path(folder)/'run'
            self.train(dest, stop_after=2)
            (dest/'model.pt').write_bytes(b'corrupt fixture')
            with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                self.train(dest, resume=True)

    def test_completed_resume_recovers_metrics_from_committed_checkpoint(self):
        with TemporaryDirectory(dir='artifacts') as folder:
            dest = Path(folder)/'run'
            original = self.train(dest)
            (dest/'metrics.json').write_text('{}')
            recovered = self.train(dest, resume=True)
            self.assertEqual(original, recovered)
            self.assertEqual(json.loads((dest/'metrics.json').read_text()), original)


if __name__ == '__main__':
    unittest.main()
