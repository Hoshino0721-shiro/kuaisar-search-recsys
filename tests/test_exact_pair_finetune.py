import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from src.v2.exact_pair_data import build_exact_pairs
from src.v2.exact_pair_finetune import fit_seed, objective_loss, eligible
from src.v2.pairwise_data import build_pairs
from src.v2.ranking_convergence import ExposureDiagnostics
from src.v2.route_attribution import normalize


class ExactPairTests(unittest.TestCase):
    def test_exact_time_dedup_conflicts_and_determinism(self):
        u = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2])
        i = np.array([10, 10, 11, 12, 12, 13, 14, 10, 11])
        t = np.array([5, 5, 5, 5, 5, 6, 6, 5, 5])
        y = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1])
        p, n, report = build_exact_pairs(u, i, t, y)
        self.assertEqual(report['pairs'], 3)
        self.assertEqual(report['consistent_duplicate_rows_collapsed'], 1)
        self.assertEqual(report['contradictory_item_rows_excluded'], 2)
        np.testing.assert_array_equal(u[p], u[n])
        np.testing.assert_array_equal(t[p], t[n])
        self.assertTrue(np.all(i[p] != i[n]))
        self.assertTrue(np.all(y[p] == 1) and np.all(y[n] == 0))
        self.assertNotIn(1, p)
        p2, n2, _ = build_exact_pairs(u, i, t, y)
        np.testing.assert_array_equal(p, p2)
        np.testing.assert_array_equal(n, n2)

    def test_vector_pair_builder_matches_independent_context_builder(self):
        rng = np.random.default_rng(1)
        u, i, t = [rng.integers(0, high, 1000) for high in (5, 6, 10)]
        y = rng.integers(0, 2, 1000)
        p, n, report = build_exact_pairs(u, i, t, y)
        old_p, _, old = build_pairs(u, i, t, y, window_ms=1)
        np.testing.assert_array_equal(p, old_p)
        self.assertEqual(report['usable_groups'], old['usable_contexts'])
        self.assertEqual(report['contradictory_item_rows_excluded'], old['contradictory_item_rows_excluded'])
        self.assertEqual(report['consistent_duplicate_rows_collapsed'], old['consistent_duplicate_rows_collapsed'])
        for a, b in zip(p, n):
            self.assertEqual((u[a], t[a], y[a]), (u[b], t[b], 1))
            self.assertEqual(y[b], 0)

    def test_empty_no_mixed_and_bad_labels(self):
        self.assertEqual(len(build_exact_pairs([], [], [], [])[0]), 0)
        self.assertEqual(len(build_exact_pairs([1, 1], [1, 2], [1, 2], [1, 0])[0]), 0)
        self.assertEqual(len(build_exact_pairs([1, 1], [1, 1], [1, 1], [1, 0])[0]), 0)
        with self.assertRaises(ValueError):
            build_exact_pairs([1], [1], [1], [2])

    def test_objective_and_guardrails(self):
        x = torch.tensor([.4, -.2], requires_grad=True)
        y = torch.tensor([1., 0.])
        p = torch.tensor([.1, .5], requires_grad=True)
        n = torch.tensor([.3, -.1], requires_grad=True)
        a, bce, pair = objective_loss(x, y, p, n, 'pair_aux', .2)
        torch.testing.assert_close(a, bce+.2*pair)
        a.backward()
        self.assertTrue((p.grad < 0).all() and (n.grad > 0).all())
        base = objective_loss(x, y, p, n, 'bce', .2)[0]
        torch.testing.assert_close(base, torch.nn.functional.binary_cross_entropy_with_logits(x, y))
        warm = {'logloss': .6, 'auc': .7}
        cfg = {'logloss_guard': .0005, 'auc_guard': .001}
        self.assertTrue(eligible(warm, warm, cfg))
        self.assertFalse(eligible({'logloss': .601, 'auc': .7}, warm, cfg))
        self.assertFalse(eligible({'logloss': .6, 'auc': .698}, warm, cfg))


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = json.loads(Path('configs/exact_pair_finetune.json').read_text())
        self.cfg.update(max_epochs=5, min_epochs=3, batch_size=16, pair_batch_size=4, early_patience=99)
        rng = np.random.default_rng(5)
        x = rng.normal(size=(32, 34)).astype(np.float32)
        self.x = torch.from_numpy(normalize(x, np.zeros(34, np.float32), np.ones(34, np.float32), list(range(28, 34))))
        self.y = np.tile([1, 0], 16).astype(np.int8)
        self.pairs = [np.arange(0, 32, 2), np.arange(1, 32, 2)]
        self.diag = ExposureDiagnostics(self.y, np.repeat(np.arange(8), 4),
                                        np.repeat(np.arange(16), 2), np.arange(32))

    def warm(self, root):
        from src.v2.models import FineDCN
        torch.manual_seed(9)
        path = root/'warm.pt'
        torch.save({'state_dict': FineDCN(34).state_dict(), 'mean': [0.]*34, 'std': [1.]*34,
                    'metadata': {'columns': list(range(28, 34))}}, path)
        return path

    def train(self, root, warm, **kwargs):
        with redirect_stdout(io.StringIO()):
            return fit_seed(self.x, self.y, self.x, self.diag, self.pairs, self.pairs,
                            warm, self.cfg, 2026, root, {'fixture': True}, **kwargs)

    def test_three_arms_matching_and_exact_group_resume(self):
        with TemporaryDirectory(dir='artifacts') as folder:
            root = Path(folder)
            warm = self.warm(root)
            a = self.train(root/'full', warm)
            self.train(root/'resumed', warm, stop_after=2)
            b = self.train(root/'resumed', warm, resume=True)
            for key in a:
                self.assertEqual(a[key]['history'], b[key]['history'])
                self.assertEqual(a[key]['selected_epoch'], b[key]['selected_epoch'])
                sa = torch.load(root/'full'/key/'model.pt', weights_only=True)
                sb = torch.load(root/'resumed'/key/'model.pt', weights_only=True)
                for name in sa['state_dict']:
                    self.assertTrue(torch.equal(sa['state_dict'][name], sb['state_dict'][name]))
                eligible_rows = [r for r in a[key]['history'] if r['guardrail_pass']]
                best = min(eligible_rows, key=lambda r: r['valid_pair']['logloss'])
                self.assertEqual(a[key]['selected_epoch'], best['epoch'])
            for epoch in range(1, 6):
                for field in ('order_sha256', 'pair_draw_sha256', 'learning_rate'):
                    self.assertEqual(len({a[k]['history'][epoch][field] for k in a}), 1)
            self.assertEqual(len({a[k]['optimizer_steps'] for k in a}), 1)
            self.cfg['aux_weight'] = .3
            with self.assertRaisesRegex(ValueError, 'protocol'):
                self.train(root/'resumed', warm, resume=True)

    def test_group_plateau_and_epoch_zero_retained(self):
        self.cfg.update(max_epochs=12, min_epochs=6, early_patience=4, lr_patience=1)
        with TemporaryDirectory(dir='artifacts') as folder:
            root = Path(folder)
            warm = self.warm(root)
            with patch('src.v2.exact_pair_finetune.pair_metrics', return_value={
                    'pairs': 16, 'logloss': .7, 'accuracy_half_ties': .5}):
                result = self.train(root/'run', warm)
            for record in result.values():
                self.assertEqual(record['selected_epoch'], 0)
                self.assertEqual(record['epochs_completed'], 6)
                self.assertEqual(record['early_stopping_reason'], 'all_arms_validation_plateau_after_shared_lr_reductions')
            self.assertEqual(len({v['optimizer_steps'] for v in result.values()}), 1)
            with self.assertRaises(FileExistsError):
                self.train(root/'run', warm)


if __name__ == '__main__':
    unittest.main()
