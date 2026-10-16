import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.ctr_features import FEATURES, FeatureState
from src.v2.features import coarse_features
from src.v2.models import COARSE_COLUMNS
from src.v2.models import FineDCN, CoarseLR
from src.v2.train_rankers import Ranker, probability_metrics, train


class ModelTests(unittest.TestCase):
    def test_cheap_path_matches_training_columns_for_seen_and_unknown_entities(self):
        state = FeatureState(np.array([0, 1, 2, 0]), 4)
        state.update(np.array([1, 1, 2]), np.array([1, 2, 2]), np.array([1, 0, 1]))
        users = np.array([1, 2, 3, 0])
        items = np.array([1, 2, 3, 0])
        full = state.features(users, items, np.full(4, 1685289600000))
        np.testing.assert_array_equal(coarse_features(state, users, items), full[:, COARSE_COLUMNS])

    def test_cross_layer_equation_and_gradient(self):
        model = FineDCN(input_dim=2, cross_layers=1)
        with torch.no_grad():
            model.weights[0].copy_(torch.tensor([.2, -.3]))
            model.biases[0].copy_(torch.tensor([.1, .2]))
            model.head.weight.zero_()
            model.head.weight[0, :2] = torch.tensor([1., 2.])
            model.head.bias.zero_()
        x = torch.tensor([[2., 3.]], requires_grad=True)
        expected_cross = x * (.2 * x[:, :1] - .3 * x[:, 1:]) + torch.tensor([.1, .2]) + x
        output = model(x)
        self.assertTrue(torch.allclose(output, expected_cross @ torch.tensor([1., 2.])))
        output.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(model.weights[0].grad)

    def test_auc_ties_and_single_class(self):
        self.assertEqual(probability_metrics([0, 1], [.5, .5])['auc'], .5)
        self.assertEqual(probability_metrics([0, 1], [.1, .9])['auc'], 1.)
        self.assertIsNone(probability_metrics([1, 1], [.1, .9])['auc'])

    def test_train_and_reload_without_any_test_files(self):
        # Tiny generated fixtures verify artifact/shape contracts only; no quality claim.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            cache.mkdir()
            rng = np.random.default_rng(4)
            for split, n in [('train', 64), ('valid', 16)]:
                x = rng.normal(size=(n, 20)).astype(np.float32)
                y = (x[:, 2] > 0).astype(np.int8)
                np.save(cache / f'{split}_X.npy', x)
                np.save(cache / f'{split}_y.npy', y)
            (cache / 'ready.json').write_text(json.dumps({'features': FEATURES}), encoding='utf-8')
            (cache / 'state.npz').write_bytes(b'fixture identity only')
            for kind in ['coarse_lr', 'fine_dcn']:
                out = root / kind
                report = train(kind, cache, out, 64, 2, 16, 2026)
                self.assertFalse(report['test_used'])
                self.assertAlmostEqual(report['selected_validation']['logloss'],
                                       min(row['valid']['logloss'] for row in report['history']), places=6)
                predictor = Ranker(out / 'model.pt')
                x = np.load(cache / 'valid_X.npy')
                p = predictor.predict(x)
                self.assertEqual(p.shape, (16,))
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p >= 0) & (p <= 1)).all())
                if kind == 'coarse_lr':
                    changed = x.copy()
                    changed[:, 19] += 100
                    np.testing.assert_array_equal(p, predictor.predict(changed))


if __name__ == '__main__':
    unittest.main()
