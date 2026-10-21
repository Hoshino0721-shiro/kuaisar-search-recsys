from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.v2 import controlled_ablation as experiment
from src.v2.train_rankers import Ranker


class ControlledAblationTests(unittest.TestCase):
    def test_masked_features_do_not_change_other_inputs_or_original(self):
        x = np.arange(60, dtype=np.float32).reshape(3, 20)
        before = x.copy()
        z = experiment.standardized(x, np.ones(20), np.full(20, 2.), list(range(16, 20)))
        np.testing.assert_array_equal(z[:, :16], (x[:, :16] - 1) / 2)
        self.assertTrue((z[:, 16:] == 0).all())
        np.testing.assert_array_equal(x, before)

    def test_weights_depend_only_on_historical_seen_flag(self):
        x = np.zeros((3, 20), dtype=np.float32)
        x[:, 10] = [0, 1, 0]
        np.testing.assert_array_equal(experiment.exposure_weights(x, 3.), [1., 3., 1.])
        with self.assertRaises(ValueError):
            experiment.exposure_weights(x, 0.)

    def test_training_reload_mask_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / 'state.npz', marker=[1])
            rng = np.random.default_rng(9)
            x = rng.normal(size=(32, 20)).astype(np.float32)
            x[:, 10] = np.tile([0, 1], 16)
            y = np.tile([0, 1], 16).astype(np.int8)
            mean = x.mean(axis=0)
            std = np.maximum(x.std(axis=0), 1e-6)
            cfg = {'epochs': 1, 'batch_size': 8, 'learning_rate': .001,
                   'weight_decay': 1e-5, 'checkpoint_rule': 'fixed epoch'}
            with patch.object(experiment, 'CACHE', root):
                model = experiment.train_one(x, y, np.arange(32), mean, std, cfg,
                                             2026, False, 1., root / 'run')
                changed = x.copy()
                changed[:, 16:] += 10000
                np.testing.assert_array_equal(model.predict(x), model.predict(changed))
                np.testing.assert_array_equal(model.predict(x), experiment.DiagnosticRanker(root / 'run/model.pt').predict(x))
                with self.assertRaises(FileExistsError):
                    experiment.train_one(x, y, np.arange(32), mean, std, cfg,
                                         2026, False, 1., root / 'run')
                # Experimental masked artifacts must not silently enter production loaders.
                with self.assertRaises(ValueError):
                    Ranker(root / 'run/model.pt')


if __name__ == '__main__':
    unittest.main()
