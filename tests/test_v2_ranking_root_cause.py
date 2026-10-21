import unittest

import numpy as np
from src.v2.ranking_root_cause import weighted_exposure_metrics
from src.v2.train_rankers import probability_metrics


class RootCauseTests(unittest.TestCase):
    def test_logged_counts_equal_expanded_rows(self):
        p = [0.2, 0.8, 0.5]
        n = [3, 4, 0]
        y = [1, 3, 0]
        actual = weighted_exposure_metrics(p, n, y)
        reference = probability_metrics([1, 0, 0, 1, 1, 1, 0], [.2, .2, .2, .8, .8, .8, .8])
        self.assertEqual(actual['rows'], 7)
        self.assertEqual(actual['clicks'], 4)
        self.assertAlmostEqual(actual['auc'], reference['auc'])
        self.assertAlmostEqual(actual['logloss'], reference['logloss'])

    def test_unexposed_does_not_become_negative(self):
        a = weighted_exposure_metrics([.8, .99], [2, 0], [1, 0])
        b = weighted_exposure_metrics([.8], [2], [1])
        self.assertEqual(a, b)

    def test_invalid_counts_and_probabilities_fail(self):
        for p, n, y in [([.5], [0], [1]), ([.5], [-1], [0]), ([float('nan')], [1], [0])]:
            with self.assertRaises(ValueError):
                weighted_exposure_metrics(p, n, y)

    def test_empty_and_single_class(self):
        self.assertIsNone(weighted_exposure_metrics([], [], [])['auc'])
        self.assertIsNone(weighted_exposure_metrics([.5], [2], [2])['auc'])


if __name__ == '__main__':
    unittest.main()
