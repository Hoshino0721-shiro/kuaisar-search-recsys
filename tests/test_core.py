import unittest
from src.data_pipeline import _time_key, split_rows
from src.metrics import auc, gauc, logloss, brier, ndcg_at_k, recall_at_k, mrr_at_k


class PipelineTests(unittest.TestCase):
    def test_timezone(self):
        expected = _time_key({'time': '2023-05-22T00:00:00Z'})
        self.assertEqual(expected, _time_key({'time': '2023-05-22T08:00:00+08:00'}))
        self.assertEqual(expected, _time_key({'time': '2023-05-22 08:00:00'}))

    def test_ties_and_boundaries(self):
        rows = [{'timestamp': str(t)} for t in [3, 2, 2, 1, 4]]
        parts = split_rows(rows, cutoffs=(2, 4))
        self.assertEqual([len(parts[s]) for s in ('train', 'valid', 'test')], [1, 3, 1])
        self.assertEqual(parts, split_rows(reversed(rows), cutoffs=(2, 4)))

    def test_session_crossing_rejected(self):
        rows = [{'timestamp': str(t), 'search_session_id': 's'} for t in [1, 2]]
        with self.assertRaises(ValueError):
            split_rows(rows, cutoffs=(2, 4))

    def test_invalid_time(self):
        for raw in ('nan', 'inf', 'not-a-date'):
            with self.assertRaises(ValueError):
                _time_key({'time': raw})


class MetricsTests(unittest.TestCase):
    def test_auc_ties(self):
        self.assertEqual(auc([0, 1], [.5, .5]), .5)
        self.assertEqual(auc([0, 1], [.9, .1]), 0)
        self.assertIsNone(auc([1, 1], [.1, .9]))

    def test_gauc(self):
        self.assertEqual(gauc([0, 1, 1], [.1, .9, .2], ['a', 'a', 'b']), 1)

    def test_probability(self):
        self.assertAlmostEqual(logloss([0, 1], [.5, .5]), .69314718056)
        self.assertEqual(brier([0, 1], [.5, .5]), .25)

    def test_bad_inputs(self):
        for labels, scores in [([], []), ([0, 1], [.1]), ([2], [.5]), ([1], [float('nan')])]:
            with self.assertRaises(ValueError):
                auc(labels, scores)
        with self.assertRaises(ValueError):
            logloss([1], [2])

    def test_ranking(self):
        self.assertEqual(recall_at_k(['a', 'b'], ['a', 'c'], 2), .5)
        self.assertEqual(mrr_at_k(['b'], ['a', 'b'], 2), .5)
        self.assertEqual(ndcg_at_k(['a'], ['a'], 1), 1)
        with self.assertRaises(ValueError):
            ndcg_at_k(['a'], ['a', 'a'], 2)


if __name__ == '__main__':
    unittest.main()
