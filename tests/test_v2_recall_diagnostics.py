import unittest
from src.v2.cascade import Budget, Candidate, Cascade
from src.v2.recall_diagnostics import fuse, hits


class RecallDiagnosticsTests(unittest.TestCase):
    def test_fusion_matches_executed_pipeline_with_duplicates_future_and_zero_weight(self):
        routes = [Candidate(1, 'popular', 1, 0), Candidate(1, 'popular', 3, 0),
                  Candidate(2, 'recent', 1, 0), Candidate(1, 'recent', 2, 0),
                  Candidate(3, 'cf', 1, 0), Candidate(4, 'recent', 1, 11)]
        for weights in ({}, {'popular': 1., 'recent': 2., 'cf': 0.}):
            score = lambda u, ids, t: [0.] * len(ids)
            executed = Cascade(score, score, Budget(3, 3, 3, 1)).run(
                0, 10, routes, route_weights=weights)['stages'][0]['items']
            self.assertEqual(fuse(routes, weights, 3, 10), executed)

    def test_event_hits_preserve_repeat_clicks_without_duplicate_candidate_credit(self):
        self.assertEqual(hits([1, 1, 4], {1: 3, 2: 1}), 3)


if __name__ == '__main__':
    unittest.main()
