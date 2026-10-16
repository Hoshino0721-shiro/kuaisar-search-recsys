import unittest
from src.v2.cascade import Budget, Candidate, Cascade


class CascadeTests(unittest.TestCase):
    def test_scoring_is_cascaded_and_positive_loss_is_visible(self):
        seen = []
        def coarse(user, items, timestamp):
            return [10 - i for i in items]
        def fine(user, items, timestamp):
            seen.extend(items)
            return [float(i) for i in items]
        pipeline = Cascade(coarse, fine, Budget(4, 2, 2, 1))
        result = pipeline.run(1, 100, [Candidate(i, 'a', i, 90) for i in range(1, 5)],
                              positives={1: 2, 4: 1})
        self.assertEqual(seen, [1, 2])
        self.assertEqual(result['items'], [2])
        self.assertEqual([s['hit_events'] for s in result['stages']], [3, 2, 2, 0])
        self.assertAlmostEqual(result['stages'][1]['positive_retention_vs_previous'], 2 / 3)

    def test_unavailable_and_explicitly_excluded_items_cannot_reenter(self):
        score = lambda user, items, ts: [1] * len(items)
        result = Cascade(score, score, Budget(4, 3, 2, 2)).run(
            1, 100, [Candidate(1, 'a', 1, 101), Candidate(2, 'a', 2, 90),
                     Candidate(3, 'a', 3, 90)], exclude_items={2})
        self.assertEqual(result['items'], [3])
        self.assertEqual(result['shortfall'], 1)
        self.assertEqual(result['rejected'], {'not_yet_available': 1, 'excluded_item': 1})

    def test_duplicates_within_a_route_do_not_multiply_rrf_contribution(self):
        score = lambda user, items, ts: [0] * len(items)
        routes = [Candidate(1, 'a', 1, 0)] * 10 + [Candidate(2, 'a', 2, 0), Candidate(2, 'b', 1, 0)]
        result = Cascade(score, score, Budget(2, 2, 2, 1)).run(1, 100, routes)
        self.assertEqual(result['stages'][0]['items'], [2, 1])
        self.assertEqual(result['candidate_sources']['2'], ['a', 'b'])

    def test_diversity_can_change_order_without_repeated_model_calls(self):
        score = lambda user, items, ts: [{1: .9, 2: .89, 3: .88}[i] for i in items]
        result = Cascade(score, score, Budget(3, 3, 3, 2, category_penalty=.1),
                         categories={1: 1, 2: 1, 3: 2}).run(
            1, 100, [Candidate(i, 'a', i, 0) for i in range(1, 4)])
        self.assertEqual(result['items'], [1, 3])

    def test_author_hard_constraint_reports_shortfall(self):
        score = lambda user, items, ts: [1] * len(items)
        result = Cascade(score, score, Budget(3, 3, 3, 3, max_per_author=1),
                         authors={1: 'same', 2: 'same', 3: 'same'}).run(
            1, 100, [Candidate(i, 'a', i, 0) for i in range(1, 4)])
        self.assertEqual(result['items'], [1])
        self.assertEqual(result['shortfall'], 2)

    def test_scorer_contract_is_enforced(self):
        good = lambda user, items, ts: [0] * len(items)
        for bad in [lambda u, i, t: [], lambda u, i, t: [float('nan')]]:
            with self.assertRaises(ValueError):
                Cascade(bad, good, Budget(1, 1, 1, 1)).run(1, 100, [Candidate(1, 'a', 1, 0)])

    def test_empty_candidates_do_not_call_models(self):
        def fail(*args):
            raise AssertionError('No model invocation expected')
        result = Cascade(fail, fail).run(1, 100, [], positives={})
        self.assertEqual(result['items'], [])
        self.assertIsNone(result['stages'][0]['observed_event_recall'])

    def test_invalid_stage_budgets_fail(self):
        for sizes in [(1, 2, 1, 1), (4, 3, 2, 0), (4, 3.0, 2, 1)]:
            with self.assertRaises(ValueError):
                Budget(*sizes)

    def test_disabling_coarse_really_bypasses_its_model(self):
        def fail(*args):
            raise AssertionError('Coarse model should be bypassed')
        seen = []
        def fine(user, items, ts):
            seen.extend(items)
            return [float(i) for i in items]
        result = Cascade(fail, fine, Budget(3, 2, 2, 1, coarse_enabled=False)).run(
            1, 100, [Candidate(i, 'a', i, 0) for i in range(1, 4)])
        self.assertEqual(seen, [1, 2, 3])
        self.assertEqual(result['items'], [3])


if __name__ == '__main__':
    unittest.main()
