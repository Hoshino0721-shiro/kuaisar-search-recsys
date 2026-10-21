import unittest
import numpy as np
from src.v2.ranking_sensitivity import derangement, permute_group, paired_interval


class SensitivityTests(unittest.TestCase):
    def test_user_mapping_is_reproducible_and_has_no_self_matches(self):
        users = [1, 4, 9, 11]
        mapping = derangement(users)
        self.assertEqual(mapping, derangement(users))
        self.assertEqual(set(mapping.values()), set(users))
        self.assertTrue(all(u != v for u, v in mapping.items()))
        with self.assertRaises(ValueError):
            derangement([1])
        with self.assertRaises(ValueError):
            derangement([1, 1])

    def test_permutation_preserves_group_rows_and_other_columns(self):
        x = np.arange(60).reshape(10, 6)
        saved = x.copy()
        y = permute_group(x, [1, 3], 2026)
        self.assertTrue(np.array_equal(x, saved))
        self.assertTrue(np.array_equal(x[:, [0, 2, 4, 5]], y[:, [0, 2, 4, 5]]))
        self.assertEqual(sorted(map(tuple, x[:, [1, 3]])), sorted(map(tuple, y[:, [1, 3]])))

    def test_bootstrap_constant_empty_and_determinism(self):
        a = paired_interval([.1] * 20, repeats=50)
        np.testing.assert_allclose(a['exploratory_95pct_user_bootstrap'], [.1, .1])
        self.assertEqual(a, paired_interval([.1] * 20, repeats=50))
        self.assertIsNone(paired_interval([])['mean_delta'])


if __name__ == '__main__':
    unittest.main()
