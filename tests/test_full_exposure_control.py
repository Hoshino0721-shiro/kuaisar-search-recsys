import unittest
import numpy as np
from src.v2.full_exposure_control import repeated_indices


class FullExposureControlTests(unittest.TestCase):
    def test_repeat_control_preserves_all_samples_without_new_examples(self):
        result=repeated_indices(7,40,2026)
        np.testing.assert_array_equal(result,repeated_indices(7,40,2026))
        self.assertEqual(len(result),40)
        self.assertEqual(set(result),set(range(7)))
        counts=np.bincount(result,minlength=7)
        self.assertEqual(int(counts.min()),5)
        self.assertEqual(int(counts.max()),6)
        self.assertFalse(np.array_equal(result,repeated_indices(7,40,2027)))
        for n,m in [(0,1),(3,2)]:
            with self.assertRaises(ValueError):
                repeated_indices(n,m,2026)


if __name__=='__main__':
    unittest.main()
