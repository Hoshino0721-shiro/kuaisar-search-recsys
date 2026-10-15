import math
import unittest
from src.baseline import evaluate


class BaselineTests(unittest.TestCase):
    def test_constant_reference(self):
        result=evaluate([0,1,0,1],[.5]*4,[1,1,2,2])
        self.assertEqual(result['auc'],.5)
        self.assertEqual(result['gauc'],.5)
        self.assertAlmostEqual(result['logloss'],math.log(2))
        self.assertEqual(result['brier'],.25)

    def test_perfect_order(self):
        self.assertEqual(evaluate([0,1],[.1,.9],[1,1])['auc'],1)
