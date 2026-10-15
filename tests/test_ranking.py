import unittest
import numpy as np
import torch
from src.ranking import metrics,GateRanker


class RankingTests(unittest.TestCase):
    def test_group_weight_and_calibration(self):
        result=metrics(np.array([0,1,1]),np.array([.5,.5,.5]),np.array([1,1,2]))
        self.assertEqual(result['gauc'],.5)
        self.assertEqual(result['gauc_effective_rows'],2)
        self.assertAlmostEqual(result['brier'],.25)
        self.assertEqual(sum(b['rows'] for b in result['reliability']),3)

    def test_missing_search_gate(self):
        model=GateRanker(True)
        a=torch.zeros((1,20));b=a.clone();b[:,16:]=100
        self.assertTrue(torch.allclose(model(a,torch.zeros(1)),model(b,torch.zeros(1))))
