import unittest
import numpy as np
from src.search_ranking import lists_metrics


class SearchRankingTests(unittest.TestCase):
    def test_session_denominators(self):
        result=lists_metrics(np.array([1,0,0]),np.array([.9,.1,.5]),np.array([1,1,2]),np.array([10,20,30]))
        self.assertEqual(result['sessions'],2)
        self.assertEqual(result['zero_click_sessions'],1)
        self.assertEqual(result['ndcg_at_10_positive_sessions'],1)
        self.assertEqual(result['ndcg_at_10_all_sessions'],.5)
