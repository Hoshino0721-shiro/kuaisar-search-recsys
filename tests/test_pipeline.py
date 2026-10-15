import unittest
import numpy as np
from src.retrieval_pipeline import rrf,rerank


class PipelineTests(unittest.TestCase):
    def test_rrf_deduplicates_and_is_deterministic(self):
        self.assertEqual(rrf([[2,1],[1,2]],[1,1],2),[1,2])
        self.assertEqual(rrf([[2,1],[3]],[1,0],2),[2,1])

    def test_diversity_penalty(self):
        cats=np.array([0,1,1,2])
        self.assertEqual(rerank([1,2,3],[.9,.89,.88],cats,0,3),[1,2,3])
        self.assertEqual(rerank([1,2,3],[.9,.89,.88],cats,.1,3),[1,3,2])
