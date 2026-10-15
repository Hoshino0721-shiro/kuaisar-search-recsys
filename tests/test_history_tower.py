import unittest
import numpy as np
import torch
from src.history_tower import causal_histories,HistoryTower


class HistoryTests(unittest.TestCase):
    def test_future_and_same_timestamp_excluded(self):
        events=[(1,10,100),(1,20,200),(1,30,200),(1,40,300)]
        u,i,h,final=causal_histories(events,{1:0},{10:1,20:2,30:3,40:4},length=2)
        np.testing.assert_array_equal(h,[[0,0],[0,1],[0,1],[2,3]])
        np.testing.assert_array_equal(final,[[3,4]])

    def test_user_histories_isolated(self):
        _,_,h,_=causal_histories([(1,10,1),(2,20,2)],{1:0,2:1},{10:1,20:2},2)
        self.assertFalse(h.any())

    def test_padding_does_not_change_pool(self):
        torch.manual_seed(1)
        m=HistoryTower(2,3,[0,1,1,2],[0,1,1,1])
        a=m.users(torch.tensor([0]),torch.tensor([[0,0,1]]))
        b=m.users(torch.tensor([0]),torch.tensor([[0,1]]))
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        self.assertTrue(torch.isfinite(m.users(torch.tensor([0]),torch.tensor([[0,0]]))).all())
