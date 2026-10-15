import unittest
import torch
from src.two_tower import TwoTower,contrastive_loss


class TwoTowerTests(unittest.TestCase):
    def test_duplicates_are_positive(self):
        # Both columns name the same true item, so neither is a negative.
        logits=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
        loss=contrastive_loss(logits,torch.tensor([0,1]),torch.tensor([2,2]))
        self.assertAlmostEqual(loss.item(),0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unique_pairs_match_crossentropy(self):
        logits=torch.tensor([[2.,-1.],[0.,3.]],requires_grad=True)
        loss=contrastive_loss(logits,torch.tensor([0,1]),torch.tensor([0,1]))
        self.assertAlmostEqual(loss.item(),torch.nn.functional.cross_entropy(logits,torch.arange(2)).item(),places=6)

    def test_towers_unit_norm(self):
        m=TwoTower(3,4,8)
        self.assertTrue(torch.allclose(m.users(torch.tensor([0,1])).norm(dim=1),torch.ones(2),atol=1e-6))
        self.assertTrue(torch.allclose(m.items(torch.tensor([0,1])).norm(dim=1),torch.ones(2),atol=1e-6))
