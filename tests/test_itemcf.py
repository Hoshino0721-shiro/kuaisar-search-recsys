import unittest
import numpy as np
from scipy.sparse import csr_matrix
from src.itemcf import rank_items,rank_sparse


class ItemCFTests(unittest.TestCase):
    def test_vectorized_selection_matches_reference(self):
        rng=np.random.default_rng(42)
        items=np.arange(300)[::-1]
        h=csr_matrix(([1.,1.],([0,0],[0,25])),shape=(1,300))
        raw=rng.integers(0,5,300).astype(float)
        raw[h.indices]+=1
        expected=raw.copy();expected[h.indices]-=1
        ranked=rank_sparse(csr_matrix(raw.reshape(1,-1)),h,items,20)
        reference=rank_items({i:v for i,v in enumerate(expected) if v>1e-10},items,20)
        self.assertEqual(ranked,reference)

    def test_tie_order(self):
        self.assertEqual(rank_items({0:1,1:1,2:2},[20,10,30],2),[30,10])

    def test_sparse_matches_dense_cosine(self):
        b=csr_matrix([[1.,1.,0.],[1.,0.,1.]])
        degree=np.asarray(b.sum(axis=0)).ravel()
        norm=b.multiply(1/np.sqrt(degree)).tocsr()
        h=b.getrow(0)
        actual=((h@norm.T)@norm).toarray()-h.toarray()
        sim=norm.toarray().T@norm.toarray()
        np.fill_diagonal(sim,0)
        np.testing.assert_allclose(actual,h.toarray()@sim,atol=1e-12)
