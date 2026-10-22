import unittest
from collections import Counter
import numpy as np
from src.v2.route_attribution import feature_mask, normalize, metric_arrays, retention_histogram, limited_orders


class RouteAttributionTests(unittest.TestCase):
    def test_masks_keep_only_named_route_and_original_features(self):
        for columns in ([],[28,30],[29,31,32],[33]):
            mask=feature_mask(columns)
            self.assertEqual(np.flatnonzero(mask).tolist(),list(range(20))+columns)
            x=np.ones((3,34),dtype=np.float32)*9
            z=normalize(x,np.full(34,3,dtype=np.float32),np.full(34,2,dtype=np.float32),columns)
            np.testing.assert_array_equal(z,np.broadcast_to(mask*3,z.shape))
        with self.assertRaises(ValueError):
            feature_mask([20])
        with self.assertRaises(ValueError):
            feature_mask([28,28])

    def test_item_id_ties_and_repeated_click_event_weights(self):
        ids=np.arange(24,0,-1)[None,:]
        scores=np.ones(ids.shape)
        hits,ndcg=metric_arrays(ids,scores,[Counter({1:3,20:1,21:7})])
        self.assertEqual(hits.tolist(),[4])
        self.assertTrue(0<ndcg[0]<1)

    def test_retention_bins_account_for_every_lost_event(self):
        h=retention_histogram([1,20,21,80,81,400,401,2000],[1,2,3,4,5,6,7,8])
        self.assertEqual(h,{'1_20':3,'21_80':7,'81_400':11,'401_2000':15})
        self.assertEqual(sum(h.values()),36)

    def test_bounded_adjustments_keep_pool_and_reserved_prefix(self):
        ids=np.arange(2000,0,-1)
        p=np.arange(2000)
        orders=limited_orders(ids,p)
        self.assertEqual(orders['rrf_head40_dcn'],ids[20:40][::-1].tolist())
        reserved=orders['reserve_rrf10_then_dcn']
        self.assertEqual(reserved[:10],ids[:10].tolist())
        for order in orders.values():
            self.assertEqual(len(order),20)
            self.assertEqual(len(set(order)),20)
            self.assertTrue(set(order)<=set(ids))


if __name__=='__main__':
    unittest.main()
