import unittest
import numpy as np
from src.ctr_features import FeatureState


class CTRFeaturesTests(unittest.TestCase):
    def test_target_label_does_not_enter_features(self):
        state=FeatureState(np.array([0,1,2]),3)
        u=np.array([1]);i=np.array([1]);t=np.array([1684738200000])
        before=state.features(u,i,t)
        state.update(u,i,np.array([1]))
        after=state.features(u,i,t)
        self.assertEqual(before[0,10],0)
        self.assertEqual(after[0,10],1)
        self.assertGreater(after[0,2],before[0,2])

    def test_search_unknown_and_user_isolation(self):
        state=FeatureState(np.array([0,1,2]),3)
        state.update_search(np.array([1,1,2]),np.array([0,1,2]))
        self.assertEqual(state.su.tolist(),[0,1,1])
        self.assertEqual(state.suc[1*state.nc+2],0)
