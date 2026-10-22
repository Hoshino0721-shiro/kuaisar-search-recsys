import unittest
import numpy as np
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from src.v2 import full_exposure_control as module
from src.ctr_features import FeatureState
from src.v2.train_rankers import sha
from src.v2.full_exposure_control import repeated_indices


class FullExposureControlTests(unittest.TestCase):
    def test_local_rebuild_trains_all_families_without_remote_models(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'reference';reference.mkdir();numeric=root/'numeric';numeric.mkdir()
            FeatureState(np.arange(25,dtype=np.int16)%3,3).save(numeric/'state.npz')
            rng=np.random.default_rng(9);x=rng.normal(size=(8,34)).astype(np.float32);y=np.array([0,1]*4,dtype=np.int8)
            np.save(numeric/'train_y.npy',y);np.save(numeric/'valid_y.npy',y)
            for name,value in [('train_X',x[:4]),('train_y',y[:4]),('valid_X',x),('mean',x[:4].mean(0)),
                    ('std',np.maximum(x[:4].std(0),1e-6)),('pool_users',np.array([1,2])),
                    ('pool_items',np.tile(np.arange(1,21),(2,1))),('pool_X',rng.normal(size=(2,20,34)).astype(np.float32)),
                    ('coarse_positions',np.tile(np.arange(20),(2,1)))]:
                np.save(reference/(name+'.npy'),value)
            (reference/'metadata.json').write_text('{}')
            cfg={'seeds':[2026],'epochs':1,'batch_size':4,'learning_rate':.001,'weight_decay':1e-5,'threads':1,
                'arms':{'base':[],'R':list(range(28,34))},'families':['full','sample_repeat'],
                'repeat_seed':2026,'checkpoint_rule':'fixed final epoch; no validation checkpoint selection'}
            config=root/'config.json';config.write_text(json.dumps(cfg))
            meta={'candidate_sha256':'fixture','config':{'budget':{'fine':20,'output':20,'category_penalty':0}}}
            def fake_coverage(ref,out):
                np.save(out/'pool_item_seen_sample.npy',np.ones((2,20),dtype=bool))
                return {'full_rows':8,'sample_rows':4}
            def fake_prepare(ref,out,config,metadata):
                out.mkdir()
                for name,value in [('full_X',x),('full_mean',x.mean(0)),('full_std',np.maximum(x.std(0),1e-6)),
                        ('sample_repeat_X',np.tile(x[:4],(2,1))),('sample_repeat_y',y)]:
                    np.save(out/(name+'.npy'),value)
                (out/'metadata.json').write_text('{}')
                return {'status':'ready','test_used':False}
            with patch.object(module,'CACHE',numeric),patch.object(module,'verify_cache',return_value=meta), \
                    patch.object(module,'coverage',side_effect=fake_coverage),patch.object(module,'prepare',side_effect=fake_prepare), \
                    patch.object(module,'load_labels',return_value=([{1:1},{2:1}],np.array([1,1]),None,None)),redirect_stdout(io.StringIO()):
                module.run(reference,root/'missing',root/'missing',root/'run',config,rebuild_sample=True)
            result=json.loads((root/'run/metrics.json').read_text())
            self.assertEqual(result['new_model_runs'],6);self.assertEqual(len(result['runs']),6)
            self.assertEqual(result['status'],'complete');self.assertFalse(result['default_changed'])
            self.assertFalse(result['runs']['sample5_base_s2026'].get('reused',False))

    def test_prepared_cache_rejects_corruption_and_wrong_state(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);ref=root/'ref';ref.mkdir();cache=root/'cache';cache.mkdir()
            (root/'state.npz').write_bytes(b'fixture');(ref/'metadata.json').write_text('{}')
            np.save(cache/'a.npy',np.arange(3))
            metadata={'status':'ready','test_used':False,'feature_state_sha256':sha(root/'state.npz'),
                'reference_metadata_sha256':sha(ref/'metadata.json'),'repeat_seed':2026,'files':{'a.npy':sha(cache/'a.npy')}}
            (cache/'metadata.json').write_text(json.dumps(metadata))
            with patch.object(module,'CACHE',root):
                self.assertEqual(module.verify_prepared(cache,ref,{'repeat_seed':2026})['status'],'ready')
                with self.assertRaises(ValueError):module.verify_prepared(cache,ref,{'repeat_seed':7})
                (cache/'a.npy').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):module.verify_prepared(cache,ref,{'repeat_seed':2026})

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
