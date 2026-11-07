import importlib.util,unittest
from pathlib import Path
import numpy as np,torch
from m2f_query_replay import replay,probability

PARAM={'pallet_score_threshold':.2,'pallet_overlap_threshold':.6,'promote_null_pallet':False}


class ReplayTests(unittest.TestCase):
    def test_exact_frozen_postprocessor_equivalence(self):
        local=Path(__file__).resolve().parents[2]/'outputs/mask2former_single_recovery_20261104/最新单模型候选包/calibrated_postprocess.py';p=local if local.exists() else Path(__file__).with_name('fixtures')/'m2f_calibrated.py';spec=importlib.util.spec_from_file_location('original_calibrated',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for seed in range(4):
            rng=np.random.default_rng(seed);c=rng.normal(size=(8,8)).astype('float32');c[np.arange(8),[0,0,1,2,2,3,6,7]]+=7;mask=rng.normal(size=(8,9,11)).astype('float32');a,segments,_,_=replay(c,mask,(24,30),PARAM);b,s=m.postprocess({'class_queries_logits':torch.tensor(c)[None],'masks_queries_logits':torch.tensor(mask)[None]},(24,30),PARAM);np.testing.assert_array_equal(a,b);self.assertEqual(segments,s)

    def test_null_not_promoted_and_native_retained(self):
        c=np.zeros((1,8),np.float32);c[0,7]=4;c[0,0]=3.8;mask=np.full((1,4,5),5,np.float32);ids,_,rows,_=replay(c,mask,(8,10),PARAM);self.assertFalse(rows[0]['class_gate']);self.assertEqual(rows[0]['original_class_gate_reason'],'argmax_null');self.assertFalse(ids.any());self.assertEqual(int((probability(mask,0,(8,10))>=.5).sum()),80)

    def test_competition_does_not_modify_native_mask(self):
        c=np.full((2,8),-5,np.float32);c[:,0]=[8,7];mask=np.full((2,4,5),4,np.float32);_,_,rows,winner=replay(c,mask,(8,10),PARAM);self.assertTrue((winner==0).all());self.assertEqual(rows[1]['legacy_selected_pixels'],0);self.assertEqual(rows[1]['native_pixels'],80);self.assertEqual(int((probability(mask,1,(8,10))>=.5).sum()),80)


if __name__=='__main__':unittest.main()
