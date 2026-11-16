"""Check point placement, model disagreement, warnings and request cache scope."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
spec=importlib.util.spec_from_file_location('point_identity_verifier',Path(__file__).resolve().parents[1]/'tools/point_identity_verifier.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def square(x=10):
    return {'id':1,'label':'pallet','points':[[x,10],[x+20,10],[x+20,30],[x,30]],'holes':[],'score':.99}


class PointIdentityTest(unittest.TestCase):
    def test_click_is_inside_shared_foreground_and_outside_hole(self):
        a,b=square(),square(12)
        a['holes']=[[[16,16],[24,16],[24,24],[16,24]]]
        point,_=module.intersection_point(a,b,100,80)
        shared=module._rasterize(a,100,80)&module._rasterize(b,100,80)
        self.assertTrue(shared[point['y'],point['x']])

    def test_disjoint_masks_cannot_make_a_shared_click(self):
        with self.assertRaises(ValueError):module.intersection_point(square(),square(60),100,80)

    def test_warning_disallows_even_a_perfect_mask_and_high_score(self):
        mask=square()
        point={'x':20,'y':20,'label':1}
        result=module.mask_agreement({**mask,'warnings':['unreliable click']},mask,point,100,80,'pallet')
        self.assertEqual(result['mask_iou'],1)
        self.assertFalse(result['point_consistent'])

    def test_high_score_does_not_override_wrong_mask(self):
        result=module.mask_agreement(square(60),square(),{'x':20,'y':20,'label':1},100,80,'pallet')
        self.assertEqual(result['mask_iou'],0)
        self.assertFalse(result['point_consistent'])

    def test_cache_reuses_request_but_rechecks_a_changed_union(self):
        with tempfile.TemporaryDirectory() as directory:
            requests=[]
            def requester(payload):
                requests.append(payload)
                return square()
            verifier=module.PointIdentityVerifier(b'image',100,80,'http://127.0.0.1:8766',directory,.85,'model',requester=requester)
            a,b=square(),square()
            first=verifier(a,b,a)
            changed=deepcopy(a)
            changed['points']=[[0,0],[99,0],[99,79],[0,79]]
            second=verifier(a,b,changed)
            self.assertTrue(first['allow_merge'])
            self.assertFalse(second['allow_merge'])
            self.assertEqual(len(requests),1)
            self.assertTrue(second['cache_hit'])
            self.assertEqual(verifier.backend_mode,'contract_test_double')

    def test_model_identity_changes_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            calls=[]
            def requester(payload):calls.append(payload);return square()
            a=square()
            for fingerprint in ('first','second'):
                verifier=module.PointIdentityVerifier(b'image',100,80,'http://127.0.0.1:8766',directory,.85,fingerprint,requester=requester)
                verifier(a,a,a)
            self.assertEqual(len(calls),2)

    def test_separate_points_use_separate_single_positive_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            calls=[]
            def requester(payload):calls.append(payload);return square()
            verifier=module.PointIdentityVerifier(b'image',100,80,'http://127.0.0.1:8766',directory,.85,'model',requester=requester)
            for x in (15,25):verifier.verify_point(square(),{'x':x,'y':20,'label':1},'pallet')
            self.assertEqual(len(calls),2)
            self.assertTrue(all(len(value['points'])==1 and set(value)=={'image','points'} for value in calls))

    def test_invalid_clicks_fail_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            calls=[]
            verifier=module.PointIdentityVerifier(b'image',100,80,'http://127.0.0.1:8766',directory,.85,'model',requester=lambda payload:calls.append(payload))
            for point in ({'x':-1,'y':20,'label':1},{'x':20,'y':20,'label':0},
                    {'x':20.1,'y':20,'label':1},{'x':70,'y':20,'label':1}):
                with self.assertRaises(ValueError):verifier.verify_point(square(),point,'pallet')
            self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
