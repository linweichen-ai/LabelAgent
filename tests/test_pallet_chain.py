"""Functional image tests, separate from real-scene quality evidence."""
import unittest
import cv2,numpy as np
from track_pallet_chain import chain


class ChainTests(unittest.TestCase):
    def test_known_translation_preserves_ids_and_round_trip(self):
        # Keep the full level-4 patch support inside the synthetic image.
        # The first 128px fixture rejected two near-border points; preserve that
        # failure below instead of changing any tracking threshold.
        a=np.random.default_rng(19).integers(0,256,(768,960),dtype=np.uint8);images={76:a}
        for frame,dx in [(75,2),(74,4)]:images[frame]=cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,0]]),(960,768))
        points=[[350,330],[390,330],[430,330],[350,370],[390,370],[430,370],[450,420]];r=chain(cv2,images,[76,75,74],points);self.assertEqual(r['valid_points'],7);self.assertTrue(r['anchor_valid']);self.assertTrue(r['diagnostic_image_correspondence_available'])
        for p in r['points']:np.testing.assert_allclose(np.array(p['endpoint_xy'])-p['source_xy'],[4,0],atol=.15)
    def test_textureless_target_cannot_create_evidence(self):
        a=np.random.default_rng(8).integers(0,256,(128,160),dtype=np.uint8);r=chain(cv2,{76:a,75:np.zeros_like(a),74:np.zeros_like(a)},[76,75,74],[[40,40],[65,40],[90,40],[40,70],[65,70],[90,70]])
        self.assertEqual(r['valid_points'],0);self.assertFalse(r['diagnostic_image_correspondence_available'])
    def test_empty_seed_stays_empty(self):
        a=np.zeros((80,80),np.uint8);r=chain(cv2,{76:a,75:a,74:a},[76,75,74],[]);self.assertEqual(r['valid_points'],0);self.assertFalse(r['anchor_valid'])
    def test_small_fixture_border_failure_stays_rejected(self):
        a=np.random.default_rng(19).integers(0,256,(128,160),dtype=np.uint8);images={76:a}
        for frame,dx in [(75,2),(74,4)]:images[frame]=cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,0]]),(160,128))
        r=chain(cv2,images,[76,75,74],[[40,40],[65,40],[90,40],[40,70],[65,70],[90,70],[110,90]])
        self.assertFalse(r['anchor_valid']);self.assertFalse(r['diagnostic_image_correspondence_available'])
        for p in r['points']:
            if p['all_edges_and_whole_cycle_valid']:np.testing.assert_allclose(np.array(p['endpoint_xy'])-p['source_xy'],[4,0],atol=.15)


if __name__=='__main__':unittest.main()
