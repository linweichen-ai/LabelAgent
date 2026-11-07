import unittest
import numpy as np
from native_prompt_guard import evaluate

class NativePromptTests(unittest.TestCase):
    def test_empty_negative_center_with_nearby_foreground_is_rejected(self):
        m=np.zeros((12,12),np.uint8);m[1,1]=1;m[6,7]=1
        r=evaluate(m,[{'x':1,'y':1,'label':1},{'x':6,'y':6,'label':0}])
        self.assertFalse(r['valid']);self.assertEqual(r['failed_indices'],[1]);self.assertFalse(r['points'][1]['center_covered']);self.assertEqual(r['points'][1]['pixels_in_required_region'],1)
    def test_consistent_points_pass_without_changing_full_bitmap(self):
        m=np.zeros((12,12),np.uint8);m[1:3,1:3]=1;before=m.copy()
        self.assertTrue(evaluate(m,[{'x':1,'y':1,'label':1},{'x':9,'y':9,'label':0}])['valid']);np.testing.assert_array_equal(m,before)
    def test_geometry_and_original_radius_cannot_silently_change(self):
        with self.assertRaises(ValueError):evaluate(np.zeros((4,4),np.uint8),[{'x':4,'y':0,'label':0}])
        with self.assertRaises(ValueError):evaluate(np.zeros((4,4),np.uint8),[],negative_radius=0)

if __name__=='__main__':unittest.main()
