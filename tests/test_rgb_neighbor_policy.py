import unittest
import numpy as np
from rgb_neighbor_policy import propose


class NeighborPromptTests(unittest.TestCase):
    def scene(self):
        rgb=np.full((180,240,3),90,np.uint8);rgb[65:110]=(20,70,130);rgb[110:]=(150,160,170);parent=np.zeros((180,240),np.uint8);parent[20:160,25:220]=1;return rgb,parent
    def test_original_image_and_complete_parent_are_immutable(self):
        rgb,parent=self.scene();a,b=rgb.copy(),parent.copy();propose(rgb,parent);np.testing.assert_array_equal(rgb,a);np.testing.assert_array_equal(parent,b)
    def test_fixed_hypothesis_and_point_budget_with_parent_bound_anchors(self):
        rgb,parent=self.scene();rows,_=propose(rgb,parent);self.assertEqual(len(rows),2)
        for row in rows:
            self.assertLessEqual(len(row['points']),3);self.assertEqual(sum(p['label'] for p in row['points']),1)
            for p in row['points']:self.assertTrue(parent[p['y'],p['x']])
    def test_flat_image_does_not_fabricate_layer_seams(self):
        rgb,parent=self.scene();rgb[:]=127;rows,diag=propose(rgb,parent);self.assertEqual(rows,[]);self.assertEqual(diag['reason'],'No RGB seam')
    def test_empty_and_tiny_parent_abstain(self):
        rgb,parent=self.scene();parent[:]=0;self.assertEqual(propose(rgb,parent)[0],[]);parent[5:8,4:7]=1;self.assertEqual(propose(rgb,parent)[0],[])
    def test_geometry_mismatch_rejected_and_replay_deterministic(self):
        rgb,parent=self.scene()
        with self.assertRaises(ValueError):propose(rgb,parent[:-1])
        self.assertEqual(propose(rgb,parent),propose(rgb,parent))


if __name__=='__main__':unittest.main()
