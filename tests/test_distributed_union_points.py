"""Check spatial independence, holes and failure on insufficient visible area."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from distributed_union_points import distributed_union_points
from source.agent.edit_agent import _rasterize


class DistributedPointsTest(unittest.TestCase):
    def setUp(self):
        self.mask={'id':1,'label':'pallet','points':[[10,10],[80,10],[80,65],[10,65]],
            'holes':[[[35,25],[55,25],[55,45],[35,45]]]}
        self.first={'x':20,'y':20,'label':1}

    def test_deterministic_separated_points_outside_holes(self):
        points=distributed_union_points(self.mask,self.first,100,80)
        self.assertEqual(points,distributed_union_points(self.mask,self.first,100,80))
        self.assertEqual(points[0],self.first)
        self.assertEqual(len(points),3)
        raster=_rasterize(self.mask,100,80)
        for i,point in enumerate(points):
            self.assertTrue(raster[point['y'],point['x']])
            for other in points[:i]:
                self.assertGreaterEqual(np.hypot(point['x']-other['x'],point['y']-other['y']),8)

    def test_insufficient_area_does_not_repeat_a_click(self):
        mask={**self.mask,'points':[[10,10],[14,10],[14,14],[10,14]],'holes':[]}
        self.assertEqual(distributed_union_points(mask,{'x':12,'y':12,'label':1},100,80),[])

    def test_boundary_click_is_not_treated_as_deep_interior(self):
        self.assertEqual(distributed_union_points(self.mask,{'x':10,'y':10,'label':1},100,80),[])

    def test_outside_click_rejected(self):
        with self.assertRaises(ValueError):distributed_union_points(self.mask,{'x':0,'y':0,'label':1},100,80)


if __name__=='__main__':unittest.main()
