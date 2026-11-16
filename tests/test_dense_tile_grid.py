"""Coverage and frozen geometry constraints for optional dense SAM grids."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
utils=load('tiled_candidate_utils')
collector=load('collect_tiled_candidates')

class DenseGridTest(unittest.TestCase):
    def test_dense_grid_covers_every_pixel_and_keeps_bounds(self):
        import numpy as np
        for width,height,divisions,fraction in [(1920,1536,3,.45),(97,83,3,1/3),(97,83,4,.3)]:
            grid=utils.tile_grid(width,height,fraction,divisions)
            self.assertEqual(len(grid),divisions**2)
            covered=np.zeros((height,width),dtype=bool)
            for x0,y0,x1,y1 in grid:
                self.assertTrue(0<=x0<x1<=width and 0<=y0<y1<=height)
                covered[y0:y1,x0:x1]=True
            self.assertTrue(covered.all())
            self.assertEqual(grid,utils.tile_grid(width,height,fraction,divisions))

    def test_gaps_nonfinite_or_duplicate_grids_rejected(self):
        for width,height,fraction,divisions in [(1920,1536,.3,3),(1920,1536,float('nan'),3),
                (1920,1536,.45,True),(1920,1536,.45,3.0),(3,3,.9,3),(2,2,.6,3)]:
            with self.assertRaises(ValueError):utils.tile_grid(width,height,fraction,divisions)

    def test_validation_requires_exact_dense_grid(self):
        methods=collector.method_grid('.5','.5',.8)
        selection={'selected_using_tuning_only':True,'metric_version':2,'metric_sha256':'sha',
            'winner':'tiles_s50_b50','frozen_parameters':methods['tiles_s50_b50'],'heldout_frames':[1,2],
            'results':[{'method':'tiles_s50_b50','eligible':True,'checks':{'gain':True}}],
            'grid_spec':{'divisions':3,'crop_fraction':.45}}
        before=deepcopy(selection)
        collector.check_frozen_selection(selection,methods,[1,2],'sha',selection['grid_spec'])
        for grid in [None,{'divisions':2,'crop_fraction':.6},{'divisions':3,'crop_fraction':.5}]:
            with self.assertRaises(ValueError):collector.check_frozen_selection(selection,methods,[1,2],'sha',grid)
        self.assertEqual(selection,before)

if __name__=='__main__':unittest.main()
