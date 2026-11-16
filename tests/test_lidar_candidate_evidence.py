"""Raw geometry must fail closed and preserve unsupported predictions."""
from copy import deepcopy
import unittest
from pathlib import Path
import tempfile
import numpy as np
from tools.lidar_candidate_evidence import fit_support_plane,candidate_evidence,height_priorities,ReferenceReadAudit


class LiDAREvidenceTests(unittest.TestCase):
    def test_predictor_read_boundary_allows_raw_data_and_denies_reference_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'source/lidar').mkdir(parents=True)
            (root/'source/labels').mkdir();(root/'source/lidar/raw.pcd').write_bytes(b'raw')
            (root/'source/labels/reference.json').write_text('{}')
            audit=ReferenceReadAudit(root);audit.start()
            try:
                self.assertEqual((root/'source/lidar/raw.pcd').read_bytes(),b'raw')
                with self.assertRaises(PermissionError):(root/'source/labels/reference.json').read_text()
            finally:audit.stop()
            self.assertEqual(audit.sources,{'lidar/raw.pcd'})
            self.assertEqual(len(audit.denied),1)
            self.assertEqual((root/'source/labels/reference.json').read_text(),'{}')

    def floor(self):
        rng=np.random.default_rng(3);xy=rng.uniform(-8,8,(3000,2))
        return np.column_stack((xy,-2+.02*xy[:,0]-.01*xy[:,1]+rng.normal(0,.003,len(xy))))

    def test_tilted_floor_is_fitted_amid_above_plane_points_without_labels(self):
        floor=self.floor();rng=np.random.default_rng(4)
        objects=np.column_stack((rng.uniform(-8,8,(2000,2)),rng.uniform(-1.5,2,2000)))
        before=floor.copy();r=fit_support_plane(np.vstack((floor,objects)))
        self.assertTrue(r['usable'],r)
        self.assertAlmostEqual(r['origin_depth'],2,places=2)
        self.assertLess(r['median_inlier_residual'],.01)
        self.assertTrue(np.array_equal(floor,before))

    def test_insufficient_vertical_and_above_origin_geometry_are_not_ground(self):
        self.assertFalse(fit_support_plane(np.zeros((2,3)))['usable'])
        floor=self.floor();wall=np.column_stack((np.ones(len(floor)),floor[:,1],floor[:,0]/3))
        self.assertFalse(fit_support_plane(wall)['usable'])
        high=floor.copy();high[:,2]+=4
        self.assertFalse(fit_support_plane(high)['usable'])

    def test_few_lidar_points_are_unknown_instead_of_negative(self):
        shape={'id':1,'label':'pallet','points':[[0,0],[10,0],[10,10],[0,10]],'holes':[],'score':.8}
        sensor={'plane_usable':True,'pixels':np.array([[5,5],[6,6]]),'heights':np.array([0.,1.])}
        facts=candidate_evidence(shape,20,20,sensor)
        self.assertFalse(facts['supported'])
        self.assertEqual(facts['points'],2)
        for mode in ('positive','penalty'):
            self.assertEqual(height_priorities([], [shape],20,20,{1:facts},mode)[1],.4 if mode=='positive' else .8)

    def test_positive_ordering_is_uniformly_normalized_to_legacy_priority_domain(self):
        shapes=[{'id':i,'label':'pallet','points':[[0,0],[10,0],[10,10],[0,10]],'holes':[],'score':1.} for i in (1,2,3)]
        facts={1:{'supported':False},2:{'supported':True,'low_object_fraction':0.},3:{'supported':True,'low_object_fraction':1.}}
        scores=height_priorities([],shapes,20,20,facts,'positive')
        self.assertEqual(scores,{1:.5,2:.5,3:1.})
        self.assertTrue(all(0<=v<=1 for v in scores.values()))

    def test_height_penalty_changes_priority_without_rewriting_score_or_geometry(self):
        shape={'id':1,'label':'pallet','points':[[0,0],[10,0],[10,10],[0,10]],'holes':[],'score':.8}
        before=deepcopy(shape)
        floor={'supported':True,'floor_fraction':1.,'low_object_fraction':0.,'tall_fraction':0.}
        low={'supported':True,'floor_fraction':0.,'low_object_fraction':1.,'tall_fraction':0.}
        self.assertLess(height_priorities([], [shape],20,20,{1:floor},'penalty')[1],
            height_priorities([], [shape],20,20,{1:low},'penalty')[1])
        self.assertEqual(shape,before)
        self.assertEqual(height_priorities([shape], [shape],20,20,{1:low},'positive')[1],0.)


if __name__=='__main__':unittest.main()
