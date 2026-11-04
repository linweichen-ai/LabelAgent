import unittest
import numpy as np
from tools.multiview_person_guard import merge_point_views,rejects_novel_points,ranking_inputs
from tools.spatial_candidate_evidence import SpatialPointNovelty
from copy import deepcopy

def view(indices,hits):return {'indices':np.array(indices,dtype=np.int64),'person_hits':np.array(hits,dtype=bool)}

class MultiViewPointTests(unittest.TestCase):
    def test_retained_anchor_is_not_duplicated_as_a_new_proposal(self):
        baseline=[{'id':1}];anchors=[{'id':1},{'id':2}];proposals=[{'id':2},{'id':3}];before=deepcopy((baseline,anchors,proposals))
        seed,props=ranking_inputs(baseline,anchors,proposals,'four')
        self.assertEqual(seed,anchors);self.assertEqual(props,[{'id':3}]);self.assertEqual((baseline,anchors,proposals),before)
        with self.assertRaises(ValueError):ranking_inputs(baseline,anchors,proposals,'unknown')

    def test_already_retained_instance_reduces_duplicate_priority(self):
        def shape(i,left,right):return {'id':i,'label':'pallet','score':.9,'points':[[left,1],[right,1],[right,10],[left,10]],'holes':[]}
        base=[shape(1,1,6)];old=base+[shape(2,20,30)];duplicate=shape(3,23,33);fresh=shape(4,45,55);proposals=[old[1],duplicate,fresh]
        pixels=np.array([[24+i,5] for i in range(6)]+[[45+i,5] for i in range(6)],dtype=np.int32)
        xyz=np.array([[i*.01,0,2] for i in range(6)]+[[1+i*.01,0,2] for i in range(6)])
        sensor={'pixels':pixels,'xyz':xyz,'depths':np.full(12,2.)}
        original=SpatialPointNovelty(base,proposals,64,32,sensor)
        seed,props=ranking_inputs(base,old,proposals,'four');retained=SpatialPointNovelty(seed,props,64,32,sensor)
        self.assertLess(retained.priority(duplicate,tuple(seed),'component_mass'),original.priority(duplicate,tuple(base),'component_mass'))
        self.assertGreater(retained.priority(fresh,tuple(seed),'component_mass'),retained.priority(duplicate,tuple(seed),'component_mass'))

    def test_different_local_orders_are_mapped_to_shared_raw_indices(self):
        result=merge_point_views(9,{0:view([7,2],[True,False]),1:view([2,7],[True,False])})
        self.assertEqual(np.flatnonzero(result['person_any']).tolist(),[2,7])
        self.assertEqual(result['person_counts'][[2,7]].tolist(),[1,1]);self.assertEqual(result['visible_counts'][[2,7]].tolist(),[2,2])

    def test_other_view_can_veto_a_point_with_no_own_view_person_hit(self):
        result=merge_point_views(6,{0:view([1,4],[False,False]),1:view([4,1],[False,True])})
        candidate=np.array([True,False]);anchor=np.array([False,False]);indices=np.array([1,4])
        self.assertFalse(rejects_novel_points(candidate,anchor,indices,result['per_view'][0]))
        self.assertTrue(rejects_novel_points(candidate,anchor,indices,result['person_any']))

    def test_person_hits_already_in_anchor_union_are_not_removed(self):
        self.assertFalse(rejects_novel_points(np.array([True,True]),np.array([True,False]),np.array([2,5]),np.array([False,False,True,False,False,False])))

    def test_invalid_or_duplicate_raw_indices_are_rejected(self):
        for indices in ([1,1],[-1,2],[1,9]):
            with self.assertRaises(ValueError):merge_point_views(6,{0:view(indices,[True,False])})
        with self.assertRaises(ValueError):merge_point_views(6,{0:{'indices':np.array([1.5]),'person_hits':np.array([True])}})

    def test_empty_camera_and_unaligned_evidence_fail_safely(self):
        result=merge_point_views(0,{0:view([],[])})
        self.assertEqual(result['person_any'].size,0)
        with self.assertRaises(ValueError):rejects_novel_points(np.array([True]),np.array([False,False]),np.array([0]),np.array([True]))

if __name__=='__main__':unittest.main()
