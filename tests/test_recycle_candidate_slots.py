from copy import deepcopy
import unittest
import numpy as np
from tools.recycle_candidate_slots import certified_redundant_additions
from tools.evaluate_projected_support import mask_support


def hit(*rows):return {r:np.array(rows,dtype=bool) for r in (0,3,6)}
def shape(i):return {'id':i,'label':'pallet'}


class RecycleSlotsTests(unittest.TestCase):
    def test_protected_zero_point_layers_are_never_removed(self):
        self.assertEqual(certified_redundant_additions([shape(1),shape(2)],[hit(0),hit(0)],{1}),([0],[1]))

    def test_removal_requires_other_coverage_in_every_radius(self):
        values=[hit(1,0),hit(1,0)];values[1][6]=np.array([True,True])
        self.assertEqual(certified_redundant_additions([shape(1),shape(2)],values,{1}),([0,1],[]))

    def test_iterative_removal_does_not_drop_last_owner(self):
        self.assertEqual(certified_redundant_additions([shape(1),shape(2)],[hit(1),hit(1)],set()),([1],[0]))

    def test_three_point_candidates_keep_matching_potential(self):
        self.assertEqual(certified_redundant_additions([shape(1),shape(2)],[hit(1,1,1),hit(1,1,1)],{1}),([0,1],[]))

    def test_input_is_immutable_and_bad_alignment_rejected(self):
        shapes=[shape(1),shape(2)];values=[hit(1,0),hit(1,0)];before=deepcopy(values)
        certified_redundant_additions(shapes,values,{1})
        for a,b in zip(values,before):
            for r in a:np.testing.assert_array_equal(a[r],b[r])
        with self.assertRaises(ValueError):certified_redundant_additions(shapes,values[:1],set())
        values[1][3]=np.array([True])
        with self.assertRaises(ValueError):certified_redundant_additions(shapes,values,set())

    def test_original_metric_three_of_five_matches_but_two_does_not(self):
        pixels=np.array([[10,10],[25,10],[40,10],[55,10],[70,10]])
        def measure(right):
            s={'id':1,'label':'pallet','points':[[7,7],[right,7],[right,13],[7,13]],'holes':[]}
            return mask_support([s],np.arange(5),pixels,np.ones(5,int),np.ones(5,int),[{'bit':1,'obj_id':1}],80,24,radii=(3,))['classes']['pallet']['radii']['3']['objects_matched_at_support_threshold']
        self.assertEqual(measure(27),0);self.assertEqual(measure(42),1)


if __name__=='__main__':unittest.main()
