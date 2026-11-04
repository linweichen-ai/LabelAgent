from copy import deepcopy
import unittest
import numpy as np
from tools.recycle_duplicate_slots import certified_duplicate_additions
from tools.evaluate_projected_support import mask_support


def hit(*rows):return {r:np.array(rows,dtype=bool) for r in (0,3,6)}
def shape(i,label='pallet'):return {'id':i,'label':label}


class DuplicateSlotsTests(unittest.TestCase):
    def test_duplicate_high_point_addition_can_be_recycled(self):
        self.assertEqual(certified_duplicate_additions([shape(1),shape(2)],[hit(1,1,1),hit(1,1,1)],{1}),([0],[1]))

    def test_all_protected_layers_stay_including_zero_point_masks(self):
        self.assertEqual(certified_duplicate_additions([shape(1),shape(2),shape(3)],[hit(0),hit(0),hit(0)],{1,2}),([0,1],[2]))

    def test_difference_at_any_radius_prevents_recycling(self):
        values=[hit(1,0),hit(1,0)];values[1][6]=np.array([True,True])
        self.assertEqual(certified_duplicate_additions([shape(1),shape(2)],values,set()),([0,1],[]))

    def test_deterministic_representative_and_people_never_removed(self):
        shapes=[shape('b'),shape('a'),shape('c','person')]
        self.assertEqual(certified_duplicate_additions(shapes,[hit(1),hit(1),hit(1)],set()),([1,2],[0]))

    def test_arrays_immutable_and_alignment_validated(self):
        values=[hit(1,1,1),hit(1,1,1)];before=deepcopy(values)
        certified_duplicate_additions([shape(1),shape(2)],values,set())
        for a,b in zip(values,before):
            for r in a:np.testing.assert_array_equal(a[r],b[r])
        with self.assertRaises(ValueError):certified_duplicate_additions([shape(1)],values,set())
        values[1][3]=np.array([True])
        with self.assertRaises(ValueError):certified_duplicate_additions([shape(1),shape(2)],values,set())

    def test_original_metric_duplicate_columns_preserve_object_support_all_radii(self):
        pixels=np.array([[10,10],[25,10],[40,10],[55,10],[70,10],[100,10],[115,10],[130,10],[145,10],[160,10]])
        shapes=[dict(id=i,label='pallet',points=[[7,top],[42,top],[42,13],[7,13]],holes=[]) for i,top in [(1,7),(2,6)]]
        def score(values):return mask_support(values,np.arange(10),pixels,np.ones(10,int),np.array([1]*5+[2]*5),[{'bit':1,'obj_id':1},{'bit':1,'obj_id':2}],180,24)['classes']['pallet']['radii']
        before,after=score(shapes),score(shapes[:1])
        for r in ('0','3','6'):
            self.assertEqual(before[r]['objects_matched_at_support_threshold'],1)
            self.assertEqual(before[r]['objects_matched_at_support_threshold'],after[r]['objects_matched_at_support_threshold'])


if __name__=='__main__':unittest.main()
