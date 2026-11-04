from copy import deepcopy
import unittest
import cv2
import numpy as np
from tools.global_anchor_additions import point_hits,union_pixels,allocate
from tools.tiled_candidate_utils import _mask_record
from source.agent.edit_agent import _rasterize
from tools.evaluate_projected_support import mask_support
from tools.select_global_anchor_additions import run
from types import SimpleNamespace

def shape(i,x0,y0,x1,y1):return {'id':i,'label':'pallet','points':[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],'holes':[]}
def candidate(i,key,record,priority=.9,all_allowed=True):
    return {'key':key,'shape':{'id':i},'record':record,'priority':priority,'allowed':{'r3':True,'all':all_allowed}}

class GlobalAnchorTests(unittest.TestCase):
    def test_cropped_dilation_matches_full_raster_at_edges_and_holes(self):
        s=shape(1,0,0,18,16);s['holes']=[[[4,4],[9,4],[9,9],[4,9]]]
        pixels=np.array([(x,y) for y in range(24) for x in range(32)],dtype=np.int32)
        record=_mask_record(s,32,24);mask=_rasterize(s,32,24)
        for radius in (0,3,6):
            full=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))) if radius else mask
            np.testing.assert_array_equal(point_hits(record,pixels,radius,32,24),full[pixels[:,1],pixels[:,0]]!=0)

    def test_global_unused_slots_can_go_to_one_image_without_dropping_anchors(self):
        a=_mask_record(shape(1,1,1,3,3),32,24);b=_mask_record(shape(2,10,1,12,3),32,24)
        c=_mask_record(shape(3,20,1,22,3),32,24)
        anchors={('a','0'):[a],('b','0'):[a]}
        selected,summary=allocate(anchors,[candidate('b',('a','0'),b),candidate('c',('a','0'),c,.8)],4,100,'all')
        self.assertEqual(len(selected),2);self.assertEqual(summary['pallet_candidates'],4)
        self.assertEqual(len(anchors['a','0']),1)

    def test_count_cap_is_strict_even_when_more_candidates_fit_area(self):
        a=_mask_record(shape(1,1,1,3,3),32,24);b=_mask_record(shape(2,10,1,12,3),32,24)
        selected,summary=allocate({('a','0'):[a]},[candidate('b',('a','0'),b)],1,100,'all')
        self.assertEqual(selected,[]);self.assertEqual(summary['pallet_candidates'],1)

    def test_area_cap_uses_union_and_skips_oversized_candidate(self):
        a=_mask_record(shape(1,1,1,3,3),32,24);large=_mask_record(shape(2,10,1,20,10),32,24)
        small=_mask_record(shape(3,20,18,21,19),32,24)
        selected,summary=allocate({('a','0'):[a]},[candidate('large',('a','0'),large),candidate('small',('a','0'),small,.8)],2,13,'all')
        self.assertEqual(selected[0]['shape']['id'],'small');self.assertEqual(summary['pallet_union_pixels'],13)

    def test_original_nms_and_all_radius_guard_are_enforced(self):
        a=_mask_record(shape(1,1,1,3,3),32,24);b=_mask_record(shape(2,10,1,12,3),32,24)
        candidates=[candidate('duplicate',('a','0'),a),candidate('r3_only',('a','0'),b,.8,False)]
        selected,_=allocate({('a','0'):[a]},candidates,3,100,'all');self.assertEqual(selected,[])
        selected,_=allocate({('a','0'):[a]},candidates,3,100,'r3');self.assertEqual(selected[0]['shape']['id'],'r3_only')

    def test_anchors_above_original_caps_fail_instead_of_being_trimmed(self):
        a=_mask_record(shape(1,1,1,3,3),32,24)
        with self.assertRaises(ValueError):allocate({('a','0'):[a]},[],0,100,'all')
        with self.assertRaises(ValueError):allocate({('a','0'):[a]},[],1,8,'all')

    def test_five_point_reference_can_match_three_candidate_hits_but_not_two(self):
        # The original metric's five-point minimum is an object denominator.
        pixels=np.array([[20,20],[40,20],[60,20],[80,20],[100,20]],dtype=np.int32)
        points=np.arange(5);bits=np.ones(5,dtype=np.uint8);owner=np.ones(5,dtype=np.int64)
        boxes=[{'bit':1,'obj_id':'synthetic_object'}]
        for right,expected in ((43,0),(63,1)):
            value=mask_support([shape(1,15,15,right,25)],points,pixels,bits,owner,boxes,128,48,radii=(3,))
            self.assertEqual(value['classes']['pallet']['supported_reference_objects'],1)
            self.assertEqual(value['classes']['pallet']['radii']['3']['objects_matched_at_support_threshold'],expected)

    def test_unsupported_total_point_gate_fails_before_opening_inputs(self):
        for invalid in (0,2,5,True,3.0):
            with self.assertRaises(ValueError):run(SimpleNamespace(minimum_extra_raw3_points=invalid))

if __name__=='__main__':unittest.main()
