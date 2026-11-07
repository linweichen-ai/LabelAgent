"""Check transport direction, scaling, holes, invalid motion and real image flow."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from temporal_candidate_utils import (estimate_bidirectional_flow,prepare_flow_warp,warp_candidate,
    exclude_predicted_person,restrict_to_semantic_support,augment_preserving_tiled)
from source.agent.edit_agent import _rasterize

SETTINGS={'minimum_inherited_source_score':.5,'minimum_area_pixels_full_resolution':50,
    'area_ratio_bounds':[.5,2.],'minimum_consistent_foreground_fraction':.9,
    'minimum_polygon_raster_IoU':.98}


def shape():
    return {'id':'original','label':'pallet','score':.8,'points':[[20,20],[60,20],[60,60],[20,60]],
        'holes':[[[30,30],[40,30],[40,40],[30,40]]]}


class TemporalCandidatesTest(unittest.TestCase):
    def test_backward_sample_moves_right_and_scales_grid_displacement(self):
        forward=np.zeros((20,25,2),dtype=np.float32)
        forward[:,:,0]=2
        warp=prepare_flow_warp(forward,-forward,100,80)
        source=shape()
        before=deepcopy(source)
        output,audit=warp_candidate(source,warp,'new',SETTINGS)
        expected=np.zeros((80,100),dtype=np.uint8)
        expected[:,8:]=_rasterize(source,100,80)[:,:-8]
        self.assertTrue(audit['accepted'])
        self.assertTrue(np.array_equal(expected,_rasterize(output,100,80)))
        self.assertEqual(source,before)
        self.assertEqual(output['id'],'new')
        self.assertFalse(audit['score_is_target_confidence'])

    def test_identity_preserves_holes_and_multiple_parts(self):
        source=shape()
        source['parts']=[{'points':source['points'],'holes':source['holes']},
            {'points':[[70,10],[85,10],[85,25],[70,25]],'holes':[]}]
        flow=np.zeros((20,25,2),dtype=np.float32)
        output,audit=warp_candidate(source,prepare_flow_warp(flow,flow,100,80),2,SETTINGS)
        self.assertTrue(audit['accepted'])
        self.assertTrue(np.array_equal(_rasterize(source,100,80),_rasterize(output,100,80)))

    def test_inconsistent_motion_rejects_without_shrinking_source(self):
        forward=np.zeros((20,25,2),dtype=np.float32)
        backward=forward.copy()
        backward[:,:,0]=-2
        source=shape()
        before=deepcopy(source)
        output,audit=warp_candidate(source,prepare_flow_warp(forward,backward,100,80),2,SETTINGS)
        self.assertIsNone(output)
        self.assertEqual(audit['reason'],'forward_backward_inconsistent')
        self.assertEqual(source,before)

    def test_out_of_frame_transport_is_not_clamped_to_image_edge(self):
        forward=np.zeros((20,25,2),dtype=np.float32)
        forward[:,:,0]=30
        output,audit=warp_candidate(shape(),prepare_flow_warp(forward,-forward,100,80),2,SETTINGS)
        self.assertIsNone(output)
        self.assertEqual(audit['transported_pixels'],0)

    def test_nan_motion_rejected_before_sampling(self):
        flow=np.zeros((20,25,2),dtype=np.float32)
        flow[4,5,0]=np.nan
        with self.assertRaises(ValueError):prepare_flow_warp(flow,flow,100,80)

    def test_other_class_cannot_be_relabelled_as_a_pallet(self):
        source={**shape(),'label':'person'}
        flow=np.zeros((20,25,2),dtype=np.float32)
        with self.assertRaises(ValueError):warp_candidate(source,prepare_flow_warp(flow,flow,100,80),2,SETTINGS)

    def test_real_farneback_recovers_known_image_translation(self):
        first=np.random.default_rng(4).integers(0,256,(128,128,3),dtype=np.uint8)
        second=cv2.warpAffine(first,np.float32([[1,0,4],[0,1,0]]),(128,128))
        config={'width':128,'height':128,'pyr_scale':.5,'levels':3,'winsize':21,'iterations':3,
            'poly_n':7,'poly_sigma':1.5,'gaussian':True}
        forward,backward=estimate_bidirectional_flow(first,second,config)
        source={'id':1,'label':'pallet','score':.9,'points':[[40,40],[80,40],[80,80],[40,80]],'holes':[]}
        output,audit=warp_candidate(source,prepare_flow_warp(forward,backward,128,128),2,SETTINGS)
        self.assertTrue(audit['accepted'])
        target={**source,'points':[[44,40],[84,40],[84,80],[44,80]]}
        actual,expected=_rasterize(output,128,128),_rasterize(target,128,128)
        self.assertGreater(np.count_nonzero(actual&expected)/np.count_nonzero(actual|expected),.95)

    def test_predicted_person_pixels_removed_only_from_new_proposal(self):
        source=shape()
        before=deepcopy(source)
        person=np.zeros((80,100),dtype=np.uint8)
        person[20:61,55:61]=1
        rules={'minimum_retained_fraction':.5,'minimum_retained_pixels':50,'minimum_polygon_raster_IoU':.98}
        output,audit=exclude_predicted_person(source,person,rules)
        self.assertTrue(audit['accepted'])
        self.assertFalse(np.any(_rasterize(output,100,80)&person))
        self.assertEqual(source,before)
        self.assertEqual(output['score'],source['score'])

    def test_large_person_occlusion_rejects_instead_of_claiming_a_good_fragment(self):
        person=np.ones((80,100),dtype=np.uint8)
        rules={'minimum_retained_fraction':.5,'minimum_retained_pixels':50,'minimum_polygon_raster_IoU':.98}
        output,audit=exclude_predicted_person(shape(),person,rules)
        self.assertIsNone(output)
        self.assertFalse(audit['accepted'])

    def test_disjoint_person_predictions_keep_proposal_byte_equivalent(self):
        source=shape()
        person=np.zeros((80,100),dtype=np.uint8)
        rules={'minimum_retained_fraction':.5,'minimum_retained_pixels':50,'minimum_polygon_raster_IoU':.98}
        output,audit=exclude_predicted_person(source,person,rules)
        self.assertEqual(output,source)
        output['score']=0
        self.assertEqual(source['score'],.8)

    def test_uncertainty_margin_protects_nearby_person_when_masks_are_dilated(self):
        person=np.zeros((80,100),dtype=np.uint8)
        person[20:61,64:75]=1
        before=person.copy()
        rules={'minimum_retained_fraction':.5,'minimum_retained_pixels':50,
            'minimum_polygon_raster_IoU':.98,'dilation_radius_pixels':6}
        output,audit=exclude_predicted_person(shape(),person,rules)
        self.assertTrue(audit['accepted'])
        actual=_rasterize(output,100,80)
        tolerant=cv2.dilate(actual,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        self.assertFalse(np.any(tolerant&person))
        self.assertTrue(np.array_equal(person,before))

    def test_semantic_refinement_cannot_add_pixels_outside_existing_class_support(self):
        source=shape()
        support=np.zeros((80,100),dtype=np.uint8)
        support[:,:52]=1
        before=deepcopy(source)
        output,audit=restrict_to_semantic_support(source,support,
            {'minimum_retained_fraction':.5,'minimum_retained_pixels':50,'minimum_polygon_raster_IoU':.98})
        self.assertTrue(audit['accepted'])
        self.assertFalse(np.any((_rasterize(output,100,80)!=0)&(support==0)))
        self.assertEqual(source,before)

    def test_semantic_refinement_without_target_evidence_is_rejected(self):
        output,audit=restrict_to_semantic_support(shape(),np.zeros((80,100),dtype=np.uint8),
            {'minimum_retained_fraction':.5,'minimum_retained_pixels':50,'minimum_polygon_raster_IoU':.98})
        self.assertIsNone(output)
        self.assertFalse(audit['accepted'])

    def test_preserved_tiles_do_not_allow_a_larger_total_addition_budget(self):
        original=[{**shape(),'id':i} for i in (1,2)]
        tiled=original+[{**shape(),'id':3}]
        proposal={**shape(),'id':4,'points':[[70,10],[90,10],[90,30],[70,30]],'holes':[]}
        output,info=augment_preserving_tiled(original,tiled,[proposal],100,80,
            {'minimum_score':.5,'addition_ratio':.5,'nms_iou':.8})
        self.assertEqual(output,tiled)
        self.assertEqual(info['temporal_candidates_added'],0)

    def test_temporal_refinement_uses_only_remaining_original_budget(self):
        original=[{**shape(),'id':i} for i in (1,2,3,4)]
        tiled=original+[{**shape(),'id':5}]
        proposal={**shape(),'id':6,'points':[[70,10],[90,10],[90,30],[70,30]],'holes':[]}
        output,info=augment_preserving_tiled(original,tiled,[proposal],100,80,
            {'minimum_score':.5,'addition_ratio':.5,'nms_iou':.8})
        self.assertEqual(output[:len(tiled)],tiled)
        self.assertEqual(len(output)-len(original),2)
        self.assertEqual(info['temporal_candidates_added'],1)


if __name__=='__main__':unittest.main()
