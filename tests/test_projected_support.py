"""Analytic checks for sparse reference agreement and its exclusion rules."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('evaluate_projected_support',Path(__file__).resolve().parents[1]/'tools/evaluate_projected_support.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def camera():
    return {'width':100,'height':80,'intrinsics':np.array([100,80,50,40]),'transform':np.eye(4)}


def obj(label,center=(0,0,3),size=(2,2,2)):
    return {'obj_type':label,'obj_psr':{'position':dict(zip('xyz',center)),
        'scale':dict(zip('xyz',size)),'rotation':{'roll':0,'pitch':0,'yaw':0}}}


def square(identifier,left,right,label='pallet'):
    return {'id':identifier,'label':label,'points':[[left,10],[right,10],[right,25],[left,25]]}


class ProjectedSupportTests(unittest.TestCase):
    def test_projection_respects_world_to_camera_rotation_and_intrinsics(self):
        info=camera()
        info['transform'][:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        source,pixels=module.project_visible(np.array([[1.,2.,10.]]),info)
        self.assertEqual(source.tolist(),[0])
        self.assertEqual(pixels.tolist(),[[30,48]])

    def test_depth_filter_excludes_behind_camera_and_occluded_samples(self):
        points=np.array([[0,0,2],[0,0,6],[.6,0,2],[0,0,-1]],dtype=float)
        source,pixels=module.project_visible(points,camera(),cell_size=8,depth_slack=.15)
        self.assertEqual(source.tolist(),[0,2])
        self.assertEqual(pixels.tolist(),[[50,40],[80,40]])

    def test_distortion_and_invalid_extrinsics_are_rejected(self):
        intrinsic={'model_type':'PINHOLE','image_width':100,'image_height':80,
            'projection_parameters':dict(zip(('fx','fy','cx','cy'),(100,80,50,40))),
            'distortion_parameters':{'k1':0}}
        extrinsic={'calibrated':True,'transform:':np.eye(4).ravel().tolist()}
        self.assertEqual(module.camera_parameters(intrinsic,extrinsic)['width'],100)
        intrinsic['distortion_parameters']['k1']=.1
        with self.assertRaises(ValueError):module.camera_parameters(intrinsic,extrinsic)
        intrinsic['distortion_parameters']['k1']=0
        extrinsic['transform:'][0]=-1
        with self.assertRaises(ValueError):module.camera_parameters(intrinsic,extrinsic)

    def test_overlapping_classes_are_not_negative_or_unambiguous_reference(self):
        points=np.array([[0,0,3],[0,0,5],[10,0,3]],dtype=float)
        bits,owner,_=module.class_membership(points,[obj('Pallet'),obj('Pedestrian')])
        self.assertEqual(bits.tolist(),[3,0,0])
        self.assertEqual(owner.tolist(),[-1,0,0])
        same_bits,same_owner,_=module.class_membership(points,[obj('Pallet'),obj('Pallet')])
        self.assertEqual(same_bits.tolist(),[1,0,0])
        self.assertEqual(same_owner[0],-1)

    def test_duplicate_predictions_cannot_double_the_matched_reference_count(self):
        pixels=np.array([[15,15],[16,15],[35,15],[36,15]])
        boxes=[{'obj_id':1,'bit':1},{'obj_id':2,'bit':1}]
        # One mask covering two equally supported objects cannot represent either instance.
        large=[square(1,10,40)]
        result=module.mask_support(large,np.arange(4),pixels,np.ones(4,dtype=np.uint8),
            np.array([1,1,2,2]),boxes,100,80,(0,),minimum_points=2)
        self.assertEqual(result['classes']['pallet']['radii']['0']['objects_matched_at_support_threshold'],0)
        # Two duplicate masks over a single object still yield one matched object.
        result=module.mask_support([square(1,10,20),square(2,10,20)],np.arange(2),pixels[:2],
            np.ones(2,dtype=np.uint8),np.array([1,1]),boxes[:1],100,80,(0,),minimum_points=2)
        self.assertEqual(result['classes']['pallet']['radii']['0']['objects_matched_at_support_threshold'],1)
        self.assertEqual(result['classes']['pallet']['candidates'],2)

    def test_near_duplicate_merged_masks_cannot_inflate_instance_support(self):
        pixels=np.array([[15,15],[16,15],[35,15],[36,15]])
        args=([square(1,10,40),square(2,9,41)],np.arange(4),pixels,np.ones(4,dtype=np.uint8),
              np.array([1,1,2,2]),[{'obj_id':1,'bit':1},{'obj_id':2,'bit':1}],100,80,(0,))
        loose=module.mask_support(*args,minimum_points=2,minimum_purity=0)
        strict=module.mask_support(*args,minimum_points=2,minimum_purity=.67)
        self.assertEqual(loose['classes']['pallet']['radii']['0']['objects_matched_at_support_threshold'],2)
        self.assertEqual(strict['classes']['pallet']['radii']['0']['objects_matched_at_support_threshold'],0)

    def test_pixel_tolerance_and_cross_class_coverage_have_separate_denominators(self):
        pixels=np.array([[15,15],[23,15],[18,15],[60,60]])
        bits=np.array([1,1,2,0],dtype=np.uint8)
        shapes=[square(1,10,20),square(2,45,55,label='person')]
        result=module.mask_support(shapes,np.arange(4),pixels,bits,np.array([1,1,2,0]),
            [{'obj_id':1,'bit':1},{'obj_id':2,'bit':2}],100,80,(0,3),minimum_points=1)
        pallet=result['classes']['pallet']
        self.assertEqual(pallet['reference_projected_points'],2)
        self.assertEqual(pallet['radii']['0']['covered_projected_points'],1)
        self.assertEqual(pallet['radii']['3']['covered_projected_points'],2)
        self.assertEqual(result['classes']['person']['reference_projected_points'],1)
        self.assertEqual(result['classes']['person']['radii']['0']['points_covered_by_other_class'],1)

    def test_receipt_rejects_candidates_or_archive_from_another_run(self):
        raw=b'{}'
        receipt={'prediction_json_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':'a'}
        module.verify_prediction_receipt(raw,receipt,{'archive_sha256':'a'})
        with self.assertRaises(ValueError):module.verify_prediction_receipt(b'changed',receipt,{'archive_sha256':'a'})
        with self.assertRaises(ValueError):module.verify_prediction_receipt(raw,receipt,{'archive_sha256':'b'})

    def test_archived_string_ids_are_accepted_without_modifying_the_source(self):
        shape=square('sam3-camera-0-pallet-0',10,20)
        result=module.mask_support([shape],np.array([0]),np.array([[15,15]]),np.array([1]),np.array([1]),
            [{'obj_id':1,'bit':1}],100,80,(0,),minimum_points=1)
        self.assertEqual(result['classes']['pallet']['radii']['0']['covered_projected_points'],1)
        self.assertEqual(shape['id'],'sam3-camera-0-pallet-0')


if __name__=='__main__':
    unittest.main()
