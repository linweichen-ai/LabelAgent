from copy import deepcopy
import unittest
import numpy as np

from tools.spatial_candidate_evidence import voxel_labels,nearest_supported_component,SpatialPointNovelty
from tools.tiled_candidate_utils import augment_candidates


def shape(identifier,x0,x1,y0=6,y1=14,score=1.):
 return {'id':identifier,'label':'pallet','points':[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],'holes':[],'score':score}


class SpatialEvidenceTests(unittest.TestCase):
 def sensor(self):
  pixels=np.array([[x,10] for x in range(10,16)]+[[x,10] for x in range(40,46)])
  xyz=np.array([[i*.01,0,1] for i in range(6)]+[[1+i*.01,0,1] for i in range(6)])
  return {'pixels':pixels,'xyz':xyz,'depths':xyz[:,2]}

 def test_diagonal_voxels_connect_across_zero_but_remote_points_stay_separate(self):
  points=np.array([[-.01,-.01,0],[.01,.01,.01],[.16,.16,.16],[2.,0,0]])
  before=points.copy();labels=voxel_labels(points)
  self.assertEqual(len(set(labels)),2)
  self.assertEqual(labels[0],labels[2]);self.assertNotEqual(labels[0],labels[3])
  np.testing.assert_array_equal(points,before)

 def test_foreground_component_is_nearest_supported_not_largest_background(self):
  points=np.array([[i*.01,0,1] for i in range(6)]+[[i*.01,0,4] for i in range(15)])
  chosen,count=nearest_supported_component(points,points[:,2])
  np.testing.assert_array_equal(chosen,np.arange(6));self.assertEqual(count,2)

 def test_near_isolated_outlier_is_not_foreground_and_empty_input_is_supported(self):
  points=np.array([[0,0,.3]]+[[i*.01,0,2] for i in range(6)])
  chosen,count=nearest_supported_component(points,points[:,2])
  np.testing.assert_array_equal(chosen,np.arange(1,7));self.assertEqual(count,1)
  chosen,count=nearest_supported_component(np.empty((0,3)),np.empty(0))
  self.assertEqual(len(chosen),0);self.assertEqual(count,0)

 def test_supported_evidence_does_not_disappear_above_previous_height_band(self):
  proposals=[shape(1,6,18),shape(2,38,48)]
  sensor=self.sensor();lifted=deepcopy(sensor);lifted['xyz'][:,2]+=3;lifted['depths']=lifted['xyz'][:,2].copy()
  a=SpatialPointNovelty([],proposals,80,20,sensor)
  b=SpatialPointNovelty([],proposals,80,20,lifted)
  for proposal in proposals:
   self.assertEqual(a.priority(proposal,(),'component_fraction'),b.priority(proposal,(),'component_fraction'))
  self.assertEqual(b.summary()['proposals_with_5_foreground_points'],2)

 def test_dynamic_component_coverage_changes_selection_without_changing_originals(self):
  baseline=[shape(i,x,x+2,0,2) for i,x in enumerate((0,25,55,72),1)]
  a,b,c=shape(5,6,16),shape(6,8,18,5,15,score=.98),shape(7,38,48,score=.9)
  proposals=[a,b,c];before=deepcopy(baseline+proposals)
  ordinary,_=augment_candidates(baseline,proposals,80,20,minimum_score=.5,addition_ratio=.5)
  context=SpatialPointNovelty(baseline,proposals,80,20,self.sensor())
  selected,info=augment_candidates(baseline,proposals,80,20,minimum_score=.5,addition_ratio=.5,
   priority_callback=context.callback('component_mass'))
  self.assertEqual([s['id'] for s in ordinary[4:]],[5,6])
  self.assertEqual([s['id'] for s in selected[4:]],[5,7])
  self.assertEqual(info['added_pallets'],2);self.assertEqual(selected[:4],baseline)
  self.assertEqual(baseline+proposals,before)

 def test_missing_support_is_soft_and_invalid_inputs_fail_closed(self):
  proposal=shape(1,6,18,score=.8)
  context=SpatialPointNovelty([], [proposal],80,20,{'pixels':np.empty((0,2),int),'xyz':np.empty((0,3)),'depths':np.empty(0)})
  self.assertEqual(context.priority(proposal,(),'component_fraction'),.4)
  with self.assertRaises(ValueError):context.callback('bad')
  with self.assertRaises(ValueError):voxel_labels(np.array([[float('nan'),0,0]]))
  with self.assertRaises(ValueError):SpatialPointNovelty([proposal],[proposal],80,20,self.sensor())


if __name__=='__main__':unittest.main()
