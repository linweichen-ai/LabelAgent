"""Unlabelled local 3D components; no reference reads or shape mutation."""
from copy import deepcopy
import math

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tools.evaluate_projected_support import project_visible
from tools.tiled_candidate_utils import _mask_record,_iou
from source.agent.edit_agent import _rasterize


SPATIAL_RECIPE={'voxel_size':.15,'voxel_connectivity':26,'minimum_component_points':5,
 'minimum_mask_points':5,'mask_dilation_pixels':3,'saturation_points':5,
 'height_limit':None,'floor_points_removed':False,
 'foreground':'nearest median camera-depth connected component having >=5 raw points',
 'component_coherence':'foreground component raw points / all raw points in candidate',
 'base':'original confidence*(1-max_mask_IoU_to_original_pallets)',
 'unknown_factor':.5,'evidence_factor_range':[.5,1.],
 'component_coverage':'foreground component indices of original and actually selected pallet candidates',
 'all_point_coverage':'all masked raw point indices of original and actually selected pallet candidates'}


def voxel_labels(points,voxel_size=.15):
 points=np.asarray(points,dtype=float)
 if (points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all()
     or not math.isfinite(voxel_size) or voxel_size<=0):raise ValueError('Finite XYZ and positive voxel size required')
 if not len(points):return np.empty(0,dtype=np.int32)
 voxels,inverse=np.unique(np.floor(points/voxel_size).astype(np.int64),axis=0,return_inverse=True)
 pairs=cKDTree(voxels).query_pairs(1.01,p=np.inf,output_type='ndarray')
 graph=coo_matrix((np.ones(len(pairs),np.uint8),(pairs[:,0],pairs[:,1])),shape=(len(voxels),len(voxels))).tocsr()
 _,labels=connected_components(graph,directed=False)
 return labels[inverse]


def nearest_supported_component(points,depths,minimum_points=5):
 points=np.asarray(points,dtype=float);depths=np.asarray(depths,dtype=float)
 if type(minimum_points) is not int or minimum_points<1:raise ValueError('Positive integer point minimum required')
 if depths.shape!=(len(points),) or not np.isfinite(depths).all() or np.any(depths<=0):raise ValueError('Positive camera depths required')
 labels=voxel_labels(points,SPATIAL_RECIPE['voxel_size'])
 groups=[np.flatnonzero(labels==identifier) for identifier in np.unique(labels)]
 supported=[group for group in groups if len(group)>=minimum_points]
 if not supported:return np.empty(0,dtype=np.int64),len(supported)
 chosen=min(supported,key=lambda group:(float(np.median(depths[group])),-len(group),int(group[0])))
 return chosen,len(supported)


def projected_spatial_sensor(points,camera):
 indices,pixels=project_visible(points,camera,cell_size=8,depth_slack=.15,near=.2)
 visible=np.asarray(points)[indices]
 depths=visible@camera['transform'][2,:3]+camera['transform'][2,3]
 return {'pixels':pixels,'xyz':visible,'depths':depths}


class SpatialPointNovelty:
 def __init__(self,baseline,proposals,width,height,sensor):
  shapes=[*baseline,*proposals]
  if len({s['id'] for s in shapes})!=len(shapes):raise ValueError('Unique candidate IDs required')
  self.proposals=deepcopy(proposals);self.baseline_ids={s['id'] for s in baseline}
  self.pixels=np.asarray(sensor['pixels']);self.xyz=np.asarray(sensor['xyz']);self.depths=np.asarray(sensor['depths'])
  if len(self.pixels)!=len(self.xyz) or len(self.xyz)!=len(self.depths):raise ValueError('Aligned sensor arrays required')
  self.masks={};self.components={};self.facts={};self.seeds={mode:np.zeros(len(self.pixels),bool) for mode in ('all','component')}
  r=SPATIAL_RECIPE['mask_dilation_pixels'];kernel=np.ones((2*r+1,2*r+1),np.uint8)
  for shape in shapes:
   if shape['label']!='pallet':continue
   mask=cv2.dilate(_rasterize(shape,width,height),kernel)
   point_mask=mask[self.pixels[:,1],self.pixels[:,0]]!=0
   rows=np.flatnonzero(point_mask)
   chosen,groups=nearest_supported_component(self.xyz[rows],self.depths[rows],SPATIAL_RECIPE['minimum_component_points'])
   component=np.zeros(len(self.pixels),bool);component[rows[chosen]]=True
   self.masks[shape['id']]=point_mask;self.components[shape['id']]=component
   self.facts[shape['id']]={'all_points':len(rows),'component_points':len(chosen),'supported_components':groups,
     'coherence':len(chosen)/len(rows) if len(rows) else 0.}
   if shape['id'] in self.baseline_ids:
    self.seeds['all']|=point_mask;self.seeds['component']|=component
  old=[_mask_record(s,width,height) for s in baseline if s['label']=='pallet']
  self.base={}
  for shape in proposals:
   record=_mask_record(shape,width,height)
   self.base[shape['id']]=float(shape['score'])*(1-max((_iou(record,v) for v in old),default=0.))
  self.tokens={mode:() for mode in self.seeds};self.covered={mode:v.copy() for mode,v in self.seeds.items()}

 def priority(self,shape,selected,policy):
  if policy not in {'all_dynamic','component_mass','component_fraction'}:raise ValueError('Unknown spatial policy')
  mode='all' if policy=='all_dynamic' else 'component'
  token=tuple(s['id'] for s in selected if s['label']=='pallet' and s['id'] not in self.baseline_ids)
  masks=self.masks if mode=='all' else self.components
  if token!=self.tokens[mode]:
   self.covered[mode]=self.seeds[mode].copy()
   for identifier in token:self.covered[mode]|=masks[identifier]
   self.tokens[mode]=token
  own=masks[shape['id']];count=int(np.count_nonzero(own));factor=SPATIAL_RECIPE['unknown_factor']
  if count>=SPATIAL_RECIPE['minimum_mask_points']:
   novel=int(np.count_nonzero(own&~self.covered[mode]))
   denominator=SPATIAL_RECIPE['saturation_points']+(count if policy=='component_fraction' else novel)
   coherence=1. if mode=='all' else self.facts[shape['id']]['coherence']
   factor=.5+.5*coherence*novel/denominator
  return self.base[shape['id']]*factor

 def callback(self,policy):
  if policy not in {'all_dynamic','component_mass','component_fraction'}:raise ValueError('Unknown spatial policy')
  return lambda shape,selected:self.priority(shape,selected,policy)

 def summary(self):
  return {'raw_visible_points':len(self.pixels),'proposals':len(self.proposals),
   'proposals_with_5_mask_points':sum(self.facts[s['id']]['all_points']>=5 for s in self.proposals),
   'proposals_with_5_foreground_points':sum(self.facts[s['id']]['component_points']>=5 for s in self.proposals),
   'proposals_with_multiple_supported_components':sum(self.facts[s['id']]['supported_components']>1 for s in self.proposals),
   'component_coherence_quantiles':np.quantile([self.facts[s['id']]['coherence'] for s in self.proposals],[.1,.5,.9]).tolist() if self.proposals else [],
   'raw_points_covered_by_original_masks':int(np.count_nonzero(self.seeds['all'])),
   'raw_points_covered_by_original_components':int(np.count_nonzero(self.seeds['component'])),
   'reference_labels_used':False}
