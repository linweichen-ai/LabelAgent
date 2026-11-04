"""Label-free raw LiDAR evidence; no semantic reference annotations are read."""
from copy import deepcopy
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

from tools.import_capture import read_pcd
from tools.evaluate_projected_support import checked_source,camera_parameters,project_visible
from tools.tiled_candidate_utils import _mask_record,_iou
from source.agent.edit_agent import _rasterize

PLANE_RECIPE={'minimum_radius':.5,'maximum_radius':20.,'minimum_z':-4.,'maximum_z':-.15,
    'lower_quantile':.5,'sample_limit':6000,'trials':256,'seed':0,'distance_threshold':.04,
    'maximum_tilt_degrees':15.,'minimum_origin_depth':.2,'maximum_origin_depth':4.,
    'minimum_pool_points':100,'minimum_inlier_fraction':.55,'minimum_xy_span':2.}
POINT_RECIPE={'cell_size':8,'depth_slack':.15,'near':.2,'mask_dilation_pixels':3,
    'minimum_support_points':5,'floor_band':.04,'low_object_height':[.04,.45],'tall_height':.7}


class ReferenceReadAudit:
    """Fail an experimental predictor if it opens any reference-label file."""
    def __init__(self,dataset):
        self.root=(Path(dataset)/'source').resolve()
        self.active=False;self.sources=set();self.denied=[]

    def start(self):
        def observe(event,args):
            if event!='open' or not self.active or not args or not isinstance(args[0],(str,bytes)):
                return
            path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
            if not path.is_relative_to(self.root):return
            relative=path.relative_to(self.root)
            if relative.parts and relative.parts[0].lower() in {'labels','ali_label','wilddet3d-labels'}:
                self.denied.append(relative.as_posix())
                raise PermissionError('Predictor may not read reference annotations')
            self.sources.add(relative.as_posix())
        sys.addaudithook(observe)
        self.active=True

    def stop(self):
        self.active=False


def read_xyz(dataset,manifest,frame):
    relative=f"lidar/{manifest['timestamps_ns'][frame]}.pcd"
    checked_source(dataset,manifest,relative)
    raw,fields,stride,count,_=read_pcd(Path(dataset)/'source'/relative)
    fields={f['name']:f for f in fields}
    xyz=np.column_stack([np.ndarray((count,),dtype={7:'<f4',8:'<f8'}[fields[k]['type']],
        buffer=raw,offset=fields[k]['offset'],strides=(stride,)) for k in ('x','y','z')])
    xyz=xyz[np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)>.001)].astype(np.float64)
    return xyz,relative


def sensor_cameras(dataset,manifest,cameras):
    relative='calib/lidar2camera.yaml'
    extrinsics=yaml.safe_load(checked_source(dataset,manifest,relative))
    paths=[relative];result={}
    for camera in cameras:
        path=f'calib/camera{camera}_calib.yaml';paths.append(path)
        text=checked_source(dataset,manifest,path).decode('utf-8')
        intrinsic=yaml.safe_load('\n'.join(s for s in text.splitlines() if not s.startswith('%YAML:')))
        result[camera]=camera_parameters(intrinsic,extrinsics[f'lidar_to_camera_{camera}'])
    return result,paths


def fit_support_plane(points,recipe=None):
    p=deepcopy(PLANE_RECIPE if recipe is None else recipe)
    if (set(p)!=set(PLANE_RECIPE) or not 0<p['lower_quantile']<1 or p['distance_threshold']<=0
            or p['minimum_pool_points']<3 or p['trials']<1 or p['sample_limit']<3):
        raise ValueError('Invalid plane fitting recipe')
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3:raise ValueError('Expected XYZ points')
    points=points[np.isfinite(points).all(axis=1)]
    radius=np.linalg.norm(points[:,:2],axis=1)
    domain=points[(radius>=p['minimum_radius'])&(radius<=p['maximum_radius'])
        &(points[:,2]>=p['minimum_z'])&(points[:,2]<=p['maximum_z'])]
    unavailable={'usable':False,'domain_points':len(domain),'plane':None}
    if len(domain)<p['minimum_pool_points']:return {**unavailable,'reason':'insufficient_lower_points'}
    pool=domain[domain[:,2]<=np.quantile(domain[:,2],p['lower_quantile'])]
    if len(pool)<p['minimum_pool_points']:return {**unavailable,'reason':'insufficient_fitting_pool'}
    rng=np.random.default_rng(p['seed'])
    if len(pool)>p['sample_limit']:pool=pool[rng.choice(len(pool),p['sample_limit'],replace=False)]
    cosine=math.cos(math.radians(p['maximum_tilt_degrees']))
    best=None
    for _ in range(p['trials']):
        a,b,c=pool[rng.choice(len(pool),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-10:continue
        normal/=length
        if normal[2]<0:normal=-normal
        offset=-float(normal@a)
        if normal[2]<cosine or not p['minimum_origin_depth']<=offset/normal[2]<=p['maximum_origin_depth']:continue
        distances=np.abs(pool@normal+offset);inliers=distances<=p['distance_threshold']
        key=(int(inliers.sum()),-float(np.sum(distances[inliers]**2)))
        if best is None or key>best[0]:best=(key,inliers)
    if best is None:return {**unavailable,'reason':'no_supported_horizontal_plane'}
    if best[0][0]/len(pool)<p['minimum_inlier_fraction']:
        return {**unavailable,'reason':'insufficient_plane_fraction','fit_fraction':best[0][0]/len(pool)}
    inliers=pool[best[1]]
    if np.min(np.ptp(inliers[:,:2],axis=0))<p['minimum_xy_span']:
        return {**unavailable,'reason':'insufficient_spatial_span'}
    coefficients=np.linalg.lstsq(np.column_stack((inliers[:,:2],np.ones(len(inliers)))), -inliers[:,2],rcond=None)[0]
    normal=np.array([*coefficients[:2],1.]);length=np.linalg.norm(normal)
    normal/=length;offset=float(coefficients[2]/length)
    distances=np.abs(pool@normal+offset);good=distances<=p['distance_threshold']
    if (normal[2]<cosine or not p['minimum_origin_depth']<=offset/normal[2]<=p['maximum_origin_depth']
            or float(np.mean(good))<p['minimum_inlier_fraction']):
        return {**unavailable,'reason':'refinement_failed'}
    return {'usable':True,'plane':[*normal.tolist(),offset],'domain_points':len(domain),
        'sampled_pool_points':len(pool),'fit_fraction':float(np.mean(good)),
        'origin_depth':offset/float(normal[2]),'tilt_degrees':math.degrees(math.acos(float(normal[2]))),
        'median_inlier_residual':float(np.median(distances[good])),
        'xy_span':np.ptp(pool[good,:2],axis=0).tolist(),
        'interpretation':'lower dominant near-horizontal plane candidate; not confirmed ground truth'}


def projected_sensor(points,camera,plane):
    indices,pixels=project_visible(points,camera,**{k:POINT_RECIPE[k] for k in ('cell_size','depth_slack','near')})
    heights=points[indices]@np.array(plane['plane'][:3])+plane['plane'][3] if plane['usable'] else None
    return {'pixels':pixels,'heights':heights,'plane_usable':plane['usable']}


def candidate_evidence(shape,width,height,sensor):
    if not sensor['plane_usable']:return {'supported':False,'points':0,'reason':'plane_unavailable'}
    mask=_rasterize(shape,width,height)
    radius=POINT_RECIPE['mask_dilation_pixels']
    mask=cv2.dilate(mask,np.ones((2*radius+1,2*radius+1),np.uint8))
    pixels=sensor['pixels'];h=sensor['heights'][mask[pixels[:,1],pixels[:,0]]!=0]
    if len(h)<POINT_RECIPE['minimum_support_points']:
        return {'supported':False,'points':len(h),'reason':'insufficient_projected_points'}
    lo,hi=POINT_RECIPE['low_object_height']
    return {'supported':True,'points':len(h),'height_quantiles':np.quantile(h,[.1,.5,.9]).tolist(),
        'floor_fraction':float(np.mean(np.abs(h)<=POINT_RECIPE['floor_band'])),
        'low_object_fraction':float(np.mean((h>=lo)&(h<=hi))),
        'tall_fraction':float(np.mean(h>POINT_RECIPE['tall_height']))}


def height_priorities(seed,proposals,width,height,evidence,mode):
    if mode not in {'positive','penalty'}:raise ValueError('Unknown raw height priority')
    old=[_mask_record(s,width,height) for s in seed if s['label']=='pallet']
    result={}
    for shape in proposals:
        record=_mask_record(shape,width,height)
        novelty=1-max((_iou(record,b) for b in old),default=0)
        facts=evidence[shape['id']]
        factor=1.
        if facts['supported']:
            factor=(1+facts['low_object_fraction'] if mode=='positive' else
                (1-.75*facts['floor_fraction'])*(1-.75*facts['tall_fraction']))
        # Uniformly rescale positive priorities into the legacy [0,1] domain.
        # This preserves their ordering and does not change model confidence.
        result[shape['id']]=float(shape.get('score',0))*novelty*factor/(2 if mode=='positive' else 1)
    return result
