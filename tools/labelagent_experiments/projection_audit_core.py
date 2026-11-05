"""Read-only, hash-bound projection diagnostics; never used by generation."""
import json,sys
from pathlib import Path
import numpy as np,yaml,cv2
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def environment():
 cfg=read(ROOT/'task_workspaces/labelagent_projection_recovery_20261105/config.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
 import evaluate_projected_support as ev
 return cfg,ev
def frame_data(dataset,manifest,frame,ev):
 stamp=str(manifest['timestamps_ns'][frame]);annotation=json.loads(ev.checked_source(dataset,manifest,f'labels/{stamp}.json'));pcd=f'lidar/{stamp}.pcd';ev.checked_source(dataset,manifest,pcd);payload,fields,stride,count,_=ev.read_pcd(dataset/'source'/pcd);fields={f['name']:f for f in fields};types={7:'<f4',8:'<f8'}
 xyz=np.column_stack([np.ndarray((count,),dtype=types[fields[n]['type']],buffer=payload,offset=fields[n]['offset'],strides=(stride,)) for n in ['x','y','z']]);finite=np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)>.001);source_rows=np.flatnonzero(finite);xyz=xyz[finite];bits,owner,boxes=ev.class_membership(xyz,annotation['objects'])
 return {'stamp':stamp,'xyz':xyz,'raw_pcd_rows':source_rows,'bits':bits,'owner':owner,'boxes':boxes,'annotation':annotation,'pcd_fields':list(fields)}
def cameras(dataset,manifest,ev):
 extrinsics=yaml.safe_load(ev.checked_source(dataset,manifest,'calib/lidar2camera.yaml'));result={}
 for cam in range(4):
  text=ev.checked_source(dataset,manifest,f'calib/camera{cam}_calib.yaml').decode('utf-8');intrinsic=yaml.safe_load('\n'.join(s for s in text.splitlines() if not s.startswith('%YAML:')));result[cam]=ev.camera_parameters(intrinsic,extrinsics[f'lidar_to_camera_{cam}'])
 return result
def matrix_audit(shapes,data,source,pixels,ev):
 from source.agent.edit_agent import _rasterize
 ss=[s for s in shapes if s['label']=='pallet'];masks=[_rasterize(s,1920,1536) for s in ss];kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7));hits=np.array([cv2.dilate(m,kernel)[pixels[:,1],pixels[:,0]]!=0 for m in masks]);owned=data['owner'][source];semantic=data['bits'][source];eligible=[i for i,b in enumerate(data['boxes']) if b['bit']==1 and np.count_nonzero(owned==i+1)>=5];support=np.zeros((len(eligible),len(masks)));raw=support.copy();purity=support.copy();known=(semantic==1)&(owned>0)
 for j,h in enumerate(hits):
  total=int((h&known).sum())
  for k,i in enumerate(eligible):
   n=int((h&(owned==i+1)).sum());coverage=n/np.count_nonzero(owned==i+1);p=n/total if total else 0;raw[k,j]=coverage;purity[k,j]=p;support[k,j]=coverage if p>=.67 else 0
 pairs=ev.match(support,.5)
 return {'shapes':ss,'masks':masks,'hits':hits,'union_hits':np.any(hits,axis=0),'eligible':eligible,'support':support,'raw':raw,'purity':purity,'pairs':pairs,'known':known}
def other_box_front_hits(data,camera,indices,depth_slack=.15):
 # Ray-box intersections are a coarse proxy, not image depth truth.
 origin=np.linalg.inv(camera['transform'])[:3,3];p=data['xyz'][indices];direction=p-origin;point_z=p@camera['transform'][2,:3]+camera['transform'][2,3];result=[[] for _ in indices]
 for i,b in enumerate(data['boxes']):
  box=b['box'];o=(origin-box.center)@box.rotation;d=direction@box.rotation;parallel=np.abs(d)<1e-10;outside=parallel&(np.abs(o)>box.size/2+1e-8);safe=np.where(parallel,1,d);lo=(-box.size/2-o)/safe;hi=(box.size/2-o)/safe;tlo=np.where(parallel,-np.inf,np.minimum(lo,hi));thi=np.where(parallel,np.inf,np.maximum(lo,hi));enter=np.maximum(0,tlo.max(1));exit=thi.min(1);valid=(~outside.any(1))&(exit>=enter)&(exit>0)&(enter<1)&((1-enter)*point_z>depth_slack)&(data['owner'][indices]!=i+1)
  for row in np.flatnonzero(valid):result[row].append({'object_id':b['obj_id'],'label':box.label,'front_depth_delta_m':float((1-enter[row])*point_z[row]),'ray_enter_fraction':float(enter[row])})
 return result
