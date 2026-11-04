"""Post-hoc upper bound only; never imported by proposal generators."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'task_workspaces/labelagent_efficiency_20261104/source_snapshot_partial'),str(ROOT/'task_workspaces/labelagent_efficiency_20261104/source_snapshot_partial/tools'),str(ROOT/'task_workspaces/cvat_mask2former_20261103/deps')]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 import cv2,numpy as np,yaml
 import evaluate_projected_support as ev
 from source.agent.edit_agent import _rasterize
 from PIL import Image
 cfg=read(ROOT/'task_workspaces/labelagent_efficiency_20261104/config_partial.json');dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json')
 stamp=str(manifest['timestamps_ns'][76]);annotation=json.loads(ev.checked_source(dataset,manifest,f'labels/{stamp}.json'))
 path=f'lidar/{stamp}.pcd';ev.checked_source(dataset,manifest,path);payload,fields,stride,count,_=ev.read_pcd(dataset/'source'/path);fields={f['name']:f for f in fields};types={7:'<f4',8:'<f8'}
 xyz=np.column_stack([np.ndarray((count,),dtype=types[fields[n]['type']],buffer=payload,offset=fields[n]['offset'],strides=(stride,)) for n in ['x','y','z']]);xyz=xyz[np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)>.001)]
 bits,owner,boxes=ev.class_membership(xyz,annotation['objects']);extrinsics=yaml.safe_load(ev.checked_source(dataset,manifest,'calib/lidar2camera.yaml'))
 original=ROOT/'outputs/labelagent_hybrid_verified_20261104';rows=[]
 for cam in [0,1]:
  case=f'camera{cam}_frame76';w,h=Image.open(original/'inputs'/(case+'.png')).size
  text=ev.checked_source(dataset,manifest,f'calib/camera{cam}_calib.yaml').decode('utf-8');intrinsic=yaml.safe_load('\n'.join(s for s in text.splitlines() if not s.startswith('%YAML:')))
  source,pixels=ev.project_visible(xyz,ev.camera_parameters(intrinsic,extrinsics[f'lidar_to_camera_{cam}']))
  mask=np.zeros((h,w),np.uint8)
  for s in read(original/'A_current'/case/'result.json')['shapes']:
   if s['label']=='pallet':mask|=_rasterize(s,w,h)
  dilated=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)));hit=dilated[pixels[:,1],pixels[:,0]]!=0
  old=read(original/'A_current/evaluation/per_image.json')[cam]['objects'];missing={r['obj_id'] for r in old if r['label']=='pallet' and r['best_candidate_supported_coverage']<.5}
  for i,b in enumerate(boxes):
   if b['obj_id'] not in missing or b['bit']!=1:continue
   selected=owner[source]==i+1;n=int(selected.sum())
   if n<5:continue
   coverage=float(hit[selected].mean());rows.append({'case':case,'obj_id':b['obj_id'],'points':n,'original_union_coverage_upper_bound':coverage,'subset_can_reach_support_half':coverage>=.5,'purity_and_one_to_one_ignored_for_upper_bound':True})
 out=ROOT/'outputs/labelagent_efficiency_20261104/diagnosis/subset_upper_bound.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps({'status':'complete','reference_use':'post-hoc diagnosis only, never prediction selection','radius':3,'support_threshold':.5,'rows':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(out.read_text(encoding='utf-8'))
if __name__=='__main__':main()
