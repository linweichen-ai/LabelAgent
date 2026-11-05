"""Post-freeze stage diagnosis using the original scorer; never selection."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_semantic_recovery_20261105';out=ROOT/'outputs/labelagent_semantic_recovery_20261105/stage_audit'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 comparison=ROOT/'outputs/labelagent_semantic_comparison_20261105'
 if not (comparison/'completed.json').exists():raise RuntimeError('Do not score before all comparison predictions are frozen')
 if (out/'completed.json').exists():raise FileExistsError('Preserve stage audit')
 cfg=read(work/'config_semantic.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
 import numpy as np,yaml
 import evaluate_projected_support as ev
 dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');stamp=str(manifest['timestamps_ns'][76]);annotation=json.loads(ev.checked_source(dataset,manifest,f'labels/{stamp}.json'))
 pcd=f'lidar/{stamp}.pcd';ev.checked_source(dataset,manifest,pcd);payload,fields,stride,count,_=ev.read_pcd(dataset/'source'/pcd);fields={f['name']:f for f in fields};types={7:'<f4',8:'<f8'}
 xyz=np.column_stack([np.ndarray((count,),dtype=types[fields[n]['type']],buffer=payload,offset=fields[n]['offset'],strides=(stride,)) for n in ['x','y','z']]);xyz=xyz[np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)>.001)]
 bits,owner,boxes=ev.class_membership(xyz,annotation['objects']);extrinsics=yaml.safe_load(ev.checked_source(dataset,manifest,'calib/lidar2camera.yaml'));generation=Path(cfg['instance_generation']);verified=read(generation/'verification_progress.json')['rows'];rows=[]
 for cam in [0,1]:
  case=f'camera{cam}_frame76';base=read(comparison/'A_current'/case/'result.json')['shapes'];raw=[s for p in sorted((generation.parent/'fixed_grid'/case).glob('*/prediction.json')) for s in read(p)['PAL_shapes']]
  coarse=[s for p in sorted((ROOT/'outputs/labelagent_recovery_20261105/fixed_grid'/case).glob('*/prediction.json')) for s in read(p)['PAL_shapes']]
  stages={'old_raw_coarse_grid':coarse,'new_raw_fine_grid':raw,'combined_unfiltered_raw':coarse+raw,
   'neutral_two_view_evidence':[r['row']['shape'] for r in verified if r['row']['case']==case and r['passed']],
   'final_whole_mask_budget_selection':read(generation/case/'C_children.json')}
  text=ev.checked_source(dataset,manifest,f'calib/camera{cam}_calib.yaml').decode('utf-8');intrinsic=yaml.safe_load('\n'.join(s for s in text.splitlines() if not s.startswith('%YAML:')));source,pixels=ev.project_visible(xyz,ev.camera_parameters(intrinsic,extrinsics[f'lidar_to_camera_{cam}']))
  for stage,proposals in stages.items():
   scored=ev.mask_support(base+proposals,source,pixels,bits,owner,boxes,1920,1536,radii=(3,),minimum_points=5,minimum_purity=.67)
   rows.append({'case':case,'stage':stage,'proposals':len(proposals),'pallet':scored['classes']['pallet']['radii']['3'],
    'person':scored['classes']['person']['radii']['3'],'five_original_misses':[o for o in scored['objects'] if o['label']=='pallet' and o['obj_id'] in ([42,48] if cam==0 else [47,48,49])]})
 save(out/'completed.json',{'status':'complete','scope':'post-freeze diagnosis; raw stages do NOT satisfy deployed count/area/semantic gates and are not adopted predictions','reference_not_used_for_selection':True,'original_scorer':True,'rows':rows})
 print(json.dumps({'stage_audit_complete':True,'rows':len(rows)}),flush=True)
if __name__=='__main__':main()
