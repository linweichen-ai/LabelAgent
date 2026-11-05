"""New independent snapshot; never mutate sealed results from earlier runs."""
import json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
work=ROOT/'task_workspaces/labelagent_semantic_recovery_20261105'
out=ROOT/'outputs/labelagent_semantic_recovery_20261105'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if work.exists():raise FileExistsError('Keep all previous experiments')
work.mkdir();out.mkdir()
repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
for name in ['source','tools','tests']:
 shutil.copytree(repo/name,work/'source_snapshot'/name,ignore=shutil.ignore_patterns('__pycache__','.git','weights','*.pt','*.pth','*.safetensors'))
cfg=json.loads((ROOT/'task_workspaces/labelagent_recovery_20261105/config_preserved.json').read_text(encoding='utf-8'))
cfg.update(source_snapshot=str(work/'source_snapshot'),output=str(ROOT/'outputs/labelagent_semantic_comparison_20261105'),source_commit='65656fad8c54a21a900f91e7da242542c1a5b364',instance_generation=str(out/'instance_generation'),optimization_plan=str(out/'optimization_plan.json'))
save(work/'config.json',cfg)
old=json.loads((ROOT/'outputs/labelagent_recovery_20261105/optimization_plan.json').read_text(encoding='utf-8'))
plan={k:old[k] for k in ['metric_unchanged','original_metric_sha256','acceptance_unchanged','limits','protection']}
plan.update(scope='exposed two images; exploratory engineering iteration, not generalization',prior='labelagent_preserved_model_comparison_20261105',
 model_generation={'fixed_grid_wh':[320,256],'x':[0,266,533,800,1066,1333,1600],'y':[0,256,512,768,1024,1280], 'forwards':84,'same_trained_checkpoint':True,'postprocessing_unchanged':True,'no_reference_positions':True,'whole_mask_selection_only':True},
 semantic_diagnosis={'ablations':['old_schema_without_text_contract','explicit_schema_contract','neutral_caption_only'],'controls':['gray','white','raw_pallet_crop','raw_floor_crop','raw_robot_crop'],'actual_model_requests':True,'no_backend_or_shared_service_mutation':True},
 floor_optimization={'three_rounds':True,'three_ops_per_round':True,'max_cumulative_points':32,'spatially_local_SAM':'one real request per operation; raw prior/image/points cropped together; output mapped back without budget clipping','approval':'same independent per-gap ground approval and complete100% gap coverage; next-round actual review remains mandatory'})
save(out/'optimization_plan.json',plan)
# New generator performs genuine forwards on a smaller, fixed, reference-free grid.
src=(ROOT/'tools/labelagent_live_handoff/generate_recovery_model_grid.py').read_text(encoding='utf-8')
src=src.replace('labelagent_recovery_20261105','labelagent_semantic_recovery_20261105').replace('for y in [0,341,682,1024]:','for y in [0,256,512,768,1024,1280]:').replace('for x in [0,426,853,1280]:','for x in [0,266,533,800,1066,1333,1600]:').replace('x+640,y+512','x+320,y+256').replace('y:y+512,x:x+640','y:y+256,x:x+320').replace('[x,y,640,512]','[x,y,320,256]').replace("calls[0]==32","calls[0]==84")
(work/'generate_fine_grid.py').write_text(src,encoding='utf-8')
# ROOT for copied helper is explicit because this independent workspace has different depth.
src=(work/'generate_fine_grid.py').read_text(encoding='utf-8').replace("ROOT=Path(__file__).resolve().parents[2]","ROOT=Path("+repr(str(ROOT))+")")
(work/'generate_fine_grid.py').write_text(src,encoding='utf-8')
save(out/'snapshot_identity.json',{'files':{str(p.relative_to(work/'source_snapshot')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (work/'source_snapshot').rglob('*.py')},'original_annotations_writes':0})
print(out)
