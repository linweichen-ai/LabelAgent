import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_instance_separation_20261105';out=ROOT/'outputs/labelagent_instance_separation_20261105'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if work.exists():raise FileExistsError('Preserve every earlier run')
work.mkdir();out.mkdir();repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
for name in ['source','tools','tests']:shutil.copytree(repo/name,work/'source_snapshot'/name,ignore=shutil.ignore_patterns('__pycache__','.git','weights','*.pt','*.pth','*.safetensors'))
cfg=json.loads((ROOT/'task_workspaces/labelagent_projection_recovery_20261105/config_wire_comparison.json').read_text(encoding='utf-8'));cfg.update(source_snapshot=str(work/'source_snapshot'),source_commit='ec41572babed04b57409e286174f942d1fd91a53',output=str(ROOT/'outputs/labelagent_instance_separation_comparison_20261105'),instance_generation=str(out/'instance_generation'),optimization_plan=str(out/'optimization_plan.json'))
save(work/'config.json',cfg)
plan={'scope':'two exposed same-capture images; development comparison, not independent accuracy','incumbent':'labelagent_context_comparison_20261105','preserve_all27_PAL_and3_person_masks':True,'initial_support':13,'initial_points':490,'initial_convergence':'6/6','original_limits':{'rounds':3,'operations_per_round':3,'cumulative_points':32,'count':36,'PAL_union_pixels':122511.25,'initial_union_pixels':122302},'references_for_generation':False,
 'parents':'merge overlapping incumbent predicted PAL components;4 groups per camera,2 blue-material priorities plus2 largest other groups; no reference positions',
 'trained_model':{'same_checkpoint':'d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291','CPU_forwards':16,'two_context_pads':[32,96],'input_height_width':[448,576],'PAL_argmax_probability_min':.2,'mask_probability_min':.5,'promote_null':False,'outputs':'same-forward standard panoptic plus independent raw PAL query masks; experimental hypotheses, no production threshold change','pixel_exact_conversion_required':True},
 'SAM':'visible image row-seam intervals, up to2 per group; up to3 foreground clicks across visible thirds, up to4 adjacent-layer negative clicks; no dense prior; complete raw response retained or rejected, never intersected with old union',
 'semantic':'unchanged neutral2-view PAL and subject guard, plus actual single-separable-visible-base check; blank/robot/stack controls must pass; no count given to model',
 'selection':'whole-mask append only; original global limits, incumbent person radius3 guard and new-proposal NMS unchanged; strict support gain over13 mandatory; all0/3/6 pixel coverage checked','annotation_writes':0,'training_updates':0,'default_install':False}
save(out/'optimization_plan.json',plan)
save(out/'resource_preflight.json',{'disk_free_bytes':shutil.disk_usage(ROOT).free,'model_weights_copied_for_execution':False})
save(out/'incumbent_binding.json',{'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'outputs/labelagent_context_comparison_20261105').glob('*/*/result.json')}})
print(out)
