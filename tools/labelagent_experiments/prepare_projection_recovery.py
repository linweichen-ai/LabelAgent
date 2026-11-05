"""Freeze an append-only experiment before consulting diagnostic references."""
import json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_projection_recovery_20261105';out=ROOT/'outputs/labelagent_projection_recovery_20261105'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if work.exists():raise FileExistsError('Preserve every prior run')
work.mkdir();out.mkdir()
repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
for name in ['source','tools','tests']:
 shutil.copytree(repo/name,work/'source_snapshot'/name,ignore=shutil.ignore_patterns('__pycache__','.git','weights','*.pt','*.pth','*.safetensors'))
old=ROOT/'task_workspaces/labelagent_semantic_recovery_20261105/config_context.json';cfg=json.loads(old.read_text(encoding='utf-8'));cfg.update(source_snapshot=str(work/'source_snapshot'),source_commit='7fd12975edcbcef93bb89c126c2fe209e13feb2f',output=str(ROOT/'outputs/labelagent_projection_retained_comparison_20261105'),instance_generation=str(out/'instance_generation'),optimization_plan=str(out/'optimization_plan.json'))
save(work/'config.json',cfg)
plan={'scope':'two exposed images; all references diagnostic only; no generalization claim','incumbent':'labelagent_context_comparison_20261105','reference_scorer_unchanged':{'radius':3,'minimum_points':5,'purity':.67,'one_to_one':.5,'zbuffer_cell':8,'depth_slack':.15},
 'acceptance':{'all_27_original_incumbent_PAL_masks_kept':True,'all_incumbent_person_masks_kept':True,'initial_pallet_support':13,'initial_pallet_points':490,'initial_review_convergence':'6/6','PAL_count_limit_from_original_A':36,'PAL_union_limit_from_original_A':122511.25,'initial_union':122302,'available_extra_union_pixels':209.25,'rounds':3,'operations_per_round':3,'cumulative_points':32,'default_adoption':False},
 'generation':{'reference_labels_positions_points_access':False,'source':'incumbent predicted PAL components and32+84 fixed-grid original trained masks','SAM_groups':8,'per_group_click_fractions':[.2,.5,.8],'SAM_max_real_calls':24,'group_selection':'largest predicted PAL components with width>=24,height>=36,area>=500;4 per camera','SAM_prompt':'one interior positive and up to2 neighboring-band negatives on actual context crop; no inherited group mask','selection':'complete model outputs only; reject entire mask on original global count/area limit or incumbent predicted-person radius3 overlap; no trimming','semantic_evidence':'unchanged neutral tight and natural context captions and subject guard; no target-class prompt induction'},
 'diagnosis':{'new_point_gains_and_losses':'exact original source-index set differences; instance matrices,purity and matching lineage','projection_checks':['source/MCAP exact timestamp join','physical exposure clock evidence availability','all4 cameras of same frame','frames74..78 fixed neighborhood','source calibration rigidity and hashes','LiDAR zbuffer sensitivity for diagnosis only','ray-versus-other-3D-reference-box front intersection proxy'],'do_not_change_original_visibility_or_reference_to_pass':True},
 'protection':{'annotation_writes':0,'training_updates':0,'old_results_preserved':True}}
save(out/'optimization_plan.json',plan)
save(out/'incumbent_binding.json',{'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'outputs/labelagent_context_comparison_20261105').glob('*/*/result.json')},'foreground_source':str(ROOT/'outputs/labelagent_context_comparison_20261105')})
print(out)
