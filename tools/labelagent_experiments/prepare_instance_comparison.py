"""Freeze complete image-only additions before unchanged integrated review."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent
work=ROOT/'task_workspaces/labelagent_instance_separation_20261105';parent=ROOT/'outputs/labelagent_instance_separation_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 new=parent/'instance_generation_consistency';combined=parent/'instance_generation_combined';source=ROOT/'outputs/labelagent_context_comparison_20261105';out=ROOT/'outputs/labelagent_instance_separation_comparison_20261105'
 assert read(new/'completed.json')['status']=='complete'
 if out.exists() or combined.exists():raise FileExistsError('Preserve frozen comparison')
 out.mkdir();combined.mkdir();old=Path(read(ROOT/'task_workspaces/labelagent_semantic_recovery_20261105/config_context.json')['instance_generation'])
 for cam in [0,1]:
  case=f'camera{cam}_frame76';save(combined/case/'C_children.json',read(old/case/'C_children.json')+read(new/case/'additional_children.json'));save(combined/case/'B_children.json',read(old/case/'B_children.json'))
 save(combined/'generation_protocol.json',{'incumbent_children_source':str(old),'new_complete_additions':str(new),'incumbent_all_foreground_kept':True,'new_selection_sha256':hashlib.sha256((new/'selection.json').read_bytes()).hexdigest(),'no_reference_inputs':True,'no_budget_mask_trimming':True})
 save(combined/'completed.json',{'status':'complete','new_generation':read(new/'completed.json')})
 cfg=read(work/'config.json');cfg.update(output=str(out),instance_generation=str(combined),generation_source=str(new));save(work/'config_comparison.json',cfg)
 for folder in ['mask2former','sam_text','refinement','inputs','gap_action_cache','A_current','B_sam7']:shutil.copytree(source/folder,out/folder)
 save(out/'agent_runs.json',[r for r in read(source/'agent_runs.json') if r['arm'] in ['A_current','B_sam7']]);save(out/'model_trace.json',read(source/'model_trace.json'))
 save(out/'cache_source_binding.json',{'source':str(source),'A_B_runs_reused':4,'C_new_integrated_runs':2,'C_initial_floor_is_original_pre_review_seed':True,'incumbent_foreground_preserved':True})
 save(out/'protocol_amendment.json',{'acceptance_unchanged':True,'rounds':3,'operations_per_round':3,'cumulative_points':32,'original_A_global_PAL_count_limit':36,'original_A_global_union_limit':122511.25,'incumbent_all27_PALs_kept':True,'required_support_strictly_above13':True,'new_candidate_generation':'paired raw trained queries and visible-seam multi-point SAM, no priors or reference points','no_reference_or_visibility_change':True,'no_budget_clipping':True,'no_default_install':True})
 runner=(helpers/'run_wire_projection_comparison.py').read_text(encoding='utf-8')
 runner=runner.replace("sha(Path(__file__).with_name('generate_wire_append.py'))","sha(Path(__file__).with_name('review_instance_consistency.py'))")
 runner=runner.replace("'new_fixed_model_grid_forwards':0","'new_fixed_model_grid_forwards':16").replace("'real_mask2former_forwards_source':126","'real_mask2former_forwards_source':2").replace("'real_mask2former_forwards_source':126","'real_mask2former_forwards_source':2")
 runner=runner.replace("'diagnostic_only_mask2former_forwards':0","'diagnostic_only_mask2former_forwards':16")
 runner=runner.replace("'wire_geometry_policy_sha256':", "'instance_geometry_sha256':sha(Path(__file__).with_name('instance_separation_geometry.py')),\n            'instance_evidence_sha256':sha(Path(__file__).with_name('instance_evidence_views.py')),\n            'instance_consistency_sha256':sha(Path(__file__).with_name('instance_identity_consistency.py')),\n            'wire_geometry_policy_sha256':")
 runner=runner.replace("Retain original SAM masks. Select entire untrimmed fixed-grid SAM masks using image evidence and trained model overlap; reject entire proposals on original count/area budget overflow; never clip to baseline union. No reference-based selection.","Retain every incumbent foreground mask. Append only complete new raw-query or no-prior seam-SAM masks passing unchanged neutral checks plus single-base controls; reject whole on budget. No reference-based selection.")
 (helpers/'run_instance_comparison.py').write_text(runner,encoding='utf-8');print(out)
if __name__=='__main__':main()
