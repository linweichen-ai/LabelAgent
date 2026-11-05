"""Freeze full-pipeline candidate comparison after reference-free generation."""
import json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent;work=ROOT/'task_workspaces/labelagent_semantic_recovery_20261105';source=ROOT/'outputs/labelagent_context_comparison_20261105';out=ROOT/'outputs/labelagent_semantic_comparison_20261105';generation=ROOT/'outputs/labelagent_semantic_recovery_20261105/instance_generation_contextual'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(source/'completed.json')['status']=='complete' and read(generation/'completed.json')['status']=='complete'
if out.exists():raise FileExistsError('Keep all comparisons')
out.mkdir();cfg=read(work/'config.json');cfg.update(instance_generation=str(generation));save(work/'config_semantic.json',cfg)
for name in ['mask2former','sam_text','refinement','inputs','gap_action_cache','A_current','B_sam7']:shutil.copytree(source/name,out/name)
save(out/'agent_runs.json',[r for r in read(source/'agent_runs.json') if r['arm'] in ['A_current','B_sam7']]);save(out/'model_trace.json',read(source/'model_trace.json'))
save(out/'cache_source_binding.json',{'source':str(source),'source_plan_sha256':hashlib.sha256((source/'frozen_plan.json').read_bytes()).hexdigest(),'A_B_runs_identical_and_reused':True,'C_new_actual_agent_runs':2,'C_floor_policy_identical':True,'candidate_generation_without_reference_labels':True})
save(out/'protocol_amendment.json',{'all_acceptance_limits_unchanged':True,'rounds':3,'ops':3,'points':32,'whole_model_candidates':'32 original and84 new reference-free grid forwards; fixed original postprocessing; natural context neutral evidence gate; same full-mask budget and person protections','no_budget_pixel_clipping':True,'previous_incumbent_and_failed_variants_retained':True})
runner=(helpers/'run_context_comparison.py').read_text(encoding='utf-8')
runner=runner.replace("'C':'unchanged previous13-support foreground; original SAM floor bootstrap; first-round two individually counted context-gap edits and one true SAM; same residual controller'","'C':'new original trained masks from32+84 fixed-grid forwards; unmarked two-view neutral image evidence; original SAM floor bootstrap and context-first three-round controller'")
runner=runner.replace("'new_fixed_model_grid_forwards':32","'new_fixed_model_grid_forwards':84").replace("'real_mask2former_forwards_source':42","'real_mask2former_forwards_source':126")
runner=runner.replace("sha(Path(__file__).with_name('select_preserved_model_candidates.py'))","sha(Path(__file__).with_name('generate_contextual_candidates.py'))")
runner=runner.replace("'residual_policy_sha256':sha(Path(__file__).with_name('residual_gap_policy.py')),","'residual_policy_sha256':sha(Path(__file__).with_name('residual_gap_policy.py')),\n            'context_first_policy_sha256':sha(Path(__file__).with_name('spatial_gap_recovery.py')),\n            'neutral_evidence_policy_sha256':sha(Path(__file__).with_name('semantic_evidence_gate.py')),")
(helpers/'run_semantic_comparison.py').write_text(runner,encoding='utf-8');print(out)
