"""Isolate floor/controller changes while holding the 13/17 foreground fixed."""
import json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_semantic_recovery_20261105';source=ROOT/'outputs/labelagent_preserved_model_comparison_20261105';out=ROOT/'outputs/labelagent_context_comparison_20261105';helpers=ROOT/'tools/labelagent_live_handoff'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if out.exists():raise FileExistsError('Preserve runs')
out.mkdir();cfg=read(work/'config.json');cfg.update(output=str(out),instance_generation=read(ROOT/'task_workspaces/labelagent_recovery_20261105/config_preserved.json')['instance_generation']);save(work/'config_context.json',cfg)
for name in ['mask2former','sam_text','refinement','inputs','gap_action_cache','A_current']:shutil.copytree(source/name,out/name)
save(out/'agent_runs.json',[r for r in read(source/'agent_runs.json') if r['arm']=='A_current'])
save(out/'model_trace.json',read(source/'model_trace.json'))
save(out/'cache_source_binding.json',{'source':str(source),'source_plan_sha256':hashlib.sha256((source/'frozen_plan.json').read_bytes()).hexdigest(),'A_only_run_cache_reused':True,'B_C_new_actual_agent_runs':4,'same_checkpoint_and_images':True,'new_SAM_request_inputs_not_reused_from_preflight':True,'per_gap_image_bound_cached_semantics_reused':True})
save(out/'protocol_amendment.json',{'all_acceptance_limits_unchanged':True,'rounds':3,'ops':3,'points':32,'C_foreground_held_identical_to_previous13_support':True,
 'C_floor_seed':'use same fixed original SAM floor seed as B, rather than technically valid but more fragmented trained floor refinement; trained alternatives retained in seed_stages',
 'controller':'first round two individually counted most-interior approved gaps, one genuine unchanged full-frame SAM for remaining; second round original residual scheduler; third actual observation required',
 'crop_preflight':'failed, not adopted','not_independent_validation':True})
review=(helpers/'ground_action_review_v5.py').read_text(encoding='utf-8')
review=review.replace('else:selected=[batch]','else:\n                    from spatial_gap_recovery import choose_context_first\n                    selected=choose_context_first(holes,assessments,operations,geometry,width,height,approved_gap_sam_operation)')
(helpers/'ground_action_review_v6.py').write_text(review,encoding='utf-8')
runner=(helpers/'run_preserved_model_comparison.py').read_text(encoding='utf-8').replace('ground_action_review_v5','ground_action_review_v6')
runner=runner.replace("present={s['label'] for s in raw};complete=copy.deepcopy(refined)","present={s['label'] for s in raw};complete=[copy.deepcopy(s) for s in refined if s['label']!='floor']\n            complete.extend(copy.deepcopy(text_seeds[topic]['floor']))")
runner=runner.replace("'C':'preserved raw trained fixed-grid instances; failed SAM/review safety prevents boundary overwrite; per-camera whole-mask reserves and predicted-person guard; bounded residual gap recovery'","'C':'unchanged previous13-support foreground; original SAM floor bootstrap; first-round two individually counted context-gap edits and one true SAM; same residual controller'")
(helpers/'run_context_comparison.py').write_text(runner,encoding='utf-8')
for name in ['ground_action_review_v6.py','spatial_gap_recovery.py','residual_gap_policy.py','preserved_model_policy.py']:
 shutil.copy2(helpers/name,work/'source_snapshot/tools'/name)
print(out)
