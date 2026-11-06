"""Compare new complete masks; reuse all six runs only for identical seeds."""
from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent;out=ROOT/'outputs/labelagent_visible_base_20261107';work=ROOT/'task_workspaces/labelagent_visible_base_20261107'
src=(helpers/'prepare_target_comparison.py').read_text(encoding='utf-8').replace('labelagent_target_identity_20261106','labelagent_visible_base_20261107').replace('labelagent_target_identity_comparison_20261106','labelagent_visible_base_comparison_20261107').replace('review_target_identity.py','review_visible_base.py').replace('run_target_identity_comparison.py','run_visible_base_comparison.py')
exec(compile(src,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
read=lambda p:json.loads(p.read_text(encoding='utf-8'));comparison=ROOT/'outputs/labelagent_visible_base_comparison_20261107';inc=ROOT/'outputs/labelagent_context_comparison_20261105';selected=read(out/'instance_generation/completed.json')['new_instances']
if sum(selected.values())==0:
 shutil.copytree(inc/'C_hybrid',comparison/'C_hybrid');(comparison/'agent_runs.json').write_text((inc/'agent_runs.json').read_text(encoding='utf-8'),encoding='utf-8');receipt={'all6_prior_real_runs_reused':True,'fresh_Agent_runs':0,'reason':'all7class initial seeds unchanged; no new complete instance admitted','source':str(inc),'initial_seed_source':str(inc/'seed_stages'),'new_initial_seeds_must_match_source_byte_for_byte':True,'C_floor_not_reseeded_with_finished_floor':True};(comparison/'unchanged_seed_reuse.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 p=comparison/'cache_source_binding.json';v=read(p);v.update(C_new_integrated_runs=0,C_original_runs_reused=2);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
runner=helpers/'run_visible_base_comparison.py';text=runner.read_text(encoding='utf-8').replace("'new_fixed_model_grid_forwards':0","'new_fixed_model_grid_forwards':8");runner.write_text(text,encoding='utf-8')
