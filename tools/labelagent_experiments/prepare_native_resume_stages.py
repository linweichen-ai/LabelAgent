"""Prepare isolated stage runners, preserving every previous executed source."""
from pathlib import Path
import sys,shutil
from native_resume_common import *
helpers=Path(__file__).parent

def generation():
 src=(helpers/'generate_visible_native.py').read_text(encoding='utf-8')
 src=src.replace('from plan_visible_bases import','from native_resume_common import').replace("out/'planning_native/","out/'planning/").replace("dest=out/'seam_generation_native'","dest=out/'seam_generation'")
 (helpers/'generate_native_resume.py').write_text(src,encoding='utf-8')

def review():
 dest=OUT/'seam_generation_all';assert not dest.exists();dest.mkdir()
 new=read(OUT/'seam_generation/all_hypotheses.json');old=read(OLD/'seam_generation_all/old_and_new_hypotheses.json')
 for filename,value in [('all_hypotheses.json',new),('old_and_new_hypotheses.json',old+new),('progress.json',read(OUT/'seam_generation/progress.json')),('completed.json',read(OUT/'seam_generation/completed.json'))]:save(dest/filename,value)
 shutil.copytree(OLD/'query_generation',OUT/'query_generation')
 for name in ['structure_cache','neutral_cache']:shutil.copytree(OLD/'instance_generation'/name,OUT/'instance_generation'/name)
 src=(helpers/'review_visible_base.py').read_text(encoding='utf-8').replace('labelagent_visible_base_20261107','labelagent_native_resume_20261107')
 (helpers/'review_native_resume.py').write_text(src,encoding='utf-8')

def comparison():
 src=(helpers/'prepare_instance_comparison.py').read_text(encoding='utf-8')
 src=src.replace('labelagent_instance_separation_20261105','labelagent_native_resume_20261107').replace('labelagent_instance_separation_comparison_20261105','labelagent_native_resume_comparison_20261107').replace("new=parent/'instance_generation_consistency'","new=parent/'instance_generation'").replace('review_instance_consistency.py','review_native_resume.py').replace('run_instance_comparison.py','run_native_resume_comparison.py')
 src=src.replace("'new_fixed_model_grid_forwards':16","'new_fixed_model_grid_forwards':0").replace("'diagnostic_only_mask2former_forwards':16","'diagnostic_only_mask2former_forwards':0")
 exec(compile(src,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
 comp=Path(read(WORK/'config_comparison.json')['output'])
 if (OUT/'transport_retry_amendment.json').exists():
  runner=helpers/'run_native_resume_comparison.py';runner.write_text(runner.read_text(encoding='utf-8').replace('review_native_resume.py','review_native_resume_retry.py'),encoding='utf-8')
 if sum(read(OUT/'instance_generation/completed.json')['new_instances'].values())==0:
  shutil.copytree(INC/'C_hybrid',comp/'C_hybrid');shutil.copy2(INC/'agent_runs.json',comp/'agent_runs.json')
  save(comp/'unchanged_seed_reuse.json',{'all6_prior_real_runs_reused':True,'fresh_Agent_runs':0,'source':str(INC),'reason':'no new complete candidate admitted; all seven-class initial masks must match before reuse'})
  b=read(comp/'cache_source_binding.json');b.update(C_new_integrated_runs=0,C_original_runs_reused=2);save(comp/'cache_source_binding.json',b)

def summarize():
 src=(helpers/'summarize_projection_results.py').read_text(encoding='utf-8').replace('labelagent_projection_recovery_20261105','labelagent_native_resume_20261107').replace('labelagent_projection_retained_comparison_20261105','labelagent_native_resume_comparison_20261107').replace('instance_generation_blue/selection.json','instance_generation/selection.json')
 src=src.replace("read(out/'instance_generation/completed.json')['sam_actual_calls']+read(out/'instance_generation_blue/completed.json')['sam_actual_calls']","read(out/'seam_generation/completed.json')['real_scene_SAM_requests']")
 exec(compile(src,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
 src=(helpers/'audit_target_purity.py').read_text(encoding='utf-8').replace('labelagent_target_identity_20261106','labelagent_native_resume_20261107').replace('labelagent_target_identity_comparison_20261106','labelagent_native_resume_comparison_20261107').replace("out/'seam_generation/old_and_new_hypotheses.json'","out/'seam_generation_all/old_and_new_hypotheses.json'")
 exec(compile(src,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

if __name__=='__main__':
 {'generation':generation,'review':review,'comparison':comparison,'summarize':summarize}[sys.argv[1]]()
