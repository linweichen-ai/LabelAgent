"""Isolated complete-mask generation, original review, and post-freeze scoring."""
import sys,shutil
from pathlib import Path
from point_ids_common import *
helpers=Path(__file__).parent

def generation():
 s=(helpers/'generate_native_resume.py').read_text(encoding='utf-8').replace('from native_resume_common import','from point_ids_common import')
 s=s.replace('for _ in range(100):\n   with urllib.request.urlopen','for _ in range(1500):\n   with urllib.request.urlopen',1).replace('time.sleep(3)','time.sleep(.2)')
 s=s.replace(" save(dest/'all_hypotheses.json',hyps);", " save(dest/'progress.json',rows);save(dest/'all_hypotheses.json',hyps);")
 (helpers/'generate_point_ids.py').write_text(s,encoding='utf-8')

def review():
 dest=OUT/'seam_generation_all'
 if dest.exists():assert not any(dest.iterdir()),'Partial aggregation is not empty; preserve and inspect'
 else:dest.mkdir()
 new=read(OUT/'seam_generation/all_hypotheses.json');old=read(PREVIOUS/'seam_generation_all/old_and_new_hypotheses.json')
 for file,value in [('all_hypotheses.json',new),('old_and_new_hypotheses.json',old+new),('progress.json',read(OUT/'seam_generation/progress.json')),('completed.json',read(OUT/'seam_generation/completed.json'))]:save(dest/file,value)
 shutil.copytree(PREVIOUS/'query_generation',OUT/'query_generation')
 for name in ['structure_cache','neutral_cache']:shutil.copytree(PREVIOUS/'instance_generation'/name,OUT/'instance_generation'/name)
 s=(helpers/'review_native_resume_retry.py').read_text(encoding='utf-8').replace('labelagent_native_resume_20261107','labelagent_point_ids_20261107')
 s=s.replace("controls_passed=all(r['passed'] for r in controls)","controls_passed=all(r['passed'] for r in controls);planner_guard=read(out/'planner_controls/completed.json')['PAL_guard_passed']")
 s=s.replace('passed=neutral_ok and single_ok and controls_passed and extent_ok',"new_ID_candidate=str(out/'seam_generation') in row['source_prediction'];passed=neutral_ok and single_ok and controls_passed and extent_ok and (planner_guard or not new_ID_candidate)")
 s=s.replace("'extent_passed':extent_ok}","'extent_passed':extent_ok,'new_ID_candidate':new_ID_candidate,'point_ID_blank_PAL_guard_passed':planner_guard}")
 s=s.replace("'annotation_writes':0}","'annotation_writes':0,'point_ID_blank_PAL_guard_passed':planner_guard}")
 (helpers/'review_point_ids.py').write_text(s,encoding='utf-8')

def comparison():
 s=(helpers/'prepare_instance_comparison.py').read_text(encoding='utf-8').replace('labelagent_instance_separation_20261105','labelagent_point_ids_20261107').replace('labelagent_instance_separation_comparison_20261105','labelagent_point_ids_comparison_20261107').replace("new=parent/'instance_generation_consistency'","new=parent/'instance_generation'").replace('review_instance_consistency.py','review_point_ids.py').replace('run_instance_comparison.py','run_point_ids_comparison.py').replace("'new_fixed_model_grid_forwards':16","'new_fixed_model_grid_forwards':0").replace("'diagnostic_only_mask2former_forwards':16","'diagnostic_only_mask2former_forwards':0")
 exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
 comp=Path(read(WORK/'config_comparison.json')['output'])
 if sum(read(OUT/'instance_generation/completed.json')['new_instances'].values())==0:
  shutil.copytree(INC/'C_hybrid',comp/'C_hybrid');shutil.copy2(INC/'agent_runs.json',comp/'agent_runs.json');save(comp/'unchanged_seed_reuse.json',{'all6_prior_real_runs_reused':True,'fresh_Agent_runs':0,'source':str(INC),'reason':'no new complete candidate admitted; all seven-class initial masks must match before reuse'})
  b=read(comp/'cache_source_binding.json');b.update(C_new_integrated_runs=0,C_original_runs_reused=2);save(comp/'cache_source_binding.json',b)

def summarize():
 s=(helpers/'summarize_projection_results.py').read_text(encoding='utf-8').replace('labelagent_projection_recovery_20261105','labelagent_point_ids_20261107').replace('labelagent_projection_retained_comparison_20261105','labelagent_point_ids_comparison_20261107').replace('instance_generation_blue/selection.json','instance_generation/selection.json')
 s=s.replace("read(out/'instance_generation/completed.json')['sam_actual_calls']+read(out/'instance_generation_blue/completed.json')['sam_actual_calls']","read(out/'seam_generation/completed.json')['real_scene_SAM_requests']")
 exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
 s=(helpers/'audit_target_purity.py').read_text(encoding='utf-8').replace('labelagent_target_identity_20261106','labelagent_point_ids_20261107').replace('labelagent_target_identity_comparison_20261106','labelagent_point_ids_comparison_20261107').replace("out/'seam_generation/old_and_new_hypotheses.json'","out/'seam_generation_all/old_and_new_hypotheses.json'")
 s=s.replace("('本轮新SAM最佳者（未通过最终验收）',[row['shape']])","('本轮完整新输出（无新输出则只显示原图）',[row['shape']] if new else [])").replace('m=np.maximum.reduce(masks);rgb=','m=np.maximum.reduce(masks) if masks else np.zeros((1536,1920),np.uint8);rgb=').replace('图示新候选：覆盖','新候选诊断（无新输出时数值仅为历史参考）：覆盖')
 exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

if __name__=='__main__':{'generation':generation,'review':review,'comparison':comparison,'summarize':summarize}[sys.argv[1]]()
