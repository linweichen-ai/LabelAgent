"""Original acceptance and all incumbent rasters remain mandatory."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=Path(__file__).with_name('summarize_projection_results.py').read_text(encoding='utf-8')
base=base.replace('labelagent_projection_recovery_20261105','labelagent_instance_separation_20261105').replace('labelagent_projection_retained_comparison_20261105','labelagent_instance_separation_comparison_20261105').replace('instance_generation_blue/selection.json','instance_generation_consistency/selection.json')
base=base.replace("read(out/'instance_generation/completed.json')['sam_actual_calls']+read(out/'instance_generation_blue/completed.json')['sam_actual_calls']","read(out/'seam_generation/completed.json')['real_scene_SAM_requests']")
base=base.replace("'incumbent_checks':incumbent_checks,","'incumbent_checks':incumbent_checks,'functional_single_instance_controls_passed':read(out/'instance_generation_consistency/completed.json')['controls_passed'],")
# Selected masks are absent in this run; retain a proof branch for a future
# immutable run rather than silently reconstructing an original prediction.
base=base.replace("if row['source']=='unchanged_raw_trained_grid':", "if row['source']=='unchanged_raw_trained_grid':")
exec(compile(base,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
