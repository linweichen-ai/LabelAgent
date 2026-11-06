"""Apply every original check to this immutable comparison."""
from pathlib import Path
p=Path(__file__).with_name('summarize_projection_results.py');s=p.read_text(encoding='utf-8').replace('labelagent_projection_recovery_20261105','labelagent_target_identity_20261106').replace('labelagent_projection_retained_comparison_20261105','labelagent_target_identity_comparison_20261106').replace('instance_generation_blue/selection.json','instance_generation/selection.json')
s=s.replace("read(out/'instance_generation/completed.json')['sam_actual_calls']+read(out/'instance_generation_blue/completed.json')['sam_actual_calls']","read(out/'seam_generation/completed.json')['real_scene_SAM_requests']")
exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
