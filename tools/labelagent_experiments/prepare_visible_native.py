"""Preserve v1 failures; freeze a separate image/coordinate repair before calls."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent;out=ROOT/'outputs/labelagent_visible_base_20261107'
assert json.loads((out/'planning/completed.json').read_text(encoding='utf-8'))['status']=='complete'
src=(helpers/'plan_visible_bases.py').read_text(encoding='utf-8').replace("dest=out/'planning'","dest=out/'planning_native'").replace('from visible_base_policy import planning_views,bind_points,PLAN_PROMPT,PLAN_SCHEMA,png','from visible_base_native_policy import planning_views,bind_points,PLAN_PROMPT,PLAN_SCHEMA,png').replace("with_name('visible_base_policy.py')","with_name('visible_base_native_policy.py')")
(helpers/'plan_visible_native.py').write_text(src,encoding='utf-8')
src=(helpers/'generate_visible_bases.py').read_text(encoding='utf-8').replace("dest=out/'seam_generation'","dest=out/'seam_generation_native'").replace("out/'planning/", "out/'planning_native/")
(helpers/'generate_visible_native.py').write_text(src,encoding='utf-8')
(out/'native_input_amendment.json').write_text(json.dumps({'prior_failed_plans_preserved':True,'repair':'real square context crop and explicit normalized0..1000 points; no padding reinterpretation, clamping or snapping','new_planner_requests_max':16,'new_SAM_requests_max':16,'Agent_round_limit_unchanged':3,'all_original_acceptance_unchanged':True,'no_reference_points':True,'policy_sha256':hashlib.sha256((helpers/'visible_base_native_policy.py').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
