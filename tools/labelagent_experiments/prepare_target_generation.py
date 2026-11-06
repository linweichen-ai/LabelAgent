"""Reuse an API transport, preserve old source parents and generate new prompts."""
from pathlib import Path
import shutil,json
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent
parent=ROOT/'outputs/labelagent_target_identity_20261106/parents';parent.mkdir(exist_ok=True)
shutil.copy2(ROOT/'outputs/labelagent_instance_separation_20261105/parents/completed.json',parent/'completed.json')
failed=ROOT/'outputs/labelagent_target_identity_20261106/startup_path_failure.json'
if not failed.exists():failed.write_text(json.dumps({'status':'fixed_before_any_model_request','error':'missing new parent hash binding file','old_parent_files_unchanged':True,'model_requests_before_failure':0},indent=2)+'\n',encoding='utf-8')
src=(helpers/'generate_seam_instances.py').read_text(encoding='utf-8').replace('labelagent_instance_separation_20261105','labelagent_target_identity_20261106')
src=src.replace('from instance_separation_geometry import visible_seam_prompts','from target_identity_policy import deck_foot_prompts as visible_seam_prompts')
src=src.replace("parents=read(out/'parents/completed.json')", "parents=read(ROOT/'outputs/labelagent_instance_separation_20261105/parents/completed.json')")
src=src.replace("save(Path(parent['folder'])/'seams.json'", "save(out/'parent_plans'/parent['case']/(parent['key']+'.json')")
src=src.replace("'no_reference_inputs':True})\n with", "'no_reference_inputs':True,'deck_foot_policy_sha256':hashlib.sha256(Path(__file__).with_name('target_identity_policy.py').read_bytes()).hexdigest()})\n with")
(helpers/'generate_target_identity.py').write_text(src,encoding='utf-8')
