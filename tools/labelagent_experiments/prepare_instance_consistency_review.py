"""Keep failed v1 untouched; v2 reuses identical bound model responses."""
import shutil,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent;out=ROOT/'outputs/labelagent_instance_separation_20261105'
old=out/'instance_generation';new=out/'instance_generation_consistency'
assert json.loads((old/'completed.json').read_text(encoding='utf-8'))['status']=='complete'
if new.exists():raise FileExistsError('Keep consistency variants immutable')
new.mkdir()
for folder in ['structure_cache','neutral_cache']:
 if (old/folder).exists():shutil.copytree(old/folder,new/folder)
source=(helpers/'review_instance_separation.py').read_text(encoding='utf-8').replace("dest=out/'instance_generation'","dest=out/'instance_generation_consistency'")
source=source.replace('from instance_evidence_views import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA,png,single_physical_base','from instance_identity_consistency import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA,png,single_physical_base,repeated_control_rejected')
source=source.replace("all(r['decision'].get('category')=='pallet_base' and r['decision'].get('instance_status')=='repeated' for r in rows)","all(repeated_control_rejected(r['decision']) for r in rows)")
(helpers/'review_instance_consistency.py').write_text(source,encoding='utf-8')
(new/'pre_review_plan.json').write_text(json.dumps({'failed_v1_preserved':str(old),'same_model_prompt_schema_and_source_images':True,'no_extra_model_calls_required_by_reparse':True,'change':'veto stack/multiple evidence despite single status; normalize object enum separator; positive control required','old_neutral_gate_unchanged':True,'no_reference_inputs':True,'consistency_policy_sha256':hashlib.sha256((helpers/'instance_identity_consistency.py').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
