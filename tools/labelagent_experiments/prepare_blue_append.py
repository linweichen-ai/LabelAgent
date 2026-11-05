"""Separate color-guided scan; no reference-derived location is supplied."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent;parent=ROOT/'outputs/labelagent_projection_recovery_20261105';source=parent/'instance_generation';out=parent/'instance_generation_blue'
assert json.loads((source/'completed.json').read_text(encoding='utf-8'))['status']=='complete'
if out.exists():raise FileExistsError('Preserve every generation variant')
out.mkdir();shutil.copytree(source/'visual_cache',out/'visual_cache')
plan={'scope':'additional exposed-image diagnostic variant, not a held-out optimization','source_hypothesis_bank_sha256':{p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*/all_hypotheses.json')},'semantic_cache_source':'exact image/prompt/model-bound first variant cache; reused calls not counted fresh','reason':'largest-area parent ranking can spend prompts on large cargo-like predictions while missing small blue pallet groups','group_rule':'predicted PAL component width>=24,height>=18,area>=300 and original RGB blue fraction>=.1; rank area*blue_fraction;4 per camera','blue_pixel_rule':'B-R>=12,B>=G-3,B>=45','no_reference_positions':True,'max_additional_real_SAM_requests':24,'same_fractional_clicks':[.2,.5,.8],'same_masks_and_semantic_acceptance':True,'all_incumbent_masks_kept':True,'same_original_global_area_and_count_limits':True,'no_budget_clipping':True}
(out/'protocol_amendment.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
src=(helpers/'generate_append_recovery.py').read_text(encoding='utf-8').replace("out=ROOT/'outputs/labelagent_projection_recovery_20261105/instance_generation'","out=ROOT/'outputs/labelagent_projection_recovery_20261105/instance_generation_blue'")
src=src.replace('if w<24 or h<36 or area<500:continue','''if w<24 or h<18 or area<300:continue
    rgb=np.asarray(images[case]).astype('int16')[mask!=0];blue=float(((rgb[:,2]-rgb[:,0]>=12)&(rgb[:,2]>=rgb[:,1]-3)&(rgb[:,2]>=45)).mean())
    if blue<.1:continue''')
src=src.replace("'source_index':index}","'source_index':index,'blue_fraction':blue}")
src=src.replace("key=lambda r:(-r['area'],r['key'])","key=lambda r:(-r['area']*r['blue_fraction'],r['key'])")
src=src.replace(" save(out/'sam_prompt_plan.json'"," for case in pool:\n  for row in read(out.parent/'instance_generation'/case/'all_hypotheses.json'):\n   pool[case].setdefault(row['key'],{**row,'mask':_rasterize(row['shape'],1920,1536)})\n save(out/'sam_prompt_plan.json'")
(helpers/'generate_blue_append.py').write_text(src,encoding='utf-8');print('Color-guided append variant frozen')
