"""Freeze a new input policy; original neutral/subject and single-base gates stay."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent;out=ROOT/'outputs/labelagent_target_identity_20261106';old=ROOT/'outputs/labelagent_instance_separation_20261105'
src=(helpers/'review_instance_consistency.py').read_text(encoding='utf-8').replace('labelagent_instance_separation_20261105','labelagent_target_identity_20261106').replace("dest=out/'instance_generation_consistency'","dest=out/'instance_generation'")
src=src.replace('from semantic_subject_guard import compatible_pallet_subject','from semantic_subject_guard import compatible_pallet_subject\n from target_identity_policy import centered_photos,extent_views,EXTENT_PROMPT,EXTENT_SCHEMA')
start=src.index("m=_rasterize(row['shape'],1920,1536);yy,xx=np.where(m)");end=src.index('    neutral_ok=',start)
src=src[:start]+"m=_rasterize(row['shape'],1920,1536);views,center_binding=centered_photos(image,m);neutral=[]\n    for name,photo in views.items():neutral.append(classify_neutral(workflow,config,png(photo),dest/'neutral_cache'))\n"+src[end:]
src=src.replace("single_ok=bool(structure_rows) and all(single_physical_base(r['decision']) for r in structure_rows);passed=neutral_ok and single_ok and controls_passed", "single_ok=bool(structure_rows) and all(single_physical_base(r['decision']) for r in structure_rows);extent=None;extent_ok=False\n    if neutral_ok and single_ok:\n     pointed,binary,_=extent_views(image,m);extent=workflow.chat(config,EXTENT_PROMPT,png(pointed),extra_images=[png(binary)],response_schema=EXTENT_SCHEMA);extent_ok=extent.get('target_extent') in ['whole_visible_object','identifiable_occluded_object'] and compatible_pallet_subject({'object':extent.get('object','')})\n    passed=neutral_ok and single_ok and controls_passed and extent_ok")
src=src.replace("'passed':passed}","'passed':passed,'center_binding':center_binding,'extent':extent,'extent_passed':extent_ok}")
src=src.replace("'old_neutral_subject_gate_unchanged':True", "'old_neutral_subject_gate_unchanged':True,'neutral_input_geometry':'foreground-anchor centered complete candidate, two unmarked contexts','complete_visible_object_extent_required':True")
# Reuse only exact identity-bound old structure calls; no hidden label-based edits.
dest=out/'instance_generation';assert not dest.exists();dest.mkdir();shutil.copytree(old/'instance_generation_consistency/structure_cache',dest/'structure_cache');shutil.copytree(out/'target_12of16_diagnostic/neutral_cache',dest/'neutral_cache')
q=out/'query_generation';q.mkdir();shutil.copy2(old/'query_generation/all_hypotheses.json',q/'all_hypotheses.json');shutil.copy2(old/'query_generation/completed.json',q/'completed.json')
# Include all seven old SAM masks as untouched image-only candidates.
new=json.loads((out/'seam_generation/all_hypotheses.json').read_text(encoding='utf-8'));prev=json.loads((old/'seam_generation/all_hypotheses.json').read_text(encoding='utf-8'));(out/'seam_generation/old_and_new_hypotheses.json').write_text(json.dumps(prev+new,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');src=src.replace("read(out/'seam_generation/all_hypotheses.json')","read(out/'seam_generation/old_and_new_hypotheses.json')")
(helpers/'review_target_identity.py').write_text(src,encoding='utf-8')
