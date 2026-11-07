"""Exploratory correction of floor/container-point binding, strict final gates."""
import hashlib,json,os,re,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
import run_material_anchor as base
from counter_material_policy import propose
from neutral_caption_cache import lookup
from review_material_cached import review_one
from run_visual_budget import payload
from semantic_subject_guard import compatible_pallet_subject

ROOT=base.ROOT;SHORT=base.SHORT;OUT=Path(os.environ.get('LABELAGENT_COUNTER_OUTPUT',str(ROOT/'outputs/labelagent_counter_material_20261108')))


def prepare():
    if OUT.exists():raise FileExistsError('Preserve counter-prompt trial')
    if shutil.disk_usage(OUT.parent).free<256*2**20:raise RuntimeError('Output space guard')
    plan=base.old.read(SHORT/'frozen_plan.json');semantic=base.old.read(SHORT/'semantic/completed.json');protected={str(SHORT/'frozen_plan.json'):base.old.digest(SHORT/'frozen_plan.json'),str(SHORT/'semantic/completed.json'):base.old.digest(SHORT/'semantic/completed.json'),str(Path(__file__)):base.old.digest(__file__),str(Path(base.__file__)):base.old.digest(base.__file__),str(Path(__file__).with_name('counter_material_policy.py')):base.old.digest(Path(__file__).with_name('counter_material_policy.py'))};rows=[];diagnostics=[];OUT.mkdir()
    for seed in plan['seeds']:
        evidence=next((r for r in semantic['rows'] if r.get('seed')==seed['index'] and r.get('frame')==76),None)
        if evidence is None or evidence['PAL_lexical_subject_guard'] or not re.search(r'\b(parking|paved|floor|container)\b',evidence['decision']['object'],re.I):continue
        camera=int(seed['case'].split('_')[0][6:]);source=next(s for s in plan['native_sources'] if s['camera']==camera and s['frame']==76);protected[source['path']]=source['sha256'];protected[evidence['response']]=evidence['response_sha256'];photo=Image.open(source['path']).convert('RGB');box=seed['binding']['source_box'];im=photo.crop(tuple(box));anchor=[seed['initial_points'][0][0]-box[0],seed['initial_points'][0][1]-box[1]];proposals,eligible,diag=propose(np.asarray(im),anchor);dest=OUT/f"seed{seed['index']:02d}";dest.mkdir();im.save(dest/'SAM_input.png');Image.fromarray(eligible*255).save(dest/'RGB_texture_cue_not_mask.png');diagnostics.append({'seed':seed['index'],'old_neutral_caption':evidence['decision']['object'],**diag})
        for p in proposals:
            q=p['points'][0];ax,ay=int(q['x']+box[0]),int(q['y']+box[1]);canvasbox=[ax-64,ay-64,ax+65,ay+65];clip=[max(0,canvasbox[0]),max(0,canvasbox[1]),min(photo.width,canvasbox[2]),min(photo.height,canvasbox[3])];canvas=Image.new('RGB',(129,129),(127,127,127));canvas.paste(photo.crop(tuple(clip)),(clip[0]-canvasbox[0],clip[1]-canvasbox[1]));file=dest/(p['cue']+'_neutral.png');canvas.resize((512,512),Image.Resampling.LANCZOS).save(file);key=hashlib.sha256((base.old.digest(dest/'SAM_input.png')+json.dumps(p['points'],sort_keys=True)).encode()).hexdigest();request=hashlib.sha256(json.dumps(payload(file.read_bytes(),256),sort_keys=True).encode()).hexdigest();rows.append({**p,'key':key,'seed':seed['index'],'case':seed['case'],'box':box,'input':str(dest/'SAM_input.png'),'input_sha256':base.old.digest(dest/'SAM_input.png'),'source_image':source['path'],'source_point':[ax,ay],'semantic_image':str(file),'semantic_image_sha256':base.old.digest(file),'semantic_canvas_box':canvasbox,'semantic_clip_box':clip,'request_sha256':request})
    model=base.old.read(SHORT/'semantic/frozen_plan.json')['model_config_sha256'];base.old.save(OUT/'frozen_plan.json',{'frozen_at':__import__('datetime').datetime.now().astimezone().isoformat(),'proposals':rows,'diagnostics':diagnostics,'protected':protected,'model_config_sha256':model,'selection_from_old_neutral_floor_or_container_caption_not_reference':True,'negative_label_is_candidate_hypothesis_not_new_truth':True,'max_two_hypotheses_per_selected_seed':True,'one_SAM_call_per_qualified_cue':True,'Agent_round_limit':3,'new_Agent_rounds':0,'no_reference_labels_points_depth_pose_calibration_or_area_inputs':True,'no_post_mask_clipping':True,'all_original_acceptance_gates_unchanged':True});print({'selected_seeds':len(diagnostics),'point_hypotheses':len(rows),'max_SAM_calls':len(rows)},flush=True)


def review():
    if (OUT/'semantic_completed.json').exists():raise FileExistsError('Preserve cache-first review')
    p=base.old.read(OUT/'frozen_plan.json');assert all(base.old.digest(f)==h for f,h in p['protected'].items());catalogue=[]
    for file in [SHORT/'semantic/completed.json',ROOT/'outputs/labelagent_native_multi_20261107/semantic/completed.json',ROOT/'outputs/labelagent_soft_prior_20261108/semantic_controls/completed.json',base.OUT/'semantic_completed.json']:catalogue+=base.old.read(file)['rows']
    rows=[]
    for r in p['proposals']:
        row=review_one(r,p['model_config_sha256'],catalogue,OUT/'semantic_responses',allow_network=True);row['PAL_guard']=compatible_pallet_subject(row['decision']);rows.append(row);catalogue.append(row);base.old.save(OUT/'semantic_progress.json',rows);print({'seed':r['seed'],'cue':r['cue'],'caption':row['decision'],'PAL_guard':row['PAL_guard'],'new_call':row['new_model_call']},flush=True)
    base.old.save(OUT/'semantic_completed.json',{'rows':rows,'logical_views':len(rows),'unique_real_VLM_calls':sum(r['new_model_call'] for r in rows),'exact_cache_reused_views':sum(not r['new_model_call'] for r in rows),'eligible_for_SAM':sum(r['PAL_guard'] for r in rows),'negative_hypothesis_not_approved_truth':True})


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='review':review()
    else:base.OUT=OUT;base.run()
