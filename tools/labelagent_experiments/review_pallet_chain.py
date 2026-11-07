"""Neutral source/neighbor anchor review for every newly available RGB pair."""
import hashlib,json,sys,time,urllib.request,urllib.error
from pathlib import Path
from PIL import Image
from run_pallet_chain import ROOT,OUT,old
from run_visual_budget import payload
from semantic_subject_guard import compatible_pallet_subject

D=OUT/'semantic'


def prepare():
    if D.exists():raise FileExistsError('Preserve frozen anchor review')
    D.mkdir();plan=old.read(OUT/'frozen_plan.json');done=old.read(OUT/'completed.json');model=old.read(ROOT/'outputs/labelagent_native_multi_20261107/semantic/frozen_plan.json')['model_config_sha256'];cache={}
    for study in ['labelagent_native_multi_20261107/semantic','labelagent_soft_prior_20261108/semantic_controls']:
        for r in old.read(ROOT/'outputs'/study/'completed.json')['rows']:cache[r['request_sha256']]=r
    requests=[];protected={str(OUT/'completed.json'):old.digest(OUT/'completed.json'),str(OUT/'frozen_plan.json'):old.digest(OUT/'frozen_plan.json')};selected=[]
    for r in done['rows']:
        if r['direct']['diagnostic_image_correspondence_available'] or not r['chain']['diagnostic_image_correspondence_available'] or not r['chain']['anchor_valid']:continue
        seed=next(s for s in plan['seeds'] if s['index']==r['seed']);camera=int(r['case'].split('_')[0][6:]);selected.append({'seed':r['seed'],'frame':r['endpoint_frame']})
        for frame,anchor in [(76,seed['initial_points'][0]),(r['endpoint_frame'],r['chain']['points'][0]['endpoint_xy'])]:
            source=next(s for s in plan['native_sources'] if s['camera']==camera and s['frame']==frame);photo=Image.open(source['path']).convert('RGB');ax,ay=[int(round(v)) for v in anchor];box=[ax-64,ay-64,ax+65,ay+65];clip=[max(0,box[0]),max(0,box[1]),min(photo.width,box[2]),min(photo.height,box[3])];im=Image.new('RGB',(129,129),(127,127,127));im.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));im=im.resize((512,512),Image.Resampling.LANCZOS);file=D/f"seed{r['seed']:02d}_frame{frame}_unmarked_anchor.png";im.save(file);protected[source['path']]=source['sha256'];requests.append({'name':file.stem,'kind':'real_anchor_review','seed':r['seed'],'camera':camera,'frame':frame,'image':str(file),'source_image':source['path'],'anchor_source_xy':anchor,'rounded_anchor':[ax,ay],'canvas_box':box,'clipped_box':clip,'native_canvas_side':129,'no_locator_or_expected_category_in_image':True,'source_pixels_only_not_mask_or_identity_proof':True})
    for color,value in [('gray',127),('white',255)]:
        file=D/(color+'_control.png');Image.new('RGB',(512,512),(value,)*3).save(file);requests.append({'name':color+'_control','kind':color+'_control','image':str(file)})
    for r in requests:
        r['image_sha256']=old.digest(r['image']);body=payload(Path(r['image']).read_bytes(),256);key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();r['request_sha256']=key
        if key in cache:
            c=cache[key];reply=old.read(c['response']);assert reply['local_vlm']['model_config_sha256']==model;r.update(reuse_response=c['response'],reuse_response_sha256=old.digest(c['response']));protected[c['response']]=old.digest(c['response'])
    old.save(D/'frozen_plan.json',{'driver_sha256':old.digest(__file__),'requests':requests,'selected_pairs':selected,'protected':protected,'model_config_sha256':model,'selection_only_new_chain_correspondence_plus_valid_original_anchor':True,'same_neutral_caption_schema_and_256_image_tokens':True,'no_reference_inputs':True,'max_fresh_requests':len({r['request_sha256'] for r in requests if 'reuse_response' not in r}),'no_mask_generated_or_adopted':True,'native_resizing_does_not_add_detail':True,'Agent_round_limit':3,'new_Agent_runs':0});print({'selected_new_pairs':len(selected),'logical_requests':len(requests),'max_fresh':len({r['request_sha256'] for r in requests if 'reuse_response' not in r})},flush=True)


def run():
    if (D/'completed.json').exists():raise FileExistsError('Do not repeat completed review')
    p=old.read(D/'frozen_plan.json');assert old.digest(__file__)==p['driver_sha256'];rows=[]
    for r in p['requests']:
        assert old.digest(r['image'])==r['image_sha256'];body=payload(Path(r['image']).read_bytes(),256);key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();assert key==r['request_sha256'];file=Path(r['reuse_response']) if 'reuse_response' in r else D/'responses'/(key+'.json')
        if not file.exists():
            for _ in range(1500):
                if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:reply=json.load(f)
                    old.save(file,reply);break
                except urllib.error.HTTPError as e:
                    if e.code==503:time.sleep(.2);continue
                    raise
            else:raise RuntimeError('Shared VLM busy; preserve all completed replies')
        reply=old.read(file);assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256];caption=json.loads(reply['choices'][0]['message']['content']);row={**r,'decision':caption,'PAL_lexical_subject_guard':compatible_pallet_subject(caption),'reused':'reuse_response' in r,'response':str(file),'response_sha256':old.digest(file),'not_physical_identity_or_mask_acceptance':True};rows.append(row);old.save(D/'progress.json',rows);print(json.dumps({'name':r['name'],'caption':caption,'PAL_guard':row['PAL_lexical_subject_guard'],'reused':row['reused']},ensure_ascii=False),flush=True)
    assert all(old.digest(f)==h for f,h in p['protected'].items());old.save(D/'completed.json',{'rows':rows,'logical_views':len(rows),'fresh_unique_VLM_calls':len({r['request_sha256'] for r in rows if not r['reused']}),'exact_reused_views':sum(r['reused'] for r in rows),'blank_PAL_false_positives':sum(r['PAL_lexical_subject_guard'] for r in rows if r['kind']!='real_anchor_review'),'lexical_subject_guard_not_instance_confirmation':True,'adopted':False,'new_masks':0,'new_Agent_runs':0})


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
