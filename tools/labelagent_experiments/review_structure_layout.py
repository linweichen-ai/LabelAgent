"""Matched structure-view layout diagnosis at the original 256-token budget.

Only uniform letterbox padding is removed. The full original photo rectangle
and every locator pixel remain represented; no source/candidate mask changes.
Original prompts, parser and repeated-instance veto remain mandatory.
"""
import hashlib,json,math,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from run_rgb_layers import read,save,sha
from aspect_views import GRIDS
from instance_identity_consistency import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA,single_physical_base,repeated_evidence
from semantic_subject_guard import compatible_pallet_subject
import review_box_layers as runner
import run_source_gap_instances as source
OUT=source.BASE/'labelagent_structure_layout_20261108'
def layout_binding(photo,original):
    ys,xs=np.where(np.any(np.asarray(original)!=np.asarray(photo),axis=2));return None if not len(xs) else [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
def render(im,binding):
    crop=im.crop(tuple(binding['crop']));size=tuple(binding['output_size']);fitted=ImageOps.contain(crop,size,Image.Resampling.LANCZOS);canvas=Image.new('RGB',size,(127,)*3);offset=((size[0]-fitted.width)//2,(size[1]-fitted.height)//2);canvas.paste(fitted,offset);return canvas
def prepare():
    if OUT.exists():raise FileExistsError('Preserve structure layout comparison')
    sem=source.OUT/'semantic';completed=read(sem/'completed.json');p=read(sem/'frozen_plan.json');selected=[r['key'] for r in completed['candidates'] if r['neutral_passed'] and r['matched_plain_controls_passed'] and not r['single_base_passed']];assert selected;OUT.mkdir();requests=[];bindings=[];protected={str(sem/'completed.json'):sha(sem/'completed.json'),str(sem/'frozen_plan.json'):sha(sem/'frozen_plan.json'),str(Path(__file__)):sha(__file__)};generation=read(source.OUT/'SAM/generation_completed.json')['rows']
    for key in selected:
        r=next(r for r in generation if r['key']==key);sourcefile=Path(r['source_image']);maskfile=Path(r['folder'])/'complete_full_mask.npy';photo=Image.open(sourcefile).convert('RGB');mask=np.load(maskfile,allow_pickle=False);views,meta=target_views(photo,mask);box=meta['source_box'];native=photo.crop(tuple(box));scale=min(496/native.width,496/native.height);tw,th=max(1,round(native.width*scale)),max(1,round(native.height*scale));ox,oy=(512-tw)//2,(512-th)//2;base=Image.new('RGB',(512,512),(127,)*3);base.paste(native.resize((tw,th),Image.Resampling.LANCZOS),(ox,oy));protected[str(sourcefile)]=sha(sourcefile);protected[str(maskfile)]=sha(maskfile)
        for view,real in views.items():
            prior=next(x for x in completed['rows'] if x['candidate']==key and x['stage']=='structure' and x['name']==view);payload=next(x['payload'] for x in p['requests'] if x['request_sha256']==prior['request_sha256']);assert sha(prior['images'][0])==sha_png(real);marker=layout_binding(base,real);assert marker;crop=[min(ox,marker[0]),min(oy,marker[1]),max(ox+tw,marker[2]),max(oy+th,marker[3])];ratio=(crop[2]-crop[0])/(crop[3]-crop[1]);gx,gy=min(GRIDS,key=lambda g:(abs(math.log(ratio/(g[0]/g[1]))),g));binding={'candidate':key,'view':view,'original_photo_box':box,'full_original_photo_rectangle_in_square':[ox,oy,ox+tw,oy+th],'all_locator_pixels_box':marker,'crop':crop,'output_size':[gx*32,gy*32],'same_visual_token_budget':gx*gy,'source_pixels_or_markers_not_cropped':True};bindings.append(binding);folder=OUT/key/view;folder.mkdir(parents=True)
            controls={color:target_views(Image.new('RGB',photo.size,(value,)*3),mask)[0][view] for color,value in [('gray',127),('white',255)]}
            for kind,im in [('real',real),*controls.items()]:
                for layout in ['square','aspect']:
                    file=folder/(kind+'_'+layout+'.png');(im if layout=='square' else render(im,binding)).save(file);body=runner.body([str(file)],INSTANCE_PROMPT,INSTANCE_SCHEMA);digest=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();row={'candidate':key,'view':view,'kind':kind,'layout':layout,'image':str(file),'image_sha256':sha(file),'payload':body,'request_sha256':digest,'exact_previous_response':None}
                    if kind=='real' and layout=='square':assert body==payload and digest==prior['request_sha256'];row.update(exact_previous_response=prior['response'],exact_previous_response_sha256=prior['response_sha256']);protected[prior['response']]=prior['response_sha256']
                    requests.append(row);protected[str(file)]=sha(file)
    for name in ['instance_identity_consistency.py','instance_evidence_views.py','semantic_subject_guard.py','aspect_views.py','review_box_layers.py']:protected[str(Path(__file__).with_name(name))]=sha(Path(__file__).with_name(name))
    save(OUT/'frozen_plan.json',{'requests':requests,'bindings':bindings,'protected':protected,'model_config_sha256':p['model_config_sha256'],'selected_source_candidates':selected,'selection':'all source-native valid candidates with original neutral/gray-white pass and failed original single-base stage; no geometric/reference-based selection','same_original_prompt_schema_model_parser_repeated_evidence_veto':True,'image_token_budget':256,'max_fresh_requests':len({r['request_sha256'] for r in requests if not r['exact_previous_response']}),'matched_gray_white_for_both_layouts':True,'no_crop_of_photo_or_locator_pixels_only_uniform_padding_removed':True,'same_real_candidate_source_original_mask_and_locator_before_layout':True,'no_new_masks_or_GT_inputs':True,'new_Agent_runs':0,'Agent_round_limit':3,'not_a_deployed_reviewer_or_adoption':True});print({'source_candidates':len(selected),'logical_conditions':len(requests),'max_fresh_VLM':len({r['request_sha256'] for r in requests if not r['exact_previous_response']}),'original_real_structure_reused':sum(r['exact_previous_response'] is not None for r in requests),'new_calls':0},flush=True)
def sha_png(im):
    import io
    b=io.BytesIO();im.save(b,'PNG');return hashlib.sha256(b.getvalue()).hexdigest()
def run():
    if (OUT/'completed.json').exists():raise FileExistsError('Preserve completed view diagnosis')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());rows=[];fresh=0;seen=set()
    for r in p['requests']:
        file=OUT/'responses'/(r['request_sha256']+'.json');reused=r['exact_previous_response'] is not None;assert sha(r['image'])==r['image_sha256']
        if not file.exists():
            if reused:
                assert sha(r['exact_previous_response'])==r['exact_previous_response_sha256'];reply=read(r['exact_previous_response'])
            else:
                for _ in range(600):
                    if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                    try:
                        req=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(r['payload']).encode(),headers={'Content-Type':'application/json'})
                        with urllib.request.urlopen(req,timeout=300) as f:reply=json.load(f)
                        fresh+=1;break
                    except urllib.error.HTTPError as e:
                        if e.code==503:time.sleep(.2);continue
                        raise
                else:raise RuntimeError('Shared reviewer busy')
            save(file,reply)
        reply=read(file);stats=reply['local_vlm'];assert stats['model_config_sha256']==p['model_config_sha256'] and stats['actual_image_token_counts']==[256];decision=json.loads(reply['choices'][0]['message']['content']);row={k:v for k,v in r.items() if k!='payload'};row.update(decision=decision,single_base=single_physical_base(decision),repeated_evidence=repeated_evidence(decision),response=str(file),response_sha256=sha(file),same_batch_reuse=r['request_sha256'] in seen,exact_prior_reuse=reused,actual_tokens=[256]);seen.add(r['request_sha256']);rows.append(row);save(OUT/'progress.json',{'rows':rows,'actual_fresh_calls':fresh});print({'candidate':r['candidate'][-7:],'view':r['view'],'kind':r['kind'],'layout':r['layout'],'single':row['single_base'],'decision':decision},flush=True)
    comparisons=[]
    for candidate in p['selected_source_candidates']:
        layouts={}
        for layout in ['square','aspect']:
            real=[r for r in rows if r['candidate']==candidate and r['layout']==layout and r['kind']=='real'];controls=[r for r in rows if r['candidate']==candidate and r['layout']==layout and r['kind']!='real'];control_ok=all(not r['single_base'] and r['decision']['category']!='pallet_base' and not compatible_pallet_subject({'object':r['decision']['object']}) for r in controls);layouts[layout]={'both_original_single_base_views_passed':all(r['single_base'] for r in real),'all_matched_controls_reject_pallet_identity':control_ok,'qualified_structure_diagnostic':all(r['single_base'] for r in real) and control_ok}
        comparisons.append({'candidate':candidate,'layouts':layouts})
    assert all(sha(f)==h for f,h in p['protected'].items());save(OUT/'completed.json',{'rows':rows,'comparisons':comparisons,'actual_new_VLM_calls':len({r['request_sha256'] for r in rows if not r['exact_prior_reuse']}),'fresh_this_execution':fresh,'exact_prior_requests_reused':len({r['request_sha256'] for r in rows if r['exact_prior_reuse']}),'same_batch_reuses':sum(r['same_batch_reuse'] for r in rows),'new_masks':0,'new_Agent_runs':0,'not_adopted_or_installed':True});print({'fresh_VLM':fresh,'comparisons':comparisons},flush=True)
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
