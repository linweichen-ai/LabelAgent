"""Real per-gap visual decisions; deterministic, mask-bounded tool arguments."""
import copy,hashlib,io,json,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def action_schema():
    def branch(action,semantic):
        return {'type':'object','properties':{
            'action':{'type':'string','enum':[action]},'semantic':{'type':'string','enum':[semantic]},
            'reason':{'type':'string','minLength':1,'maxLength':80}},
            'required':['action','semantic','reason'],'additionalProperties':False}
    return {'anyOf':[branch('keep','non_ground'),branch('fill_gap','ground')]}


def validate_action(value):
    if (not isinstance(value,dict) or set(value)!={'action','semantic','reason'}
        or (value.get('action'),value.get('semantic')) not in [('keep','non_ground'),('fill_gap','ground')]
        or not isinstance(value.get('reason'),str) or not value['reason'].strip()):
        raise ValueError('Missing or contradictory explicit visual action')
    return value


def gap_polygon(hole,width,height):
    x1,y1,x2,y2=hole['bbox']
    left=max(0.,x1-.25);top=max(0.,y1-.25)
    right=min(width-1.,x2-.25);bottom=min(height-1.,y2-.25)
    return [[left,top],[right,top],[right,bottom],[left,bottom]]


def evidence_sheet(image,floor,hole):
    x1,y1,x2,y2=hole['bbox'];w,h=image.size
    margin=max(32,min(96,int(max(x2-x1,y2-y1)*.4)))
    crop=(max(0,x1-margin),max(0,y1-margin),min(w,x2+margin),min(h,y2+margin))
    binary=Image.fromarray((floor*255).astype('uint8')).convert('RGB')
    marked=np.asarray(binary).copy();marked[hole['mask']!=0]=(255,40,120)
    sheet=Image.new('RGB',(1152,450),'#172333');draw=ImageDraw.Draw(sheet)
    for i,(name,panel) in enumerate(zip(['original crop','current floor mask','pink: proposed gap'],
                                      [image.crop(crop),binary.crop(crop),Image.fromarray(marked).crop(crop)])):
        panel.thumbnail((376,396),Image.Resampling.LANCZOS if i==0 else Image.Resampling.NEAREST)
        sheet.paste(panel,(i*384+(384-panel.width)//2,40+(396-panel.height)//2))
        draw.text((i*384+10,12),name,fill='white')
    output=io.BytesIO();sheet.save(output,'PNG');return output.getvalue(),list(crop)


def raw_verifier_schema():
    return {'type':'object','properties':{
        'object':{'type':'string','minLength':1,'maxLength':100},
        'category':{'type':'string','enum':['bare_floor_or_floor_marking','robot_or_machine','person',
            'pallet_or_cargo','wall_or_rack','mixed_or_uncertain']}},
        'required':['object','category'],'additionalProperties':False}


def raw_gap_image(source,hole,crop):
    import cv2
    raw=np.asarray(source.crop(crop)).copy()
    local=hole['mask'][crop[1]:crop[3],crop[0]:crop[2]].astype('uint8')
    contours,_=cv2.findContours(local,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(raw,contours,-1,(255,235,0),1)
    buffer=io.BytesIO();Image.fromarray(raw).save(buffer,'PNG');return buffer.getvalue()


def ordinary_floor_confirmed(verification):
    return (verification.get('category')=='bare_floor_or_floor_marking'
        and not re.search(r'\b(robot|machine|person|human|wall|rack|box|cargo|pallet|chair|equipment)\b',
                          verification.get('object',''),re.I))


def install_review_batches(workflow,batch_size=1,operation_cap=3,callback=None,cache_dir=None):
    original=workflow.review
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    schema=action_schema()
    def review(config,image,overlay,description,shapes,labels,history,**kwargs):
        holes=kwargs.get('holes') or []
        if not holes:return original(config,image,overlay,description,shapes,labels,history,**kwargs)
        geometry=kwargs.get('candidate_geometry')
        if geometry is None:raise ValueError('Exact candidate geometry is required for per-gap evidence')
        source=Image.open(io.BytesIO(image)).convert('RGB');width,height=source.size
        floor=np.zeros((height,width),np.uint8)
        from source.agent.edit_agent import _rasterize,_valid_region
        for shape in geometry:
            if shape['label'].lower() in ('floor','ground','地面'):floor|=_rasterize(shape,width,height)
        # Object identity edits cannot generate arbitrary boundary polygons.
        global_config=dict(config,llm_object_only_review=True)
        global_kwargs=dict(kwargs,holes=[],gap_images=[],hole_image=None)
        global_decision=original(global_config,image,overlay,description,shapes,labels,history,**global_kwargs)
        protected=set(config.get('protected_initial_ids',[]))
        guarded=[op for op in global_decision['operations'] if any(identifier in protected for identifier in op.get('shape_ids',[]))]
        assessments=[];operations=[op for op in global_decision['operations'] if op not in guarded];bindings=[]
        for number,hole in enumerate(holes,1):
            sheet,crop=evidence_sheet(source,floor,hole)
            prompt=('Inspect ONE proposed floor gap using the three panels in this single image. '
                'Left is the uncolored original crop. Middle is the current floor mask (white included, black excluded). '
                'Right shows this exact proposed gap in pink; pink is not already included. '
                'Choose fill_gap and ground ONLY if the ENTIRE pink region is visibly ordinary horizontal floor or a floor marking. '
                'Choose keep and non_ground if the region contains equipment, cargo, a person, wall, robot, occlusion, mixed non-floor, '
                'or cannot safely be filled as a whole. Never infer floor only from a hole shape or pink color. '
                'Your fill_gap action authorizes adding ONLY pixels of this exact proposed gap. No outside pixel is allowed. '
                'Do not output coordinates. Give a brief concrete visual reason. No human-confirmed seven-class truth was supplied. '
                'Rules: '+description['description'])
            key=hashlib.sha256(sheet+json.dumps({'prompt':prompt,'schema':schema,'model':config['llm_model']},sort_keys=True).encode()).hexdigest()
            file=cache_dir/(key+'.json');picture=cache_dir/(key+'.png')
            cached=file.exists()
            if cached:
                row=json.loads(file.read_text(encoding='utf-8'));decision=validate_action(row['decision'])
                if row['evidence_sha256']!=hashlib.sha256(sheet).hexdigest():raise ValueError('Evidence binding changed')
            else:
                local_config=dict(config,llm_max_tokens=256)
                last=None
                for attempt in range(workflow.MAX_PROTOCOL_RETRIES+1):
                    state={'attempt':attempt+1,'feedback':str(last) if last else None};token=workflow._review_call.set(state)
                    try:
                        try:
                            decision=workflow.chat(local_config,prompt+(' Previous reply was invalid: '+str(last) if last else ''),sheet,response_schema=schema)
                            try:validate_action(decision)
                            except ValueError as error:raise workflow.ModelOutputError(str(error),json.dumps(decision))
                        except workflow.ModelOutputError as error:
                            last=error;record=state.get('record')
                            if record is not None:
                                record.update(status='rejected',error=str(error),retrying=attempt<workflow.MAX_PROTOCOL_RETRIES);workflow._emit_trace(record)
                            if attempt==workflow.MAX_PROTOCOL_RETRIES:raise
                        else:break
                    finally:workflow._review_call.reset(token)
                picture.write_bytes(sheet)
                row={'decision':decision,'evidence_sha256':hashlib.sha256(sheet).hexdigest(),'model':config['llm_model'],
                     'schema':schema,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'decision_source':'real_VLM_request'}
                file.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            proposed=copy.deepcopy(decision);verification=None;verification_file=None
            if decision['action']=='fill_gap':
                raw=raw_gap_image(source,hole,crop)
                verification_prompt=('Describe the actual material or object INSIDE the thin yellow contour in this uncolored photo. '
                    'The line only locates the region; it does not recolor the interior. First name the visible object, then choose category. '
                    'bare_floor_or_floor_marking means the ordinary horizontal warehouse walking surface, including its paint marks. '
                    'A robot or machine STANDING ON floor is robot_or_machine, never floor. '
                    'Pallets, boxes and cargo are pallet_or_cargo; walls and racks are wall_or_rack. '
                    'If mixed, unclear or not safely all floor, choose mixed_or_uncertain. Judge the outlined interior, not surrounding floor. '
                    'No segmentation overlay, white mask or repair suggestion is provided.')
                verification_key=hashlib.sha256(raw+verification_prompt.encode()+json.dumps(raw_verifier_schema(),sort_keys=True).encode()).hexdigest()
                verification_file=cache_dir/(verification_key+'_raw.json');raw_file=cache_dir/(verification_key+'_raw.png')
                if verification_file.exists():
                    previous=json.loads(verification_file.read_text(encoding='utf-8'))
                    if previous.get('image_sha256')!=hashlib.sha256(raw).hexdigest():raise ValueError('Raw-image cache binding changed')
                    verification=previous['decision']
                else:
                    verification=workflow.chat(dict(config,llm_max_tokens=256),verification_prompt,raw,response_schema=raw_verifier_schema())
                    if (not isinstance(verification,dict) or set(verification)!={'object','category'}
                        or verification['category'] not in raw_verifier_schema()['properties']['category']['enum']
                        or not isinstance(verification['object'],str) or not verification['object'].strip()):
                        raise ValueError('Raw-photo verification is incomplete; no fill allowed')
                    raw_file.write_bytes(raw)
                    verification_file.write_text(json.dumps({'decision':verification,'image_sha256':hashlib.sha256(raw).hexdigest(),
                        'decision_source':'independent real VLM raw-photo request'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                if not ordinary_floor_confirmed(verification):
                    decision={'action':'keep','semantic':'non_ground','reason':'Raw-photo veto: '+verification['object']}
            assessments.append({'hole_id':hole['id'],'semantic':decision['semantic'],'reason':decision['reason']})
            binding={'hole_id':hole['id'],'number':number,'cache':cached,'evidence_file':str(picture),'crop':crop,
                     'decision':copy.deepcopy(decision),'proposed_decision':proposed,'raw_photo_verification':verification,
                     'raw_verification_file':str(verification_file) if verification_file else None,
                     'geometry_source':'input gap mask, never model-drawn coordinates'}
            if decision['action']=='fill_gap':
                polygon=gap_polygon(hole,width,height)
                if not _valid_region(polygon,width,height):raise ValueError('Invalid deterministic gap bounds')
                operations.append({'operation':'fill_gap','shape_ids':[hole['shape_id']],'hole_id':hole['id'],
                                   'polygon':polygon,'reason':decision['reason']})
            bindings.append(binding)
            if callback:callback({'round':len(history)+1,'local_gap':binding})
        selected=operations[:operation_cap]
        merged={'status':'continue' if selected else 'done',
                'reason':'Object review and explicit per-gap visual actions; no model coordinates.',
                'hole_assessment':assessments,'operations':selected,'guarded_proposals':guarded}
        if 'sam_assessment' in global_decision:merged['sam_assessment']=global_decision['sam_assessment']
        merged=workflow._validate_review_result(merged,holes,shapes,labels,width,height,kwargs.get('sam_comparison'))
        if callback:callback({'round':len(history)+1,'merged':True,'all_holes_assessed':True,'original_holes':len(holes),
                             'proposed_operations':len(operations),'selected_operations':len(selected),'deferred_operations':len(operations)-len(selected),
                             'global_decision':global_decision})
        return merged
    workflow.review=review
    return original
