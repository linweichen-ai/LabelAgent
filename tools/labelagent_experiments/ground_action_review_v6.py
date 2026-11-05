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
    # Gap bboxes in edit_agent include their right/bottom endpoint.
    right=min(width-1.,x2+.75);bottom=min(height-1.,y2+.75)
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
                          verification.get('object','').replace('_',' '),re.I))


def approved_gap_sam_operation(holes,assessments,geometry,width,height,negative_ids=None):
    import cv2
    from source.agent.edit_agent import _rasterize
    approved={r['hole_id'] for r in assessments if r['semantic']=='ground'}
    groups={}
    for hole in holes:
        if hole['id'] in approved:groups.setdefault(hole['shape_id'],[]).append(hole)
    if not groups:return None
    identifier,group=max(groups.items(),key=lambda kv:(len(kv[1]),-kv[0]))
    group=group[:31]
    if len(group)<2:return None
    points=[]
    def point(mask,label):
        distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
        y,x=np.unravel_index(np.argmax(distance),distance.shape)
        return {'x':int(x),'y':int(y),'label':label}
    for hole in group:points.append(point(hole['mask'],1))
    parent=next(s for s in geometry if s['id']==identifier)
    points.append(point(_rasterize(parent,width,height),1))
    kept=sorted([h for h in holes if h['shape_id']==identifier and h['id'] not in approved
                 and (negative_ids is None or h['id'] in negative_ids)],key=lambda h:-h['area'])
    points += [point(h['mask'],0) for h in kept[:32-len(points)]]
    box=[min(h['bbox'][0] for h in group),min(h['bbox'][1] for h in group),
         max(h['bbox'][2] for h in group),max(h['bbox'][3] for h in group)]
    return {'operation':'sam_points','shape_ids':[identifier],
            'polygon':[[0.,0.],[width-.25,0.],[width-.25,height-.25],[0.,height-.25]],
            'points':points,'reviewed_gap_ids':[h['id'] for h in group],
            'reason':'One real SAM3 request for explicitly reviewed ground gaps; outside pixels locked.'}


def sam_gap_assessment(workflow,config,image,geometry,comparison,labels,cache_dir):
    from source.agent.edit_agent import _rasterize
    source=Image.open(io.BytesIO(image)).convert('RGB');w,h=source.size
    before=next(s for s in comparison['before_shapes'] if s['id']==comparison['shape_id'])
    after=next(s for s in geometry if s['id']==comparison['shape_id'])
    old=_rasterize(before,w,h);new=_rasterize(after,w,h)
    lookup={x['id']:x for x in workflow.candidate_holes(comparison['before_shapes'],labels,w,h)}
    gaps=[lookup[identifier] for identifier in comparison['reviewed_gap_ids']]
    allowed=np.maximum.reduce([g['mask'] for g in gaps]);added=(new!=0)&(old==0)
    if np.any((old!=0)&(new==0)) or np.any(added&(allowed==0)):
        raise ValueError('Unapproved pixel change in SAM before/after comparison')
    pages=[];counts=[]
    for start in range(0,len(gaps),4):
        selected=gaps[start:start+4];page=Image.new('RGB',(1080,len(selected)*300),'#172333');draw=ImageDraw.Draw(page)
        for row,gap in enumerate(selected):
            x1,y1,x2,y2=gap['bbox'];crop=(max(0,x1-24),max(0,y1-24),min(w,x2+25),min(h,y2+25))
            panels=[source.crop(crop),Image.fromarray(old*255).convert('RGB').crop(crop),Image.fromarray(new*255).convert('RGB').crop(crop)]
            filled=int((added&(gap['mask']!=0)).sum());counts.append({'id':gap['id'],'target_pixels':int(gap['mask'].sum()),'newly_included_pixels':filled})
            draw.text((10,row*300+5),f"approved gap {start+row+1}: original | BEFORE | AFTER; new pixels {filled}",fill='white')
            for column,panel in enumerate(panels):
                panel.thumbnail((352,260),Image.Resampling.LANCZOS if column==0 else Image.Resampling.NEAREST)
                page.paste(panel,(column*360+(360-panel.width)//2,row*300+30+(260-panel.height)//2))
        stream=io.BytesIO();page.save(stream,'PNG');pages.append(stream.getvalue())
    schema={'type':'object','properties':{'improved':{'type':'boolean'},'reason':{'type':'string','minLength':1,'maxLength':120}},'required':['improved','reason'],'additionalProperties':False}
    prompt=('Compare a REAL SAM3 multi-point floor repair. Each row shows the same uncolored source crop, '
        'BEFORE binary floor mask, AFTER binary floor mask (white included, black excluded). '
        'Every target gap was independently reviewed as visible ordinary ground using an uncolored photo. '
        'Program checks confirm no old floor pixels removed and no added pixel outside those exact approved gaps. '
        'Judge whether SAM improved inclusion of these reviewed floor gaps without filling equipment. '
        'Do not call an unmodified gap improved. Return improved true only if the actual after masks visibly improve the approved gaps. '
        'These are candidate edits, not ground truth. Exact pixel changes: '+json.dumps(counts,separators=(',',':')))
    key=hashlib.sha256(b''.join(pages)+prompt.encode()).hexdigest();file=cache_dir/(key+'_sam_comparison.json')
    if file.exists():return json.loads(file.read_text(encoding='utf-8'))['decision']
    decision=workflow.chat(dict(config,llm_max_tokens=256),prompt,pages[0],extra_images=tuple(pages[1:]),response_schema=schema)
    if type(decision.get('improved')) is not bool or not isinstance(decision.get('reason'),str) or not decision['reason'].strip():
        raise ValueError('SAM repair was not assessed by the model')
    for index,data in enumerate(pages):(cache_dir/(key+f'_sam_comparison_{index}.png')).write_bytes(data)
    file.write_text(json.dumps({'decision':decision,'counts':counts,'pixel_guard':{'removed':0,'added_outside_approved':0}},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return decision


def install_review_batches(workflow,batch_size=1,operation_cap=3,callback=None,cache_dir=None):
    original=workflow.review
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    schema=action_schema()
    def review(config,image,overlay,description,shapes,labels,history,**kwargs):
        holes=kwargs.get('holes') or []
        floor_task=any(label.lower() in ('floor','ground','地面') for label in labels)
        if not floor_task:return original(config,image,overlay,description,shapes,labels,history,**kwargs)
        geometry=kwargs.get('candidate_geometry')
        if geometry is None:raise ValueError('Exact candidate geometry is required for per-gap evidence')
        source=Image.open(io.BytesIO(image)).convert('RGB');width,height=source.size
        floor=np.zeros((height,width),np.uint8)
        from source.agent.edit_agent import _rasterize,_valid_region
        for shape in geometry:
            if shape['label'].lower() in ('floor','ground','地面'):floor|=_rasterize(shape,width,height)
        # Object identity edits cannot generate arbitrary boundary polygons.
        global_config=dict(config,llm_object_only_review=True)
        comparison=kwargs.get('sam_comparison')
        assessment=(sam_gap_assessment(workflow,config,image,geometry,comparison,labels,cache_dir)
                    if comparison and comparison.get('reviewed_gap_ids') else None)
        global_kwargs=dict(kwargs,holes=[],gap_images=[],hole_image=None,sam_comparison=None if assessment is not None else comparison)
        foreground=[s for s in shapes if s.get('label','').lower() not in ('floor','ground','地面')]
        foreground_geometry=[s for s in geometry if s['label'].lower() not in ('floor','ground','地面')]
        object_overlay=workflow.candidate_overlay(image,foreground_geometry)
        immutable=all(s['id'] in config.get('protected_initial_ids',[]) for s in foreground)
        reference=kwargs.get('reference')
        ref_key=None if reference is None else {'image':hashlib.sha256(reference['image']).hexdigest(),'marked':hashlib.sha256(reference['marked_image']).hexdigest(),'annotations':reference['annotations']}
        key=hashlib.sha256(image+object_overlay+json.dumps({'shapes':foreground,'labels':labels,'description':description,
            'reference':ref_key,'model':config['llm_model']},sort_keys=True).encode()).hexdigest()
        file=cache_dir/(key+'_identity.json')
        cached_identity=immutable and file.exists() and (comparison is None or assessment is not None)
        if cached_identity:global_decision=json.loads(file.read_text(encoding='utf-8'))['decision']
        else:
            global_decision=original(global_config,image,object_overlay,description,foreground,labels,[] if immutable else history,**global_kwargs)
            if immutable and (comparison is None or assessment is not None):file.write_text(json.dumps({'decision':global_decision,'source':'real VLM on immutable foreground evidence'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        global_decision=copy.deepcopy(global_decision)
        if assessment is not None:global_decision['sam_assessment']=assessment
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
        prior_batches=[op for row in history for op in row.get('decision',{}).get('operations',[]) if op.get('reviewed_gap_ids')]
        rollback=(any(row.get('sam_rollback') or any(t.get('fallback') for t in row.get('tool_results',[])) for row in history)
                  or assessment is not None and assessment['improved'] is False)
        if config.get('llm_batch_reviewed_gaps',False) and len(operations)>operation_cap and len(prior_batches)<2 and not rollback:
            verified_negative={r['hole_id'] for r in bindings if (r.get('raw_photo_verification') or {}).get('category')
                               in ('robot_or_machine','person','pallet_or_cargo','wall_or_rack')}
            if config.get('sam_gap_negative_policy')=='positive_only':verified_negative=set()
            batch=approved_gap_sam_operation(holes,assessments,geometry,width,height,verified_negative)
            if batch is not None:
                if prior_batches and config.get('llm_residual_gap_recovery',False):
                    from residual_gap_policy import choose_residual_schedule
                    scheduled=choose_residual_schedule(holes,assessments,operations,geometry,width,height,
                        config.get('_gap_points_by_shape',{}),config.get('_gap_feedback',{}),approved_gap_sam_operation)
                    selected=scheduled if scheduled is not None else [batch]
                else:
                    from spatial_gap_recovery import choose_context_first
                    selected=choose_context_first(holes,assessments,operations,geometry,width,height,approved_gap_sam_operation)
        merged={'status':'continue' if selected else 'done',
                'reason':'Object review and explicit per-gap visual actions; no model coordinates.',
                'hole_assessment':assessments,'operations':selected,'guarded_proposals':guarded}
        if 'sam_assessment' in global_decision:merged['sam_assessment']=global_decision['sam_assessment']
        merged=workflow._validate_review_result(merged,holes,shapes,labels,width,height,kwargs.get('sam_comparison'))
        if callback:callback({'round':len(history)+1,'merged':True,'all_holes_assessed':True,'original_holes':len(holes),
                             'proposed_operations':len(operations),'selected_operations':len(selected),'deferred_operations':len(operations)-len(selected),
                             'global_decision':global_decision,'identity_reply_reused':cached_identity,
                             'ground_actions_compiled':sum(a['semantic']=='ground' for a in assessments),
                             'SAM_batch_selected':bool(selected and selected[0].get('reviewed_gap_ids'))})
        return merged
    workflow.review=review
    return original
