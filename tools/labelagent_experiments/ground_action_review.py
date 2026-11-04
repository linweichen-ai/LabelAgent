"""Real per-gap visual decisions; deterministic, mask-bounded tool arguments."""
import copy,hashlib,io,json
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
        assessments=[];operations=list(global_decision['operations']);bindings=[]
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
            assessments.append({'hole_id':hole['id'],'semantic':decision['semantic'],'reason':decision['reason']})
            binding={'hole_id':hole['id'],'number':number,'cache':cached,'evidence_file':str(picture),'crop':crop,
                     'decision':copy.deepcopy(decision),'geometry_source':'input gap mask, never model-drawn coordinates'}
            if decision['action']=='fill_gap':
                polygon=gap_polygon(hole,width,height)
                if not _valid_region(polygon,width,height):raise ValueError('Invalid deterministic gap bounds')
                operations.append({'operation':'fill_gap','shape_ids':[hole['shape_id']],'hole_id':hole['id'],
                                   'polygon':polygon,'reason':decision['reason']})
            bindings.append(binding)
            if callback:callback({'round':len(history)+1,'local_gap':binding})
        selected=operations[:operation_cap]
        merged={'status':'continue' if selected or global_decision['status']=='continue' else 'done',
                'reason':'Object review and explicit per-gap visual actions; no model coordinates.',
                'hole_assessment':assessments,'operations':selected}
        if 'sam_assessment' in global_decision:merged['sam_assessment']=global_decision['sam_assessment']
        merged=workflow._validate_review_result(merged,holes,shapes,labels,width,height,kwargs.get('sam_comparison'))
        if callback:callback({'round':len(history)+1,'merged':True,'all_holes_assessed':True,'original_holes':len(holes),
                             'proposed_operations':len(operations),'selected_operations':len(selected),'deferred_operations':len(operations)-len(selected),
                             'global_decision':global_decision})
        return merged
    workflow.review=review
    return original
