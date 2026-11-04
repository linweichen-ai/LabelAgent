"""Image-only SAM3 instance decomposition inside the preserved SAM PAL union."""
import sys,json,hashlib,io,time,copy,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'task_workspaces/labelagent_efficiency_20261104'
OUT=ROOT/'outputs/labelagent_efficiency_20261104'
OLD=ROOT/'outputs/labelagent_hybrid_verified_20261104'
sys.path[:0]=[str(WORK/'source_snapshot')]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import cv2,numpy as np
 from PIL import Image,ImageDraw,ImageFont
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize,_shape_from_mask
 cfg=read(ROOT/'task_workspaces/labelagent_hybrid_repair_20261104/config_verified.json')
 config={'sam3_url':cfg['sam_url'],'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_prompt_profile':'compact','llm_max_tokens':256,'llm_keyed_holes':True}
 out=OUT/'instance_generation';out.mkdir(exist_ok=True)
 if (out/'completed.json').exists():raise FileExistsError('Completed generation retained')
 def yield_workbench():
  while True:
   with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=15) as f:r=json.load(f)
   if r['status']!='running':return
   print(json.dumps({'waiting_for_workbench':True}),flush=True);time.sleep(5)
 trace={r['id']:r for r in read(out/'model_trace.json')} if (out/'model_trace.json').exists() else {}
 def log(row):trace[row['id']]=copy.deepcopy(row);save(out/'model_trace.json',list(trace.values()))
 def busy_call(fn,*args,**kwargs):
  for _ in range(80):
   yield_workbench()
   try:return fn(*args,**kwargs)
   except ValueError as e:
    if 'Another model request' not in str(e):raise
    time.sleep(3)
  raise RuntimeError('Model service remained busy')
 original_chat=workflow._chat
 workflow._chat=lambda *a,**k:busy_call(original_chat,*a,**k)
 parents={};baseline={};candidate_masks={};windows={};geometry={}
 def bounds(mask,pad,w,h):
  yy,xx=np.where(mask);return [max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(w,int(xx.max())+pad+1),min(h,int(yy.max())+pad+1)]
 def box_iou(a,b):
  inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
 with workflow.trace_calls(log):
  # Finish SAM inference before switching the shared worker to visual review.
  for cam in (0,1):
   case=f'camera{cam}_frame76';image=Image.open(OLD/'inputs'/(case+'.png')).convert('RGB');w,h=image.size
   base=[s for s in read(OLD/'A_current'/case/'result.json')['shapes'] if s['label']=='pallet'];bm=[_rasterize(s,w,h) for s in base];union=np.maximum.reduce(bm)
   model=[s for s in read(ROOT/'outputs/labelagent_hybrid_repair_20261104/mask_tiled'/case/'shapes.json') if s['label']=='pallet']
   baseline[case]=base;geometry[case]=(w,h);parents[case]=[]
   for kind,values in [('SAM',base),('trained_model',model)]:
    for shape in values:
     mask=_rasterize(shape,w,h)
     if not mask.any():continue
     box=bounds(mask,32,w,h)
     duplicate=next((r for r in parents[case] if box_iou(box,r['box'])>=.9),None)
     if duplicate:
      if kind=='SAM':duplicate['for_B']=True
      continue
     key=hashlib.sha256(json.dumps([case,box,'pallet']).encode()).hexdigest()
     row={'key':key,'box':box,'source':kind,'for_B':kind=='SAM','for_C':True};parents[case].append(row)
     crop=image.crop(box);encoded=io.BytesIO();crop.save(encoded,'PNG');file=out/'SAM_crop_cache'/(key+'.json')
     response=read(file) if file.exists() else busy_call(workflow.sam_request,config,'/segment_text',encoded.getvalue(),prompt='pallet')
     save(file,response)
     (out/'SAM_crop_cache'/(key+'.png')).write_bytes(encoded.getvalue())
     print(json.dumps({'case':case,'crop':len(parents[case]),'source':kind,'SAM_results':len(response.get('results',[]))}),flush=True)
   save(out/case/'parents.json',parents[case])
   raw=[]
   for parent in parents[case]:
    response=read(out/'SAM_crop_cache'/(parent['key']+'.json'));box=parent['box'];cw,ch=box[2]-box[0],box[3]-box[1]
    for rank,result in enumerate(response.get('results',[])):
     if float(result.get('score',0))<.5:continue
     parts=result.get('parts') or []
     if not parts:continue
     local={'id':1,'label':'pallet','parts':parts};lm=_rasterize(local,cw,ch)
     mapped=np.zeros((h,w),np.uint8);mapped[box[1]:box[3],box[0]:box[2]]=lm
     area=int(mapped.sum());clipped=mapped&union;retained=int(clipped.sum())/max(1,area)
     if retained<.8 or not clipped.any():continue
     cm_area=int(clipped.sum())
     if any(int((clipped&m).sum())/max(1,cm_area+int(m.sum())-int((clipped&m).sum()))>=.5 for m in bm):continue
     shape=_shape_from_mask(clipped,'pallet',len(raw)+1)
     if shape is None:continue
     # Polygon representation must preserve the generation area constraint.
     restored=_rasterize(shape,w,h)
     if np.any((restored!=0)&(union==0)):continue
     identifier=hashlib.sha256(restored.tobytes()).hexdigest()
     shape['score']=float(result['score']);shape['proposal_key']=identifier
     row={'key':identifier,'shape':shape,'for_B':parent['for_B'],'for_C':True,'parent':parent['key'],
          'SAM_score':float(result['score']),'retained_fraction':retained,'rank_score':float(result['score'])*retained}
     prev=next((r for r in raw if r['key']==identifier),None)
     if prev:
      prev['for_B']|=row['for_B'];continue
     raw.append(row)
   save(out/case/'pre_verification_proposals.json',raw)
   candidate_masks[case]=raw
  # Explicit raw-photo semantic gate; no 3D object/point data is imported.
  schema={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':100},
   'category':{'type':'string','enum':['single_visible_pallet_base','multiple_pallet_group','cargo_or_other','uncertain']}},
   'required':['object','category'],'additionalProperties':False}
  font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
  for cam in (0,1):
   case=f'camera{cam}_frame76';image=Image.open(OLD/'inputs'/(case+'.png')).convert('RGB');w,h=geometry[case];verified=[]
   for row in candidate_masks[case]:
    shape=row['shape'];mask=_rasterize(shape,w,h);box=bounds(mask,32,w,h);crop=np.asarray(image.crop(box)).copy()
    region=mask[box[1]:box[3],box[0]:box[2]];contours,_=cv2.findContours(region,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,contours,-1,(255,235,0),1)
    encoded=io.BytesIO();Image.fromarray(crop).save(encoded,'PNG')
    prompt=('Describe the actual object INSIDE the thin yellow outline in this uncolored photo. '
     'A pallet annotation means exactly ONE visible pallet base structure, including its deck/slats and feet, '
     'not the cargo box standing on it and not a stack/group of multiple pallets as one instance. '
     'Choose single_visible_pallet_base only for one complete visible pallet base. Choose multiple_pallet_group '
     'if it contains multiple separate pallet bases; cargo_or_other for boxes, people, robot, floor or racks; '
     'uncertain if it is a fragment, ambiguous or cannot be decided. Preserve original colors; the outline is only a locator.')
    key=hashlib.sha256(encoded.getvalue()+prompt.encode()+json.dumps(schema,sort_keys=True).encode()).hexdigest();file=out/'visual_cache'/(key+'.json')
    decision=read(file)['decision'] if file.exists() else workflow.chat(config,prompt,encoded.getvalue(),response_schema=schema)
    if not isinstance(decision,dict) or decision.get('category') not in schema['properties']['category']['enum']:raise ValueError('Invalid pallet verification')
    save(file,{'decision':decision,'image_sha256':hashlib.sha256(encoded.getvalue()).hexdigest(),'proposal_key':row['key']})
    (out/'visual_cache'/(key+'.png')).write_bytes(encoded.getvalue())
    row={**row,'visual_decision':decision,'visual_evidence':str(out/'visual_cache'/(key+'.png'))}
    if decision['category']=='single_visible_pallet_base':verified.append(row)
    save(out/case/'verification_progress.json',{'processed_key':row['key'],'accepted_so_far':len(verified)})
    print(json.dumps({'case':case,'verified':row['key'][:8],'decision':decision}),flush=True)
   selected={}
   for arm in ('B','C'):
    keep=[];masks=[];budget=int(1.5*len(baseline[case]))-len(baseline[case])
    ordered=sorted((r for r in verified if r['for_'+arm]),key=lambda r:(-r['rank_score'],r['key']))
    for row in ordered:
     mask=_rasterize(row['shape'],w,h);area=int(mask.sum());duplicate=False
     for old in masks:
      inter=int((mask&old).sum());oa=int(old.sum())
      if inter/max(1,area+oa-inter)>=.5 or inter/max(1,min(area,oa))>=.8:duplicate=True;break
     if duplicate:continue
     if len(keep)>=budget:break
     keep.append(row);masks.append(mask)
    selected[arm]=keep
    save(out/case/(arm+'_children.json'),[r['shape'] for r in keep])
   save(out/case/'selected_provenance.json',selected);save(out/case/'all_verified.json',verified)
  save(out/'completed.json',{'status':'complete','references_used_for_selection':False,'annotation_writes':0,
      'new_reference_inference':False,'cases':{c:{a:len(read(out/c/(a+'_children.json'))) for a in ['B','C']} for c in baseline},
      'crop_counts':{c:len(v) for c,v in parents.items()},'model_trace_records':len(trace)})
 print(json.dumps(read(out/'completed.json')),flush=True)
if __name__=='__main__':main()
