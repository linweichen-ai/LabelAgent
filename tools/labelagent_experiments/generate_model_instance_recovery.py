"""Actual trained-model instance proposals, SAM refinement, focused review."""
import sys,json,hashlib,io,copy,time,base64,re,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_recovery_20261105';out=ROOT/'outputs/labelagent_recovery_20261105/instance_generation_model_refined'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot'])
 import cv2,numpy as np
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize,_shape_from_mask
 out.mkdir(parents=True,exist_ok=True)
 if (out/'completed.json').exists():raise FileExistsError('Completed model proposals retained')
 config={'sam3_url':cfg['sam_url'],'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_prompt_profile':'compact','llm_max_tokens':256}
 traces={r['id']:r for r in read(out/'model_trace.json')} if (out/'model_trace.json').exists() else {}
 def log(r):traces[r['id']]=copy.deepcopy(r);save(out/'model_trace.json',list(traces.values()))
 def busy(fn,*a,**kw):
  for _ in range(100):
   with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=15) as f:running=json.load(f).get('status')=='running'
   if running:time.sleep(3);continue
   try:return fn(*a,**kw)
   except ValueError as e:
    if 'Another model request' not in str(e):raise
    time.sleep(3)
  raise RuntimeError('Shared service remained busy')
 original=workflow._chat;workflow._chat=lambda *a,**kw:busy(original,*a,**kw)
 protocol={'source':'all32 delivered-model fixed-grid forward outputs','SAM_support':'original independent SAM group intersection fraction >=0.5','baseline_duplicate_IoU':.5,
  'NMS_IoU':.5,'NMS_overlap':.8,'max_proposals_per_image_before_refinement':16,'ranking':'trained score * independent SAM containment',
  'refinement':'real SAM3 on source crop, prior individual model mask, one positive and up to2 negative clicks in other parts of SAM group',
  'failure':'retain original trained hypothesis, mark refinement failure; never call it SAM refined',
  'semantic_review':'selected original pixels only, plus independent raw context with a thin outline; target-specific single-base category required in both',
  'B':'reuse the frozen pure-SAM attempt, including its zero accepted proposals','whole_masks_only':True,'no_baseline_union_intersection':True,
  'selection':'reject whole masks exceeding original global count and union-area budgets; no reference selection','image_evidence_only':True}
 save(out/'generation_protocol.json',protocol);pools={};images={};bmasks={};unions={};records=[]
 def iou(a,b):
  inter=int((a&b).sum());return inter/max(1,int(a.sum())+int(b.sum())-inter)
 def png(im):
  b=io.BytesIO();im.save(b,'PNG');return b.getvalue()
 with workflow.trace_calls(log):
  for cam in [0,1]:
   case=f'camera{cam}_frame76';image=Image.open(ROOT/'outputs/labelagent_efficiency_partial_comparison_20261104/inputs'/(case+'.png')).convert('RGB');w,h=image.size;images[case]=image
   baseline=read(ROOT/'outputs/labelagent_efficiency_partial_comparison_20261104/A_current'/case/'result.json')['shapes'];bmasks[case]=[_rasterize(s,w,h) for s in baseline if s['label']=='pallet'];unions[case]=np.maximum.reduce(bmasks[case]);candidates={}
   for file in sorted((out.parent/'fixed_grid'/case).glob('*/prediction.json')):
    for shape in read(file)['PAL_shapes']:
     mask=_rasterize(shape,w,h)
     if not mask.any() or any(iou(mask,b)>=.5 for b in bmasks[case]):continue
     fraction=float((mask&unions[case]).sum())/int(mask.sum())
     if fraction<.5:continue
     key=hashlib.sha256(case.encode()+mask.tobytes()).hexdigest();row={'key':key,'case':case,'shape':shape,'trained_score':float(shape['score']),'SAM_containment':fraction,'rank':float(shape['score'])*fraction,'source_prediction':str(file.relative_to(out.parent))}
     if key not in candidates or row['rank']>candidates[key]['rank']:candidates[key]=row
   pool=[];masks=[]
   for row in sorted(candidates.values(),key=lambda r:(-r['rank'],r['key'])):
    mask=_rasterize(row['shape'],w,h)
    if any(iou(mask,b)>=.5 or int((mask&b).sum())/max(1,min(int(mask.sum()),int(b.sum())))>=.8 for b in masks):continue
    pool.append(row);masks.append(mask)
    if len(pool)>=16:break
   pools[case]=pool;save(out/case/'pre_refinement.json',pool)
   for row,mask in zip(pool,masks):
    yy,xx=np.where(mask);pad=32;box=[max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(w,int(xx.max())+pad+1),min(h,int(yy.max())+pad+1)];x1,y1,x2,y2=box;local=mask[y1:y2,x1:x2]
    d=cv2.distanceTransform(local,cv2.DIST_L2,5);y,x=np.unravel_index(np.argmax(d),d.shape);points=[{'x':int(x),'y':int(y),'label':1}]
    negatives=(unions[case][y1:y2,x1:x2]!=0)&(local==0)
    for _ in range(2):
     if not negatives.any():break
     d=cv2.distanceTransform(negatives.astype('uint8'),cv2.DIST_L2,5);y,x=np.unravel_index(np.argmax(d),d.shape);points.append({'x':int(x),'y':int(y),'label':0});cv2.circle(negatives.view('uint8'),(int(x),int(y)),8,0,-1)
    folder=out/'refinement'/row['key'];folder.mkdir(parents=True,exist_ok=True);raw=png(image.crop(box));prior=png(Image.fromarray(local*255));(folder/'input.png').write_bytes(raw);(folder/'prior.png').write_bytes(prior)
    cached=folder/'response.json'
    if cached.exists():response=read(cached)
    else:
     try:response=busy(workflow.sam_request,config,'/segment_points',raw,points=points,mask=base64.b64encode(prior).decode())
     except ValueError as error:
      if '无法排除' not in str(error) and '提示点' not in str(error):raise
      response={'rejected_reason':str(error)}
     save(cached,response)
    refined=copy.deepcopy(row['shape']);accepted=False
    if response.get('parts') and not response.get('warnings'):
     m=_rasterize(response,x2-x1,y2-y1);mapped=np.zeros((h,w),np.uint8);mapped[y1:y2,x1:x2]=m;s=_shape_from_mask(mapped,'pallet',1)
     if s:refined=s;refined['score']=row['trained_score'];accepted=True
    record={**row,'original_model_shape':row['shape'],'shape':refined,'refinement_accepted':accepted,'refinement_warning':response.get('warnings',[]),'refinement_rejection':response.get('rejected_reason'),'clicks':points,'crop':box}
    records.append(record);save(out/'refinement_progress.json',{'rows':records});print(json.dumps({'case':case,'key':row['key'][:8],'refined':accepted}),flush=True)
  schema={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':120},'category':{'type':'string','enum':['single_visible_pallet_base','multiple_pallet_group','cargo_or_other','uncertain']}},'required':['object','category'],'additionalProperties':False}
  prompt=('Identify ONLY the selected target, not other nearby objects. The image is either selected original pixels on gray background, or original context with one thin target outline. '
   'Ignore the gray background, outline and all unselected neighbors. Describe visible target material and structure. '
   'Use single_visible_pallet_base for one actual pallet deck and its feet, including visible parts under occlusion; multiple_pallet_group only if the SELECTED target itself spans multiple distinct pallet decks. '
   'Boxes, people, robot and floor are cargo_or_other; blank or unresolved evidence is uncertain. Never assume a pallet is present. A colored line is never the object.')
  def classify(im):
   b=png(im);key=hashlib.sha256(b+prompt.encode()+json.dumps(schema,sort_keys=True).encode()).hexdigest();file=out/'visual_cache'/(key+'.json')
   decision=read(file)['decision'] if file.exists() else workflow.chat(config,prompt,b,response_schema=schema)
   if decision.get('category') not in schema['properties']['category']['enum']:raise ValueError('Invalid target evidence')
   save(file,{'decision':decision,'image_sha256':hashlib.sha256(b).hexdigest()});file.with_suffix('.png').write_bytes(b);return decision,str(file.with_suffix('.png'))
  controls=[]
  for color in ['gray','white']:
   d,f=classify(Image.new('RGB',(256,256),color));controls.append({'color':color,'decision':d,'passed':d['category']=='uncertain'})
  save(out/'blank_controls.json',{'synthetic_functional_control_only':True,'rows':controls,'all_passed':all(v['passed'] for v in controls)})
  verified=[]
  for row in records:
   case=row['case'];image=images[case];w,h=image.size;mask=_rasterize(row['shape'],w,h);yy,xx=np.where(mask);pad=max(24,int(max(xx.max()-xx.min(),yy.max()-yy.min())*.5));box=[max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(w,int(xx.max())+pad+1),min(h,int(yy.max())+pad+1)]
   raw=np.array(image.crop(box)).copy();region=mask[box[1]:box[3],box[0]:box[2]];selected=np.full_like(raw,128);selected[region!=0]=raw[region!=0];context=raw.copy();cv2.drawContours(context,cv2.findContours(region,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,30,220),1)
   views=[classify(Image.fromarray(selected)),classify(Image.fromarray(context))];row={**row,'visual_decisions':[v[0] for v in views],'visual_evidence':[v[1] for v in views]}
   row['visual_confirmed']=all(v[0]['category']=='single_visible_pallet_base' and re.search(r'pallet|skid|托盘',v[0]['object'],re.I) and not re.search(r'outline|contour|locator',v[0]['object'],re.I) for v in views)
   verified.append(row);save(out/'verification_progress.json',{'rows':verified});print(json.dumps({'case':case,'key':row['key'][:8],'confirmed':row['visual_confirmed']}),flush=True)
  selected={case:[] for case in unions};union={case:u.copy() for case,u in unions.items()};initial_area=sum(int(m.sum()) for m in union.values());initial_count=sum(len(v) for v in bmasks.values());rejected=[]
  for row in sorted((r for r in verified if r['visual_confirmed']),key=lambda r:(-r['rank'],r['key'])):
   case=row['case'];w,h=images[case].size;mask=_rasterize(row['shape'],w,h);novel=int(((mask!=0)&(union[case]==0)).sum())
   if sum(len(v) for v in selected.values())+initial_count>=int(1.5*initial_count) or sum(int(v.sum()) for v in union.values())+novel>1.25*initial_area:
    rejected.append({'key':row['key'],'reason':'whole mask over original count/area budget'});continue
   old=[_rasterize(r['shape'],w,h) for r in selected[case]]
   if any(iou(mask,b)>=.5 or int((mask&b).sum())/max(1,min(int(mask.sum()),int(b.sum())))>=.8 for b in old):continue
   selected[case].append(row);union[case]|=mask
  for case in unions:
   save(out/case/'C_children.json',[r['shape'] for r in selected[case]])
   save(out/case/'B_children.json',read(out.parent/'instance_generation'/case/'B_children.json'))
  save(out/'C_selection.json',{'selected':selected,'rejected_whole_masks':rejected,'PAL_count':initial_count+sum(len(v) for v in selected.values()),'PAL_union_pixels':sum(int(v.sum()) for v in union.values()),'original_union_pixels':initial_area,'no_pixel_clipping':True})
  save(out/'completed.json',{'status':'complete','references_used_for_selection':False,'annotation_writes':0,'whole_masks_only':True,'model_trace_records':len(traces),'cases':{c:{a:len(read(out/c/(a+'_children.json'))) for a in ['B','C']} for c in unions},'blank_controls_passed':all(v['passed'] for v in controls)})
 print(json.dumps(read(out/'completed.json')),flush=True)
if __name__=='__main__':main()
