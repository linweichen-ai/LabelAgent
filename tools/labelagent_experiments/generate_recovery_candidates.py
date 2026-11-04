"""Unclipped fixed-grid PAL proposals; references are never opened here."""
import sys,json,io,hashlib,copy,time,re,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_recovery_20261105';out=ROOT/'outputs/labelagent_recovery_20261105/instance_generation'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot'])
 import cv2,numpy as np
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize,_shape_from_mask
 out.mkdir(parents=True,exist_ok=True)
 if (out/'completed.json').exists():raise FileExistsError('Completed proposals immutable')
 config={'sam3_url':cfg['sam_url'],'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_prompt_profile':'compact','llm_max_tokens':256}
 trace={r['id']:r for r in read(out/'model_trace.json')} if (out/'model_trace.json').exists() else {}
 def log(r):trace[r['id']]=copy.deepcopy(r);save(out/'model_trace.json',list(trace.values()))
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
 protocol={'full_masks_only':True,'queries':['pallet','wooden pallet base'],'score_min':.5,'pre_verification_cap_per_camera':32,
  'baseline_duplicate_IoU':.5,'proposal_NMS_IoU':.5,'proposal_NMS_overlap':.8,'rank_B':'SAM score +0.05 cross-query agreement','rank_C':'rank_B +0.10 trained-model overlap score',
  'whole_mask_admission':'global count<=1.5A and union<=1.25A; no clipping','two_actual_views_required':True,'blank_functional_controls':['gray','white']}
 save(out/'generation_protocol.json',protocol)
 images={};base={};unions={};pools={};masks={};w=h=0
 def iou(a,b):
  inter=int((a&b).sum());return inter/max(1,int(a.sum())+int(b.sum())-inter)
 with workflow.trace_calls(log):
  for cam in [0,1]:
   case=f'camera{cam}_frame76';image=Image.open(ROOT/'outputs/labelagent_efficiency_partial_comparison_20261104/inputs'/(case+'.png')).convert('RGB');images[case]=image;w,h=image.size
   base[case]=[s for s in read(ROOT/'outputs/labelagent_efficiency_partial_comparison_20261104/A_current'/case/'result.json')['shapes'] if s['label']=='pallet'];bm=[_rasterize(s,w,h) for s in base[case]];unions[case]=np.maximum.reduce(bm)
   raw={};model_masks=[]
   for file in sorted((out.parent/'fixed_grid'/case).glob('*/prediction.json')):
    for s in read(file)['PAL_shapes']:model_masks.append((_rasterize(s,w,h),float(s['score'])))
   for tile in sorted((out.parent/'fixed_grid'/case).glob('*/input.png')):
    meta=read(tile.with_name('prediction.json'));x,y,cw,ch=meta['crop_xywh'];responses=[]
    for qi,prompt in enumerate(protocol['queries']):
     key=hashlib.sha256(tile.read_bytes()+prompt.encode()).hexdigest();cached=out/'SAM_grid_cache'/(key+'.json')
     result=read(cached) if cached.exists() else busy(workflow.sam_request,config,'/segment_text',tile.read_bytes(),prompt=prompt);save(cached,result)
     rows=[]
     for prediction in result.get('results',[]):
      if float(prediction.get('score',0))<.5 or not prediction.get('parts'):continue
      lm=_rasterize(prediction,cw,ch);mapped=np.zeros((h,w),np.uint8);mapped[y:y+ch,x:x+cw]=lm
      if not mapped.any():continue
      shape=_shape_from_mask(mapped,'pallet',1)
      if shape is None:continue
      mapped=_rasterize(shape,w,h);novel=int(((mapped!=0)&(unions[case]==0)).sum())
      if any(iou(mapped,m)>=.5 for m in bm):continue
      # Reject an entire oversized hypothesis; never trim it to fit.
      if novel>sum(int(u.sum()) for u in unions.values())*.25 and len(unions)==2:continue
      identifier=hashlib.sha256(case.encode()+mapped.tobytes()).hexdigest();shape['score']=float(prediction['score'])
      row={'key':identifier,'case':case,'shape':shape,'SAM_score':float(prediction['score']),'query_index':qi,'tile':str(tile.relative_to(out.parent)),
       'new_pixels_outside_original_union':novel,'full_mask_pixels':int(mapped.sum())}
      old=raw.get(identifier)
      if old is None or row['SAM_score']>old['SAM_score']:raw[identifier]=row;masks[identifier]=mapped
      rows.append(identifier)
     responses.append(rows)
    print(json.dumps({'case':case,'tile':tile.parent.name,'query_results':[len(v) for v in responses]}),flush=True)
   candidates=list(raw.values())
   for row in candidates:
    mask=masks[row['key']];agree=max([iou(mask,masks[r['key']]) for r in candidates if r['query_index']!=row['query_index']]+[0])
    ms=max([iou(mask,m)*score for m,score in model_masks]+[0]);row.update(cross_query_agreement=agree,trained_model_overlap_score=ms,rank_B=row['SAM_score']+.05*agree,rank_C=row['SAM_score']+.05*agree+.10*ms)
   # The same image-only NMS pool is reviewed for both arms.
   pool=[]
   for row in sorted(candidates,key=lambda r:(-max(r['rank_B'],r['rank_C']),r['key'])):
    mask=masks[row['key']]
    if any(iou(mask,masks[k['key']])>=.5 or int((mask&masks[k['key']]).sum())/max(1,min(int(mask.sum()),int(masks[k['key']].sum())))>=.8 for k in pool):continue
    pool.append(row)
    if len(pool)>=32:break
   pools[case]=pool;save(out/case/'pre_verification_proposals.json',pool)
  schema={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':120},'category':{'type':'string','enum':['single_visible_pallet_base','multiple_pallet_group','cargo_or_other','uncertain']}},'required':['object','category'],'additionalProperties':False}
  prompt=('Identify the actual object at the CENTER of this raw photo crop. Describe its visible material and structure first. '
   'A single_visible_pallet_base requires one visible load-supporting pallet deck/slats/feet; cargo boxes or robot components are cargo_or_other. '
   'Multiple pallet bases are multiple_pallet_group. Blank, featureless, ambiguous or no identifiable object is uncertain. '
   'A thin outline, if present, only locates a target and is never an object. Never assume a pallet is present. Classify the centered target, not nearby cargo.')
  def classify(photo):
   b=io.BytesIO();photo.save(b,'PNG');data=b.getvalue();key=hashlib.sha256(data+prompt.encode()+json.dumps(schema,sort_keys=True).encode()).hexdigest();file=out/'visual_cache'/(key+'.json')
   decision=read(file)['decision'] if file.exists() else workflow.chat(config,prompt,data,response_schema=schema)
   if decision.get('category') not in schema['properties']['category']['enum']:raise ValueError('Incomplete visual decision')
   save(file,{'decision':decision,'image_sha256':hashlib.sha256(data).hexdigest()});file.with_suffix('.png').write_bytes(data)
   return decision,str(file.with_suffix('.png'))
  controls=[]
  for color in ['gray','white']:
   a,ap=classify(Image.new('RGB',(256,256),color));b,bp=classify(Image.new('RGB',(512,512),color));controls.append({'color':color,'local':a,'context':b,'passed':a['category']=='uncertain' and b['category']=='uncertain'})
  save(out/'blank_controls.json',{'synthetic_functional_control_only':True,'rows':controls,'all_passed':all(r['passed'] for r in controls)})
  try:
   with urllib.request.urlopen(cfg['vlm_url'].removesuffix('/v1')+'/health',timeout=10) as f:save(out/'backend_receipt.json',json.load(f))
  except Exception as e:save(out/'backend_receipt.json',{'metadata_unavailable':type(e).__name__})
  verified=[]
  for case,pool in pools.items():
   for row in pool:
    image=images[case];mask=masks[row['key']];yy,xx=np.where(mask);x1,x2=int(xx.min()),int(xx.max())+1;y1,y2=int(yy.min()),int(yy.max())+1;extent=max(x2-x1,y2-y1)
    views=[]
    for pad in [max(12,int(.2*extent)),max(48,int(.8*extent))]:
     box=(max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad));views.append(classify(image.crop(box)))
    row={**row,'visual_decisions':[v[0] for v in views],'visual_evidence':[v[1] for v in views]}
    row['visual_confirmed']=all(v[0]['category']=='single_visible_pallet_base' and re.search(r'pallet|skid|托盘',v[0]['object'],re.I) and not re.search(r'outline|contour|yellow line|locator',v[0]['object'],re.I) for v in views)
    verified.append(row);save(out/'verification_progress.json',{'rows':verified});print(json.dumps({'case':case,'key':row['key'][:8],'two_view_confirmed':row['visual_confirmed']}),flush=True)
  totalbase=sum(int(m.sum()) for m in unions.values());countbase=sum(len(v) for v in base.values());selected={a:[] for a in ['B','C']}
  for arm in ['B','C']:
   union={c:u.copy() for c,u in unions.items()};picked={c:[] for c in unions};rejections=[]
   for row in sorted((r for r in verified if r['visual_confirmed']),key=lambda r:(-r['rank_'+arm],r['key'])):
    case=row['case'];mask=masks[row['key']];newarea=sum(int(v.sum()) for v in union.values())+int(((mask!=0)&(union[case]==0)).sum())
    if len(selected[arm])+countbase>=int(1.5*countbase) or newarea>1.25*totalbase:rejections.append({'key':row['key'],'reason':'whole candidate over frozen count/area budget'});continue
    if any(iou(mask,masks[r['key']])>=.5 or int((mask&masks[r['key']]).sum())/max(1,min(int(mask.sum()),int(masks[r['key']].sum())))>=.8 for r in picked[case]):continue
    picked[case].append(row);selected[arm].append(row);union[case]|=mask
   for case in unions:save(out/case/(arm+'_children.json'),[r['shape'] for r in picked[case]])
   save(out/(arm+'_selection.json'),{'selected':selected[arm],'rejected_whole_masks':rejections,'PAL_count':countbase+len(selected[arm]),'PAL_union_pixels':sum(int(v.sum()) for v in union.values()),'original_union_pixels':totalbase,'no_pixel_clipping':True})
  save(out/'completed.json',{'status':'complete','references_used_for_selection':False,'annotation_writes':0,'new_reference_inference':False,'cases':{c:{a:len(read(out/c/(a+'_children.json'))) for a in ['B','C']} for c in unions},'model_trace_records':len(trace),'whole_masks_only':True,'blank_controls_passed':all(r['passed'] for r in controls)})
 print(json.dumps(read(out/'completed.json')),flush=True)
if __name__=='__main__':main()
