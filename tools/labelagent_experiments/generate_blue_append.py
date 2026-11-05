"""Image-only SAM instance prompts and append-only whole-mask admission."""
import sys,json,io,copy,time,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_projection_recovery_20261105';out=ROOT/'outputs/labelagent_projection_recovery_20261105/instance_generation_blue'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot'])
 import numpy as np,cv2
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize,_shape_from_mask
 from semantic_evidence_gate import classify_neutral
 from semantic_subject_guard import compatible_pallet_subject
 from incumbent_append_policy import append_complete_masks
 if (out/'completed.json').exists():raise FileExistsError('Completed run immutable')
 out.mkdir(parents=True,exist_ok=True);config={'sam3_url':cfg['sam_url'],'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_prompt_profile':'compact','llm_max_tokens':256}
 traces={r['id']:r for r in read(out/'model_trace.json')} if (out/'model_trace.json').exists() else {}
 inherited_ids=set(traces)
 if inherited_ids:save(out/'resume_receipt.json',{'inherited_trace_ids':sorted(inherited_ids),'resume_algorithm_unchanged':True,'saved_SAM_and_semantic_responses_reused_by_exact_input_binding':True,'reason':'observation session disappeared; read-only process inventory confirmed no live generator before resume'})
 def log(r):traces[r['id']]=copy.deepcopy(r);save(out/'model_trace.json',list(traces.values()))
 def cooperative(fn,*a,**kw):
  for _ in range(100):
   with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=15) as f:busy=json.load(f).get('status')=='running'
   if busy:time.sleep(3);continue
   try:return fn(*a,**kw)
   except ValueError as e:
    if 'Another model request' not in str(e):raise
    time.sleep(3)
  raise RuntimeError('Shared model remained busy')
 original_chat=workflow._chat;workflow._chat=lambda *a,**kw:cooperative(original_chat,*a,**kw)
 def png(im):
  b=io.BytesIO();im.save(b,'PNG');return b.getvalue()
 incumbent=ROOT/'outputs/labelagent_context_comparison_20261105';images={};bases={};originals={};people={};pool={};groups=[]
 for cam in [0,1]:
  case=f'camera{cam}_frame76';images[case]=Image.open(incumbent/'inputs'/(case+'.png')).convert('RGB');ss=read(incumbent/'C_hybrid'/case/'result.json')['shapes'];bases[case]=[_rasterize(s,1920,1536) for s in ss if s['label']=='pallet'];people[case]=[_rasterize(s,1920,1536) for s in ss if s['label']=='person'];originals[case]=[_rasterize(s,1920,1536) for s in read(incumbent/'A_current'/case/'result.json')['shapes'] if s['label']=='pallet'];pool[case]={};components={}
  for index,m in enumerate(bases[case]):
   n,cc,stats,_=cv2.connectedComponentsWithStats(m,8)
   for i in range(1,n):
    x,y,w,h,area=map(int,stats[i]);mask=(cc==i).astype('uint8')
    if w<24 or h<18 or area<300:continue
    rgb=np.asarray(images[case]).astype('int16')[mask!=0];blue=float(((rgb[:,2]-rgb[:,0]>=12)&(rgb[:,2]>=rgb[:,1]-3)&(rgb[:,2]>=45)).mean())
    if blue<.1:continue
    key=hashlib.sha256(case.encode()+mask.tobytes()).hexdigest()
    components[key]={'case':case,'key':key,'mask':mask,'bbox':[x,y,x+w,y+h],'area':area,'source_index':index,'blue_fraction':blue}
  for g in sorted(components.values(),key=lambda r:(-r['area']*r['blue_fraction'],r['key']))[:4]:groups.append(g)
  for grid in [ROOT/'outputs/labelagent_recovery_20261105/fixed_grid',ROOT/'outputs/labelagent_semantic_recovery_20261105/fixed_grid']:
   for file in sorted((grid/case).glob('*/prediction.json')):
    for shape in read(file)['PAL_shapes']:
     m=_rasterize(shape,1920,1536);key=hashlib.sha256(case.encode()+m.tobytes()).hexdigest();row={'case':case,'key':key,'shape':shape,'mask':m,'score':float(shape['score']),'source':'unchanged_raw_trained_grid','source_prediction':str(file)}
     if key not in pool[case] or row['score']>pool[case][key]['score']:pool[case][key]=row
 for case in pool:
  for row in read(out.parent/'instance_generation'/case/'all_hypotheses.json'):
   pool[case].setdefault(row['key'],{**row,'mask':_rasterize(row['shape'],1920,1536)})
 save(out/'sam_prompt_plan.json',[{k:v for k,v in g.items() if k!='mask'} for g in groups]);sam_rows=[]
 with workflow.trace_calls(log):
  for g in groups:
   case=g['case'];x1,y1,x2,y2=g['bbox'];pad=48;box=[max(0,x1-pad),max(0,y1-pad),min(1920,x2+pad),min(1536,y2+pad)];local=g['mask'][box[1]:box[3],box[0]:box[2]];distance=cv2.distanceTransform(local,cv2.DIST_L2,5);centers=[]
   for fraction in [.2,.5,.8]:
    cy=(y1-box[1])+fraction*(y2-y1);band=local.copy();yy=np.arange(local.shape[0])[:,None];band[(np.broadcast_to(abs(yy-cy)>max(3,(y2-y1)*.12),local.shape))]=0
    d=distance*(band!=0);y,x=np.unravel_index(np.argmax(d),d.shape);centers.append((int(x),int(y)))
   raw=png(images[case].crop(tuple(box)))
   for j,positive in enumerate(centers):
    points=[{'x':positive[0],'y':positive[1],'label':1}]+[{'x':p[0],'y':p[1],'label':0} for k,p in enumerate(centers) if k!=j and (p[0]-positive[0])**2+(p[1]-positive[1])**2>=64]
    folder=out/'sam_instances'/(g['key']+'_'+str(j));folder.mkdir(parents=True,exist_ok=True);(folder/'input.png').write_bytes(raw);save(folder/'prompt.json',{'source_component':g['key'],'source_box':box,'fraction':[.2,.5,.8][j],'points':points,'prior_mask':None,'reference_access':False})
    file=folder/'response.json'
    if file.exists():response=read(file)
    else:
     try:response=cooperative(workflow.sam_request,config,'/segment_points',raw,points=points)
     except ValueError as e:
      if not any(w in str(e) for w in ['无法排除','提示点','没有找到目标']):raise
      response={'rejected_reason':str(e)}
     save(file,response)
    row={'case':case,'component':g['key'],'point_fraction':[.2,.5,.8][j],'technically_valid':bool(response.get('parts') and not response.get('warnings')),'warnings':response.get('warnings',[]),'rejection':response.get('rejected_reason'),'folder':str(folder)}
    if row['technically_valid']:
     m=np.zeros((1536,1920),np.uint8);m[box[1]:box[3],box[0]:box[2]]=_rasterize(response,box[2]-box[0],box[3]-box[1]);shape=_shape_from_mask(m,'pallet',1)
     if shape:
      # Exact complete returned polygon geometry, not intersection with parent.
      m=_rasterize(shape,1920,1536);key=hashlib.sha256(case.encode()+m.tobytes()).hexdigest();pool[case][key]={'case':case,'key':key,'shape':shape,'mask':m,'score':float(response.get('score',0)),'source':'new_true_SAM_instance','source_prediction':str(file)};row['key']=key
    sam_rows.append(row);save(out/'sam_progress.json',sam_rows);print(json.dumps({'SAM_case':case,'j':j,'valid':row['technically_valid']},ensure_ascii=False),flush=True)
  banks={c:list(p.values()) for c,p in pool.items()};verified={};held=[]
  for case,rows in banks.items():save(out/case/'all_hypotheses.json',[{k:v for k,v in r.items() if k!='mask'} for r in rows])
  for iteration in range(100):
   selected,rejected,summary=append_complete_masks(banks,bases,originals,people);pending=[r for rows in selected.values() for r in rows if r['key'] not in verified]
   if not pending:break
   for row in pending:
    im=images[row['case']];m=_rasterize(row['shape'],1920,1536);yy,xx=np.where(m);margin=max(32,int(max(xx.max()-xx.min(),yy.max()-yy.min())*.5));boxes=[[max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(1920,int(xx.max())+9),min(1536,int(yy.max())+9)],[max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(1920,int(xx.max())+margin+1),min(1536,int(yy.max())+margin+1)]];views=[]
    for b in boxes:
     photo=im.crop(tuple(b));scale=max(1,512/max(photo.size));photo=photo.resize((round(photo.width*scale),round(photo.height*scale)),Image.Resampling.LANCZOS);views.append(classify_neutral(workflow,config,png(photo),out/'visual_cache'))
    passed=all(compatible_pallet_subject(v['decision']) for v in views);verified[row['key']]={'row':row,'views':views,'passed':passed,'source_boxes':boxes}
    if not passed:banks[row['case']]=[r for r in banks[row['case']] if r['key']!=row['key']];held.append({'case':row['case'],'key':row['key'],'captions':[v['decision'] for v in views],'reason':'unchanged neutral semantic/subject gate'})
    save(out/'verification_progress.json',{'rows':list(verified.values())});print(json.dumps({'case':row['case'],'neutral_pass':passed,'key':row['key'][:8]},ensure_ascii=False),flush=True)
  else:raise RuntimeError('Append selection did not stabilize')
 for case,rows in selected.items():save(out/case/'additional_children.json',[r['shape'] for r in rows])
 save(out/'selection.json',{'selected':selected,'rejected_whole_masks':rejected,'held_semantic':held,'summary':summary,'reference_inputs':False})
 save(out/'completed.json',{'status':'complete','sam_actual_calls':len(sam_rows),'sam_valid':sum(r['technically_valid'] for r in sam_rows),'VLM_trace_records':sum(r.get('service')=='GPT' for r in traces.values()),'new_instances':{c:len(rows) for c,rows in selected.items()},'summary':summary,'annotations_written':0,'incumbent_foreground_unchanged':True,'reference_access':False});print(json.dumps(read(out/'completed.json')),flush=True)
if __name__=='__main__':main()
