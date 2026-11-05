"""Reference-free whole-mask selection with neutral, paired photo evidence."""
import sys,json,copy,hashlib,io,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_semantic_recovery_20261105';out=ROOT/'outputs/labelagent_semantic_recovery_20261105/instance_generation_contextual'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot'])
 import numpy as np,cv2
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize
 from preserved_model_policy import select_complete_masks
 from semantic_evidence_gate import classify_neutral
 out.mkdir(parents=True,exist_ok=True)
 if (out/'completed.json').exists():raise FileExistsError('Completed bank is immutable')
 config={'sam3_url':cfg['sam_url'],'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_prompt_profile':'compact','llm_max_tokens':256}
 original=workflow._chat
 def cooperative(*a,**kw):
  for _ in range(100):
   with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=10) as f:busy=json.load(f).get('status')=='running'
   if busy:time.sleep(3);continue
   try:return original(*a,**kw)
   except ValueError as e:
    if 'Another model request' not in str(e):raise
    time.sleep(3)
  raise RuntimeError('Shared service busy')
 workflow._chat=cooperative;traces={}
 def log(r):traces[r['id']]=copy.deepcopy(r);save(out/'model_trace.json',list(traces.values()))
 pool={};base={};people={};images={};verified={};rejected_visual=[]
 for cam in [0,1]:
  case=f'camera{cam}_frame76';old=ROOT/'outputs/labelagent_preserved_model_comparison_20261105';ss=read(old/'A_current'/case/'result.json')['shapes'];base[case]=[_rasterize(s,1920,1536) for s in ss if s['label']=='pallet'];people[case]=[_rasterize(s,1920,1536) for s in ss if s['label']=='person'];images[case]=Image.open(old/'inputs'/(case+'.png')).convert('RGB');proposals={}
  for grid in [ROOT/'outputs/labelagent_recovery_20261105/fixed_grid',out.parent/'fixed_grid']:
   for file in sorted((grid/case).glob('*/prediction.json')):
    for shape in read(file)['PAL_shapes']:
     m=_rasterize(shape,1920,1536);key=hashlib.sha256(case.encode()+m.tobytes()).hexdigest()
     row={'case':case,'key':key,'shape':shape,'mask':m,'trained_score':float(shape['score']),'source_prediction':str(file),'mask_sha256':hashlib.sha256(m.tobytes()).hexdigest(),'review_status':'unconfirmed candidate; neutral image evidence is not human truth'}
     if key not in proposals or row['trained_score']>proposals[key]['trained_score']:proposals[key]=row
  pool[case]=list(proposals.values());save(out/case/'all_raw_model_hypotheses.json',[{k:v for k,v in r.items() if k!='mask'} for r in pool[case]])
 def png(im):
  b=io.BytesIO();im.save(b,'PNG');return b.getvalue()
 with workflow.trace_calls(log):
  for iteration in range(100):
   selected,rejected,summary=select_complete_masks(pool,base,people);pending=[r for rows in selected.values() for r in rows if r['key'] not in verified]
   if not pending:break
   for row in pending:
    image=images[row['case']];mask=_rasterize(row['shape'],1920,1536);yy,xx=np.where(mask);pad=24;box=[max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(1920,int(xx.max())+pad+1),min(1536,int(yy.max())+pad+1)]
    raw=np.array(image.crop(box));local=mask[box[1]:box[3],box[0]:box[2]];isolated=np.full_like(raw,128);isolated[local!=0]=raw[local!=0]
    # No overlay nor target class list appears in either visual observation.
    tight=[max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(1920,int(xx.max())+9),min(1536,int(yy.max())+9)]
    margin=max(32,int(max(xx.max()-xx.min(),yy.max()-yy.min())*.5))
    wider=[max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(1920,int(xx.max())+margin+1),min(1536,int(yy.max())+margin+1)]
    photos=[image.crop(tuple(b)) for b in [tight,wider]]
    for i,photo in enumerate(photos):
     # Pixel interpolation is declared; this adds no source detail. It avoids
     # a very small input silently receiving only the encoder's 64-token min.
     scale=max(1,512/max(photo.size));photos[i]=photo.resize((round(photo.width*scale),round(photo.height*scale)),Image.Resampling.LANCZOS)
    views=[classify_neutral(workflow,config,png(im),out/'visual_cache') for im in photos]
    passed=all(v['pallet_evidence'] for v in views);verified[row['key']]={'row':row,'views':views,'passed':passed,'crop':box,'actual_source_crops':[tight,wider],'interpolation':'Lanczos to minlongedge512; no new source detail'}
    if not passed:
     pool[row['case']]=[r for r in pool[row['case']] if r['key']!=row['key']];rejected_visual.append({'key':row['key'],'case':row['case'],'decisions':[v['decision'] for v in views],'reason':'Neutral photo observation does not independently name pallet in both views'})
    print(json.dumps({'key':row['key'][:8],'case':row['case'],'neutral_evidence':passed,'captions':[v['decision']['object'] for v in views]},ensure_ascii=False),flush=True)
   save(out/'verification_progress.json',{'rows':list(verified.values())})
  else:raise RuntimeError('Selection did not stabilize')
 for case in pool:
  save(out/case/'C_children.json',[r['shape'] for r in selected[case]]);save(out/case/'B_children.json',[])
 controls=read(out.parent/'neutral_controls/completed.json');save(out/'blank_controls.json',{'all_passed':controls['negative_controls_passed'],'positive_diagnostic_recall':controls['positive_diagnostic_recall'],'positive_diagnostics':2,'source':'../neutral_controls/completed.json','functional_not_accuracy':True})
 save(out/'C_selection.json',{'selected':selected,'rejected_whole_masks':rejected,'rejected_visual':rejected_visual,'summary':summary,'no_pixel_clipping':True})
 save(out/'generation_protocol.json',{'same_checkpoint':True,'raw_fixed_grids':[32,84],'total_real_forwards_source':116,'reference_inputs':False,'semantic_gate':'neutral center-object caption on unmarked tight and natural-context photo crops; minlongedge512 interpolation disclosed, no target class list','class_word_positive_evidence':'pallet/skid required, blank/uncertain/locator rejected','whole_masks_only':True,'ranking':'original trained score times soft novelty','budgets':'same per-camera1.5 count and1.25 union; original predicted-person radius3 guard','SAM_boundaries':'advisory from previous failure trials; new trained masks not overwritten','human_truth':False})
 save(out/'completed.json',{'status':'complete','references_used_for_selection':False,'annotation_writes':0,'whole_masks_only':True,'model_trace_records':len(traces),'fresh_model_requests':len(traces),'cases':{c:{'B':0,'C':len(selected[c])} for c in pool},'blank_controls_passed':controls['negative_controls_passed'],'raw_model_instances_preserved':True})
 print(json.dumps(read(out/'completed.json')),flush=True)
if __name__=='__main__':main()
