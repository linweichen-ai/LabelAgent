"""Prepare the entire RGB cohort before bounded managed SAM inference."""
import base64,datetime,hashlib,io,json,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from rgb_layer_policy import propose
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/labelagent_rgb_layers_20261107';INC=ROOT/'outputs/labelagent_context_comparison_20261105'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepare():
 if OUT.exists():raise FileExistsError('Preserve frozen experiment')
 OUT.mkdir();cohort=read(ROOT/'outputs/labelagent_point_ids_20261107/planning/cohort.json');prepared=[];diagnostics=[];protected={}
 cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json')
 for p in [Path(cfg['reference_annotations']),Path(cfg['reference_description'])]+list(INC.glob('C_hybrid/*/result.json')):protected[str(p)]=sha(p)
 for seed in cohort:
  source=INC/'inputs'/(seed['case']+'.png');seedfile=Path(seed['folder'])/'seed_mask.png';protected[str(source)]=sha(source);protected[str(seedfile)]=sha(seedfile)
  box=seed['binding']['source_box'];im=Image.open(source).convert('RGB').crop(tuple(box));mask=(np.asarray(Image.open(seedfile))>0)[box[1]:box[3],box[0]:box[2]];a=seed['binding']['anchor_source'];plans,diag=propose(np.asarray(im),mask,(a[0]-box[0],a[1]-box[1]));diagnostics.append({'seed':seed['index'],**diag})
  folder=OUT/f"seed{seed['index']:02d}";folder.mkdir();im.save(folder/'input.png');Image.fromarray(mask.astype('uint8')*255).save(folder/'seed.png')
  for plan in plans:
   key=hashlib.sha256((sha(folder/'input.png')+json.dumps(plan['points'],sort_keys=True)).encode()).hexdigest()
   prepared.append({**plan,'seed':seed['index'],'case':seed['case'],'box':box,'key':key,'input':str(folder/'input.png')})
 save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'proposals':prepared,'diagnostics':diagnostics,'policy_sha256':sha(Path(__file__).with_name('rgb_layer_policy.py')),'driver_sha256':sha(__file__),'spec_sha256':sha(ROOT/'task_workspaces/labelagent_raw_mask_audit_20261107/raw_capture_next_experiment.json'),'protected':protected,'max_requests':16,'raw_capture_required':True,'no_reference_inputs':True,'negative_points':0})
 print(json.dumps({'prepared':len(prepared),'unique':len({p['key'] for p in prepared})}),flush=True)
def run():
 if (OUT/'generation_completed.json').exists():raise FileExistsError('Already complete')
 plan=read(OUT/'frozen_plan.json');assert sha(Path(__file__).with_name('rgb_layer_policy.py'))==plan['policy_sha256'];rows=[]
 for p in plan['proposals']:
  folder=OUT/'captures'/p['key'];folder.mkdir(parents=True,exist_ok=True);file=folder/'response.json'
  if not file.exists():
   raw=Path(p['input']).read_bytes();body=json.dumps({'image':base64.b64encode(raw).decode(),'points':p['points'],'include_raw_capture':True}).encode()
   for _ in range(300):
    if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
    req=urllib.request.Request('http://127.0.0.1:8766/segment_points',data=body,headers={'Content-Type':'application/json'})
    try:
     with urllib.request.urlopen(req,timeout=600) as f:response=json.load(f)
     break
    except urllib.error.HTTPError as e:
     if e.code==503:time.sleep(1);continue
     response=json.loads(e.read());break
   else:raise RuntimeError('Manager busy; resume missing only')
   save(file,response)
  response=read(file);capture=response['raw_capture'];metadata=capture['metadata'];npz=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(npz);save(folder/'metadata.json',metadata)
  row={**p,'folder':str(folder),'valid':False,'status':metadata['status']}
  if metadata.get('raw_capture_complete') and metadata.get('selected_index_observed'):
   arrays=np.load(io.BytesIO(npz),allow_pickle=False);raw=arrays[f"{metadata['selected_prediction_call']}_masks"][metadata['selected_index']].squeeze()
   if np.isin(raw,[0,1]).all():
    m=(raw!=0).astype('uint8');full=np.zeros((1536,1920),np.uint8);x1,y1,x2,y2=p['box'];assert m.shape==(y2-y1,x2-x1);full[y1:y2,x1:x2]=m
    assert int(m.sum())==int(full.sum());np.save(folder/'complete_full_mask.npy',full);Image.fromarray(m*255).save(folder/'raw.png')
    border=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any());row.update(raw_pixels=int(m.sum()),touches_border=border,valid=metadata['status']=='segment_returned' and not border and not response.get('warnings'),mask_sha256=hashlib.sha256(full.tobytes()).hexdigest())
  rows.append(row);save(OUT/'progress.json',rows);print(json.dumps({'seed':p['seed'],'rank':p['rank'],'valid':row['valid'],'pixels':row.get('raw_pixels')}),flush=True)
 assert all(sha(p)==s for p,s in plan['protected'].items())
 unique={p['key'] for p in rows};save(OUT/'generation_completed.json',{'status':'complete','proposals':len(rows),'unique_real_SAM_calls':len(unique),'valid_complete_outputs':len({r['key'] for r in rows if r['valid']}),'rows':rows,'protected_unchanged':True,'annotation_writes':0,'Agent_runs':0})
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
