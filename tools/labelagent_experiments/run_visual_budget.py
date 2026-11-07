"""Controlled single-image 256/1024-token comparison; defaults never change."""
import base64,datetime,hashlib,io,json,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from semantic_evidence_gate import CAPTION_PROMPT,CAPTION_SCHEMA
from semantic_subject_guard import compatible_pallet_subject
from target_identity_policy import centered_photos
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/labelagent_visual_budget_20261107';INC=ROOT/'outputs/labelagent_context_comparison_20261105'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def health():
 with urllib.request.urlopen('http://127.0.0.1:18868/health',timeout=15) as f:return json.load(f)
def payload(raw,budget):return {'model':'labelagent-vlm','temperature':0,'max_tokens':256,'image_token_budget':budget,'messages':[{'role':'user','content':[{'type':'text','text':CAPTION_PROMPT},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}}]}],'response_format':{'type':'json_schema','json_schema':{'name':'labelagent','schema':CAPTION_SCHEMA,'strict':True}}}
def prepare():
 if OUT.exists():raise FileExistsError('New output required')
 h=health();assert h['supports_single_image_budget_override'] and h['image_token_budget']==256
 OUT.mkdir();protected=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];protected={str(ROOT/p):s for p,s in protected.items()};inputs=[]
 for study in ('rgb_layers','opposed_layers'):
  for row in read(ROOT/f'outputs/labelagent_{study}_20261107/generation_completed.json')['rows']:
   if not row['valid']:continue
   maskfile=Path(row['folder'])/'complete_full_mask.npy';mask=np.load(maskfile,allow_pickle=False);sourcefile=INC/'inputs'/(row['case']+'.png');source=Image.open(sourcefile).convert('RGB');_,bindings=centered_photos(source,mask);protected[str(maskfile)]=sha(maskfile);protected[str(sourcefile)]=sha(sourcefile)
   for binding in bindings:
    box=binding['canvas_box'];clip=binding['source_box'];native=Image.new('RGB',(box[2]-box[0],box[3]-box[1]),(127,127,127));native.paste(source.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));image=native.resize((1024,1024),Image.Resampling.LANCZOS);name=study+'_'+row['key'][:8]+'_'+binding['view'];path=OUT/(name+'.png');image.save(path);inputs.append({'name':name,'kind':'candidate_diagnostic_not_human_truth','image':str(path),'sha256':sha(path),'binding':binding,'native_canvas_size':list(native.size),'resampling_does_not_add_native_detail':True})
 controls=ROOT/'outputs/labelagent_semantic_recovery_20261105/neutral_controls'
 for name in ('wood_pallet_base','raw_robot','gray','white'):
  source=controls/(name+'.png');protected[str(source)]=sha(source);path=OUT/('control_'+name+'.png');Image.open(source).convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS).save(path);inputs.append({'name':'control_'+name,'kind':'control','expected_PAL':name=='wood_pallet_base','image':str(path),'sha256':sha(path),'resampling_does_not_add_native_detail':True})
 requests=[]
 for row in inputs:
  for budget in (256,1024):
   body=payload(Path(row['image']).read_bytes(),budget);digest=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();requests.append({**row,'budget':budget,'request_sha256':digest})
 for a,b in zip(requests[::2],requests[1::2]):
  pa=payload(Path(a['image']).read_bytes(),a['budget']);pb=payload(Path(b['image']).read_bytes(),b['budget']);pa.pop('image_token_budget');pb.pop('image_token_budget');assert pa==pb
 cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json')
 for p in (cfg['reference_annotations'],cfg['reference_description']):protected[p]=sha(p)
 save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':sha(__file__),'health_before':{k:h.get(k) for k in ('model','quantization','image_token_budget','supports_single_image_budget_override','model_config_sha256')},'input_resolution':[1024,1024],'within_this_study_only_budget_changes':True,'not_byte_identical_to_old_512px_studies':True,'max_successful_requests':len(requests),'requests':requests,'protected':protected,'prompt':CAPTION_PROMPT,'schema':CAPTION_SCHEMA,'new_SAM_calls':0,'Agent_rounds':3,'acceptance_unchanged':True})
 print(json.dumps({'prepared_requests':len(requests),'paired_photos':len(inputs)}),flush=True)
def run():
 if (OUT/'completed.json').exists():raise FileExistsError('Already complete')
 plan=read(OUT/'frozen_plan.json');assert sha(__file__)==plan['driver_sha256'];rows=[];transport=read(OUT/'transmissions.json') if (OUT/'transmissions.json').exists() else []
 for row in plan['requests']:
  file=OUT/'responses'/(row['request_sha256']+'.json');raw=Path(row['image']).read_bytes();assert sha(row['image'])==row['sha256'];body=payload(raw,row['budget']);assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==row['request_sha256']
  if not file.exists():
   for _ in range(1500):
    if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
    req=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    try:
     with urllib.request.urlopen(req,timeout=300) as f:response=json.load(f)
     transport.append({'request':row['request_sha256'],'http':200});save(OUT/'transmissions.json',transport);save(file,response);break
    except urllib.error.HTTPError as e:
     transport.append({'request':row['request_sha256'],'http':e.code});save(OUT/'transmissions.json',transport)
     if e.code!=503:save(OUT/'failed_http.json',{'request':row,'http':e.code,'body':e.read().decode('utf-8','replace')});raise
     time.sleep(.2)
   else:raise RuntimeError('Shared model busy; preserve successful responses')
  response=read(file);stats=response['local_vlm'];assert stats['requested_image_token_budget']==row['budget'] and stats['effective_image_token_budget']==row['budget'] and stats['actual_image_token_counts']==[row['budget']]
  decision=json.loads(response['choices'][0]['message']['content']);assert set(decision)=={'object'} and isinstance(decision['object'],str);result={**row,'decision':decision,'PAL_subject_passed':compatible_pallet_subject(decision),'actual_tokens':stats['actual_image_token_counts'],'model_config_sha256':stats['model_config_sha256'],'response':str(file)};rows.append(result);save(OUT/'progress.json',rows);print(json.dumps({'name':row['name'],'budget':row['budget'],'decision':decision,'actual_tokens':stats['actual_image_token_counts']},ensure_ascii=False),flush=True)
 assert all(sha(p)==h for p,h in plan['protected'].items());assert len({r['model_config_sha256'] for r in rows})==1
 h=health();assert h['image_token_budget']==256
 control={str(budget):[{'name':r['name'],'passed':r['PAL_subject_passed']==r['expected_PAL']} for r in rows if r['budget']==budget and r['kind']=='control'] for budget in (256,1024)}
 save(OUT/'completed.json',{'status':'complete_single_factor_budget_comparison','successful_responses':len(rows),'rows':rows,'controls':control,'default_budget_after':h['image_token_budget'],'protected_unchanged':True,'new_masks':0,'new_Agent_runs':0,'adopted':False,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'scope':'development captions, no new instance success or human truth'})
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
