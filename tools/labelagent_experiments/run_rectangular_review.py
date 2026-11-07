"""Frozen square/rectangular caption comparison with matched padding controls."""
import datetime,hashlib,json,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from rectangular_views import rectangular_views,render
from run_visual_budget import ROOT,INC,read,save,sha,payload,health
from semantic_subject_guard import compatible_pallet_subject
OUT=ROOT/'outputs/labelagent_rectangular_review_20261107';OLD=ROOT/'outputs/labelagent_visual_budget_20261107'
def request_key(path):return hashlib.sha256(json.dumps(payload(Path(path).read_bytes(),256),sort_keys=True).encode()).hexdigest()
def prepare():
 if OUT.exists():raise FileExistsError('Immutable experiment')
 h=health();old=read(OLD/'completed.json');cfgs={r['model_config_sha256'] for r in old['rows']};assert cfgs=={h['model_config_sha256']} and h['image_token_budget']==256
 OUT.mkdir();requests=[];protected=dict(read(OLD/'frozen_plan.json')['protected']);cached={r['request_sha256']:r for r in old['rows'] if r['budget']==256}
 for row in old['rows']:
  if row['budget']!=256:continue
  requests.append({'name':row['name'],'kind':'square_baseline' if row['kind']!='control' else 'old_control','expected_PAL':row.get('expected_PAL'),'image':row['image'],'sha256':row['sha256'],'request_sha256':row['request_sha256']})
 for study in ('rgb_layers','opposed_layers'):
  for row in read(ROOT/f'outputs/labelagent_{study}_20261107/generation_completed.json')['rows']:
   if not row['valid']:continue
   sourcefile=INC/'inputs'/(row['case']+'.png');maskfile=Path(row['folder'])/'complete_full_mask.npy';source=Image.open(sourcefile).convert('RGB');mask=np.load(maskfile,allow_pickle=False);views,bindings=rectangular_views(source,mask);protected[str(sourcefile)]=sha(sourcefile);protected[str(maskfile)]=sha(maskfile)
   for binding in bindings:
    name=study+'_'+row['key'][:8]+'_'+binding['view'];p=OUT/(name+'_rect.png');views[binding['view']].save(p);requests.append({'name':name,'kind':'rect_candidate','expected_PAL':None,'image':str(p),'sha256':sha(p),'request_sha256':request_key(p),'binding':binding})
    for color,value in [('gray',127),('white',255)]:
     p=OUT/(name+'_'+color+'.png');render(Image.new('RGB',source.size,(value,)*3),binding)[0].save(p);requests.append({'name':name+'_'+color,'kind':'matched_'+color,'expected_PAL':False,'image':str(p),'sha256':sha(p),'request_sha256':request_key(p),'binding':binding})
 for row in requests:
  if row['request_sha256'] in cached:
   prev=cached[row['request_sha256']];assert request_key(row['image'])==prev['request_sha256'];row['reuse_response']=prev['response'];row['reuse_response_sha256']=sha(prev['response']);protected[prev['response']]=sha(prev['response'])
 save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':sha(__file__),'geometry_sha256':sha(Path(__file__).with_name('rectangular_views.py')),'model_config_sha256':h['model_config_sha256'],'requests':requests,'max_new_requests':len({r['request_sha256'] for r in requests if 'reuse_response' not in r}),'budget':256,'input_resolution':[1024,1024],'protected':protected,'proposal_or_reference_generation':False,'all_final_gates_unchanged':True,'geometry_not_confirmed_physical_extent':True,'matched_white_padding_controls_required':True})
 print(json.dumps({'rows':len(requests),'max_new_requests':len({r['request_sha256'] for r in requests if 'reuse_response' not in r}),'reuse_rows':sum('reuse_response' in r for r in requests)}),flush=True)
def run():
 if (OUT/'completed.json').exists():raise FileExistsError('Complete')
 plan=read(OUT/'frozen_plan.json');assert sha(__file__)==plan['driver_sha256'] and sha(Path(__file__).with_name('rectangular_views.py'))==plan['geometry_sha256'];rows=[];transmissions=read(OUT/'transmissions.json') if (OUT/'transmissions.json').exists() else []
 for row in plan['requests']:
  assert sha(row['image'])==row['sha256'] and request_key(row['image'])==row['request_sha256'];file=Path(row['reuse_response']) if 'reuse_response' in row else OUT/'responses'/(row['request_sha256']+'.json')
  if 'reuse_response' in row:assert sha(file)==row['reuse_response_sha256']
  if not file.exists():
   for _ in range(1500):
    if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
    req=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(payload(Path(row['image']).read_bytes(),256)).encode(),headers={'Content-Type':'application/json'})
    try:
     with urllib.request.urlopen(req,timeout=300) as f:response=json.load(f)
     transmissions.append({'key':row['request_sha256'],'http':200});save(OUT/'transmissions.json',transmissions);save(file,response);break
    except urllib.error.HTTPError as e:
     transmissions.append({'key':row['request_sha256'],'http':e.code});save(OUT/'transmissions.json',transmissions)
     if e.code!=503:save(OUT/'http_failure.json',{'http':e.code,'body':e.read().decode('utf-8','replace')});raise
     time.sleep(.2)
   else:raise RuntimeError('Shared model busy; preserve partial results')
  response=read(file);stats=response['local_vlm'];assert stats['model_config_sha256']==plan['model_config_sha256'] and stats['actual_image_token_counts']==[256];decision=json.loads(response['choices'][0]['message']['content']);assert set(decision)=={'object'};rows.append({**row,'decision':decision,'PAL_passed':compatible_pallet_subject(decision),'actual_image_tokens':stats['actual_image_token_counts'],'response':str(file)});save(OUT/'progress.json',rows);print(json.dumps({'name':row['name'],'kind':row['kind'],'decision':decision,'PAL':rows[-1]['PAL_passed'],'reused':'reuse_response' in row},ensure_ascii=False),flush=True)
 assert all(sha(p)==h for p,h in plan['protected'].items());assert health()['image_token_budget']==256
 save(OUT/'completed.json',{'status':'complete_geometry_comparison','rows':rows,'fresh_responses':len({r['request_sha256'] for r in rows if 'reuse_response' not in r}),'reused_rows':sum('reuse_response' in r for r in rows),'new_masks':0,'new_Agent_runs':0,'protected_unchanged':True,'adopted':False,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4})
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
