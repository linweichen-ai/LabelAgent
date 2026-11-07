"""New aspect policy; exact old caption requests are reused without rerunning."""
import datetime,hashlib,sys
from pathlib import Path
from PIL import Image
from aspect_views import aspect_view
import run_rectangular_review as base
ROOT=base.ROOT;OLD=base.OUT;OUT=ROOT/'outputs/labelagent_aspect_review_20261107';base.OUT=OUT
def prepare():
 if OUT.exists():raise FileExistsError('Immutable')
 h=base.health();old=base.read(OLD/'completed.json');previous=base.read(OLD/'frozen_plan.json');assert h['model_config_sha256']==previous['model_config_sha256'] and h['image_token_budget']==256
 OUT.mkdir();requests=[];protected=dict(previous['protected']);cached={r['request_sha256']:r for r in old['rows']}
 for r in old['rows']:
  if r['kind'] not in ['rect_candidate','old_control']:continue
  row={k:r.get(k) for k in ['name','expected_PAL','image','sha256','request_sha256','binding']};row['kind']='square_letterbox_baseline' if r['kind']=='rect_candidate' else 'old_control';requests.append(row)
 for r in old['rows']:
  if r['kind']!='rect_candidate':continue
  sourcefile=base.INC/'inputs'/'camera0_frame76.png';source=Image.open(sourcefile).convert('RGB');binding=r['binding'];image,geometry=aspect_view(source,binding);p=OUT/(r['name']+'_aspect.png');image.save(p);requests.append({'name':r['name'],'kind':'aspect_candidate','expected_PAL':None,'image':str(p),'sha256':base.sha(p),'request_sha256':base.request_key(p),'binding':binding,'aspect_geometry':geometry})
  assert geometry['candidate_area_fraction_estimate']>geometry['square_candidate_fraction']
  for name,value in [('gray',127),('white',255)]:
   p=OUT/(r['name']+'_'+name+'.png');aspect_view(Image.new('RGB',source.size,(value,)*3),binding)[0].save(p);requests.append({'name':r['name']+'_'+name,'kind':'matched_'+name,'expected_PAL':False,'image':str(p),'sha256':base.sha(p),'request_sha256':base.request_key(p),'binding':binding,'aspect_geometry':geometry})
 for row in requests:
  if row['request_sha256'] in cached:
   prev=cached[row['request_sha256']];assert base.request_key(row['image'])==prev['request_sha256'];row['reuse_response']=prev['response'];row['reuse_response_sha256']=base.sha(prev['response']);protected[prev['response']]=base.sha(prev['response'])
 plan={'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':base.sha(Path(base.__file__)),'geometry_sha256':base.sha(Path(__file__).with_name('rectangular_views.py')),'actual_driver_sha256':base.sha(__file__),'aspect_policy_sha256':base.sha(Path(__file__).with_name('aspect_views.py')),'subject_guard_sha256':base.sha(Path(__file__).with_name('semantic_subject_guard.py')),'model_config_sha256':h['model_config_sha256'],'requests':requests,'protected':protected,'budget':256,'new_masks':0,'all_final_gates_unchanged':True,'max_new_requests':len({r['request_sha256'] for r in requests if 'reuse_response' not in r}),'not_new_native_detail':True}
 base.save(OUT/'frozen_plan.json',plan);print({'rows':len(requests),'max_new_requests':plan['max_new_requests']},flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 else:
  plan=base.read(OUT/'frozen_plan.json');assert base.sha(__file__)==plan['actual_driver_sha256'] and base.sha(Path(__file__).with_name('aspect_views.py'))==plan['aspect_policy_sha256'] and base.sha(Path(__file__).with_name('semantic_subject_guard.py'))==plan['subject_guard_sha256'];base.run()
