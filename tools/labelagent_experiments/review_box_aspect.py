"""Paired caption geometry diagnosis; never rewrite original gate failures."""
import datetime,sys
from pathlib import Path
import numpy as np
from PIL import Image
import run_rectangular_review as base
from rectangular_views import rectangular_views
from aspect_views import aspect_view
from run_box_layers import OUT as EXP
ROOT=base.ROOT;OUT=EXP/'aspect_diagnostic';base.OUT=OUT


def prepare():
    if OUT.exists():raise FileExistsError('Preserve comparison')
    OUT.mkdir();prior=base.read(EXP/'semantic/frozen_plan.json');completed=base.read(EXP/'semantic/completed.json');audit=base.read(EXP/'support_area_audit.json');requests=[];protected={str(EXP/'semantic/completed.json'):base.sha(EXP/'semantic/completed.json')}
    for row in completed['rows']:
        if row['stage']!='neutral':continue
        p=Path(row['images'][0]);assert base.request_key(p)==row['request_sha256'];requests.append({'name':row['candidate']+'_'+row['name']+'_'+row['kind'],'kind':'original_'+row['kind'],'image':str(p),'sha256':base.sha(p),'request_sha256':row['request_sha256'],'reuse_response':row['response'],'reuse_response_sha256':base.sha(row['response']),'candidate':row['candidate'],'view':row['name']});protected[row['response']]=base.sha(row['response'])
    for key in prior['selected']:
        r=next(r for r in audit['candidates'] if r['key']==key);mfile=Path(r['folder'])/'complete_full_mask.npy';mask=np.load(mfile,allow_pickle=False);sourcefile=base.INC/'inputs'/(r['case']+'.png');source=Image.open(sourcefile).convert('RGB');protected.update({str(mfile):base.sha(mfile),str(sourcefile):base.sha(sourcefile)})
        for binding in rectangular_views(source,mask)[1]:
            for kind,value in [('candidate',None),('gray',127),('white',255)]:
                im,geometry=aspect_view(source if value is None else Image.new('RGB',source.size,(value,)*3),binding);name=f"{r['seed']}_{binding['view']}_{kind}";p=OUT/(name+'.png');im.save(p);requests.append({'name':name,'kind':'aspect_'+kind,'image':str(p),'sha256':base.sha(p),'request_sha256':base.request_key(p),'candidate':key,'view':binding['view'],'binding':binding,'geometry':geometry})
    base.save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':base.sha(Path(base.__file__)),'actual_driver_sha256':base.sha(__file__),'geometry_sha256':base.sha(Path(__file__).with_name('rectangular_views.py')),'aspect_policy_sha256':base.sha(Path(__file__).with_name('aspect_views.py')),'model_config_sha256':prior['model_config_sha256'],'requests':requests,'protected':protected,'max_new_requests':len({r['request_sha256'] for r in requests if 'reuse_response' not in r}),'budget':256,'original_failed_gates_not_overridden':True,'no_mask_changes':True,'diagnostic_only':True})
    print({'logical_views':len(requests),'max_unique_new':len({r['request_sha256'] for r in requests if 'reuse_response' not in r})},flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        p=base.read(OUT/'frozen_plan.json');assert base.sha(__file__)==p['actual_driver_sha256'] and base.sha(Path(__file__).with_name('aspect_views.py'))==p['aspect_policy_sha256'];base.run()
