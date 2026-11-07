"""Complete-bbox center vs deepest-point center with exact cached baselines."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import run_rectangular_review as base
from bbox_center_views import views
from aspect_views import aspect_view
from run_box_layers import OUT as EXP
ROOT=base.ROOT;OUT=EXP/'center_diagnostic';base.OUT=OUT


def prepare():
    if OUT.exists():raise FileExistsError('Preserve comparison')
    OUT.mkdir();old=base.read(EXP/'aspect_diagnostic/completed.json');prior=base.read(EXP/'semantic/frozen_plan.json');audit=base.read(EXP/'support_area_audit.json');protected={};requests=[]
    for r in old['rows']:
        row={k:v for k,v in r.items() if k not in ['decision','PAL_passed','actual_image_tokens','response','reused']};row.update(reuse_response=r['response'],reuse_response_sha256=base.sha(r['response']));requests.append(row);protected[r['response']]=base.sha(r['response'])
    for key in prior['selected']:
        r=next(x for x in audit['candidates'] if x['key']==key);mf=Path(r['folder'])/'complete_full_mask.npy';sourcefile=base.INC/'inputs'/(r['case']+'.png');mask=np.load(mf,allow_pickle=False);source=Image.open(sourcefile).convert('RGB');protected.update({str(mf):base.sha(mf),str(sourcefile):base.sha(sourcefile)})
        for b in views(source,mask)[1]:
            for kind,value in [('candidate',None),('gray',127),('white',255)]:
                im,geometry=aspect_view(source if value is None else Image.new('RGB',source.size,(value,)*3),b);name=f"{r['seed']}_{b['view']}_{kind}";p=OUT/(name+'.png');im.save(p);requests.append({'name':name,'kind':'bbox_'+kind,'image':str(p),'sha256':base.sha(p),'request_sha256':base.request_key(p),'candidate':key,'view':b['view'],'binding':b,'geometry':geometry})
    base.save(OUT/'frozen_plan.json',{'driver_sha256':base.sha(Path(base.__file__)),'actual_driver_sha256':base.sha(__file__),'geometry_sha256':base.sha(Path(__file__).with_name('rectangular_views.py')),'center_policy_sha256':base.sha(Path(__file__).with_name('bbox_center_views.py')),'aspect_policy_sha256':base.sha(Path(__file__).with_name('aspect_views.py')),'model_config_sha256':prior['model_config_sha256'],'requests':requests,'protected':protected,'budget':256,'diagnostic_only':True,'original_failures_preserved':True,'no_mask_changes':True,'max_new_requests':len({r['request_sha256'] for r in requests if 'reuse_response' not in r})})
    print({'max_new_requests':len({r['request_sha256'] for r in requests if 'reuse_response' not in r})},flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        p=base.read(OUT/'frozen_plan.json');assert base.sha(__file__)==p['actual_driver_sha256'] and base.sha(Path(__file__).with_name('bbox_center_views.py'))==p['center_policy_sha256'];base.run()
