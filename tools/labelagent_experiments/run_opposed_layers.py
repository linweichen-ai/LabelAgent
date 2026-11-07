"""New image-only proposal policy; opposite-side negatives are not mask edits."""
import sys,json,hashlib,datetime
from pathlib import Path
import numpy as np,cv2
from PIL import Image
import run_rgb_layers as base
from rgb_layer_policy import propose
ROOT=base.ROOT;OUT=ROOT/'outputs/labelagent_opposed_layers_20261107';base.OUT=OUT
def prepare():
 if OUT.exists():raise FileExistsError('Preserve immutable experiment')
 OUT.mkdir();cohort=base.read(ROOT/'outputs/labelagent_point_ids_20261107/planning/cohort.json');plans=[];diagnostics=[];protected={}
 cfg=base.read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json')
 for p in [Path(cfg['reference_annotations']),Path(cfg['reference_description'])]+list(base.INC.glob('*/*/result.json')):protected[str(p)]=base.sha(p)
 for seed in cohort:
  source=base.INC/'inputs'/(seed['case']+'.png');maskfile=Path(seed['folder'])/'seed_mask.png';protected[str(source)]=base.sha(source);protected[str(maskfile)]=base.sha(maskfile)
  box=seed['binding']['source_box'];image=Image.open(source).convert('RGB').crop(tuple(box));mask=(np.asarray(Image.open(maskfile))>0)[box[1]:box[3],box[0]:box[2]];ax,ay=seed['binding']['anchor_source'];anchor=(ax-box[0],ay-box[1]);original,diag=propose(np.asarray(image),mask,anchor)
  pairs=sorted(original+diag['rejected'],key=lambda r:r['rank']);folder=OUT/f"seed{seed['index']:02d}";folder.mkdir();image.save(folder/'input.png');Image.fromarray(mask.astype('uint8')*255).save(folder/'seed.png');rejected=[]
  if pairs:
   a,b=pairs[0]['lines'];lo=max(a[0],b[0],anchor[0]-max(24,min(96,round(.25*image.width))));hi=min(a[2],b[2],anchor[0]+max(24,min(96,round(.25*image.width))))
   y=lambda line,x:line[1]+(x-line[0])*(line[3]-line[1])/(line[2]-line[0]);dilated=cv2.dilate(mask.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
   for side,t in [('upper',.25),('lower',.75)]:
    points=[]
    for label,fraction in [(1,t),(0,1-t)]:
     for fractionx in (.25,.75):
      x=lo+(hi-lo)*fractionx;ys=sorted([y(a,x),y(b,x)]);points.append({'x':round(x),'y':round(ys[0]+fraction*(ys[1]-ys[0])),'label':label})
    valid=all(0<=p['x']<image.width and 0<=p['y']<image.height for p in points) and all((p['x']-q['x'])**2+(p['y']-q['y'])**2>=9 for k,p in enumerate(points) for q in points[:k])
    valid=valid and any(dilated[p['y'],p['x']] for p in points if p['label']==1)
    row={'rank':0 if side=='upper' else 1,'side':side,'lines':[a,b],'points':points,'valid':bool(valid),'seed':seed['index'],'case':seed['case'],'box':box,'input':str(folder/'input.png'),'opposite_points_inside_search_seed':sum(bool(mask[p['y'],p['x']]) for p in points if p['label']==0),'no_reference_inputs':True}
    row['key']=hashlib.sha256((base.sha(row['input'])+json.dumps(points,sort_keys=True)).encode()).hexdigest();(plans if valid else rejected).append(row)
  diagnostics.append({'seed':seed['index'],'detected_pairs':diag['pairs'],'selected_pair_rank':pairs[0]['rank'] if pairs else None,'rejected':rejected})
 plan={'at':datetime.datetime.now().astimezone().isoformat(),'proposals':plans,'diagnostics':diagnostics,'policy_sha256':base.sha(Path(__file__).with_name('rgb_layer_policy.py')),'opposed_policy_sha256':base.sha(__file__),'shared_generator_sha256':base.sha(Path(base.__file__)),'protected':protected,'max_requests':16,'raw_capture_required':True,'no_reference_inputs':True,'explicit_change':'Allow negatives inside an unconfirmed parent search seed; no original/accepted mask is treated as removable. Positive binding to seed radius3 remains. All final gates unchanged.','source_geometry':'same RGB line-pair detector and first pair by old fixed ordering, both opposed quarter-band hypotheses, no score feedback or fallback points','Agent_rounds':3}
 base.save(OUT/'frozen_plan.json',plan);print(json.dumps({'prepared':len(plans),'unique':len({p['key'] for p in plans}),'seed_count':8}),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='run':
  plan=base.read(OUT/'frozen_plan.json');assert base.sha(__file__)==plan['opposed_policy_sha256'];assert base.sha(Path(base.__file__))==plan['shared_generator_sha256'];base.run()
 elif sys.argv[1]=='audit':
  import audit_rgb_layers;audit_rgb_layers.main()
