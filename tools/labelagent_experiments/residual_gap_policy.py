"""Schedule individual edits and one real residual SAM request within 3 ops."""
import copy
import cv2,numpy as np

def choose_residual_schedule(holes,assessments,operations,geometry,width,height,points_by_shape,feedback,compiler):
 ground={r['hole_id'] for r in assessments if r['semantic']=='ground'}
 actionable=[h for h in holes if h['id'] in ground]
 groups={}
 for h in actionable:groups.setdefault(h['shape_id'],[]).append(h)
 if not groups:return operations[:3]
 shape_id,group=max(groups.items(),key=lambda kv:(len(kv[1]),-kv[0]));state=feedback.get(shape_id)
 if not state:return None
 previous=points_by_shape.get(shape_id,[]);old={(p['x'],p['y']):copy.deepcopy(p) for p in previous}
 raw=state['mask']
 def coverage(g):return float((raw&g['mask']).sum())/max(1,int(g['mask'].sum()))
 ordered=sorted(group,key=lambda g:(coverage(g),-g['area'],g['id']))
 if len(ordered)<=3:return [next(o for o in operations if o.get('hole_id')==g['id']) for g in ordered]
 fills=[next(o for o in operations if o.get('hole_id')==g['id']) for g in ordered[:2]]
 remaining=ordered[2:];accepted=[];new=[]
 def add(p):
  key=(p['x'],p['y'])
  if key in old:return True
  if len(old)>=32:return False
  old[key]=p;new.append(p);return True
 for g in remaining:
  inside=[p for p in old.values() if p['label']==1 and g['mask'][int(p['y']),int(p['x'])]]
  if not inside:
   d=cv2.distanceTransform(g['mask'].astype(np.uint8),cv2.DIST_L2,5);y,x=np.unravel_index(np.argmax(d),d.shape)
   if not add({'x':int(x),'y':int(y),'label':1}):continue
  accepted.append(g)
 for _ in range(2):
  for g in accepted:
   if len(old)>=32:break
   missed=(g['mask']!=0)&(raw==0)
   ys,xs=np.where(missed)
   if not len(xs):continue
   distance=np.full(len(xs),np.inf)
   for p in old.values():distance=np.minimum(distance,(xs-p['x'])**2+(ys-p['y'])**2)
   index=int(np.argmax(distance))
   if distance[index]<=0:continue
   add({'x':int(xs[index]),'y':int(ys[index]),'label':1})
 if len(accepted)<2:return fills
 decision=compiler(accepted,assessments,geometry,width,height,set())
 if decision is None:return fills
 # Old points remain in the cumulative budget. New arguments introduce only
 # actual residual clicks; the executor separately routes active prompts.
 decision['points']=list(old.values());decision['reason']='Two independent hard-gap edits, then one real focused SAM request with budgeted residual clicks.'
 return fills+[decision]
