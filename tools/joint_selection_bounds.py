"""Reference-informed numeric feasibility only; never a prediction policy."""
import math
import time
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_matrix
from tools.compare_boxes3d import match


def selected_statistics(images,selected):
 total={'matched':0,'points':0,'person_points':0,'candidates':0}
 for image,columns in zip(images,selected):
  columns=sorted(columns)
  total['matched']+=len(match(np.asarray(image['matrix'],float)[:,columns],.5))
  total['points']+=len(set().union(*(set(image['point_sets'][i]) for i in columns)))
  total['person_points']+=len(set().union(*(set(image['person_sets'][i]) for i in columns)))
  total['candidates']+=len(columns)
 return total


def solve_joint_selection(images,minimum_points,maximum_person_points,time_limit=30.):
 """Return counts plus a private internal selection for readback; do not export it.

 All new candidates are supplied explicitly. Pixel-union area is intentionally
 absent, so this is an optimistic bound until actual area is checked separately.
 """
 if any(type(v) is not int or v<0 for v in (minimum_points,maximum_person_points)):
  raise ValueError('Nonnegative integer point requirements required')
 if not math.isfinite(time_limit) or time_limit<=0:raise ValueError('Positive finite solver time limit required')
 costs=[];rows=[];x_variables=[];total_objects=0;base_points=0;base_people=0
 point_incidence={};person_incidence={}
 def variable(cost=0.):
  index=len(costs);costs.append(cost);return index
 def constrain(coefficients,lower=-np.inf,upper=np.inf):rows.append((coefficients,float(lower),float(upper)))
 for image_index,image in enumerate(images):
  matrix=np.asarray(image['matrix'])
  if matrix.ndim!=2 or not np.isin(matrix,[0,1]).all():raise ValueError('Boolean object/candidate incidence required')
  n_objects,n_columns=matrix.shape;total_objects+=n_objects
  fixed=list(image['fixed_columns']);fixed_set=set(fixed)
  if len(fixed)!=len(fixed_set) or any(type(i) is not int or i<0 or i>=n_columns for i in fixed):
   raise ValueError('Valid distinct fixed columns required')
  if type(image['budget']) is not int or image['budget']<0:raise ValueError('Nonnegative new-candidate budget required')
  if len(image['point_sets'])!=n_columns or len(image['person_sets'])!=n_columns:raise ValueError('Aligned point sets required')
  for key in ('point_sets','person_sets'):
   if any(any(not isinstance(p,(int,np.integer)) or p<0 for p in values) for values in image[key]):raise ValueError('Nonnegative point indices required')
  new=[i for i in range(n_columns) if i not in fixed_set]
  variables={i:variable() for i in new};x_variables.append(variables)
  constrain({v:1 for v in variables.values()},upper=image['budget'])
  for a,b in image.get('conflicts',[]):
   if a==b or a not in variables or b not in variables:raise ValueError('Conflicts must join distinct new columns')
   constrain({variables[a]:1,variables[b]:1},upper=1)
  point_base=set().union(*(set(image['point_sets'][i]) for i in fixed))
  person_base=set().union(*(set(image['person_sets'][i]) for i in fixed))
  base_points+=len(point_base);base_people+=len(person_base)
  for column,x in variables.items():
   for p in set(image['point_sets'][column])-point_base:point_incidence.setdefault((image_index,int(p)),[]).append(x)
   for p in set(image['person_sets'][column])-person_base:person_incidence.setdefault((image_index,int(p)),[]).append(x)
  object_variables=[[] for _ in range(n_objects)];candidate_variables=[[] for _ in range(n_columns)]
  for row,column in np.argwhere(matrix):
   y=variable(-1.);object_variables[row].append(y);candidate_variables[column].append(y)
   if column in variables:constrain({y:1,variables[column]:-1},upper=0)
  for group in object_variables+candidate_variables:constrain({y:1 for y in group},upper=1)
 point_variables=[];person_variables=[]
 for incidence in point_incidence.values():
  z=variable();point_variables.append(z)
  constrain({z:1,**{x:-1 for x in incidence}},upper=0)
 for incidence in person_incidence.values():
  h=variable();person_variables.append(h)
  for x in incidence:constrain({x:1,h:-1},upper=0)
 constrain({z:1 for z in point_variables},lower=max(0,minimum_points-base_points))
 constrain({h:1 for h in person_variables},upper=maximum_person_points-base_people)
 metadata={'variables':len(costs),'constraints':len(rows),'objects':total_objects,
  'new_candidates':sum(map(len,x_variables)),'new_reference_point_variables':len(point_variables),
  'new_person_point_variables':len(person_variables),'fixed_points':base_points,'fixed_person_points':base_people,
  'minimum_points':minimum_points,'maximum_person_points':maximum_person_points,'time_limit_seconds':time_limit}
 if not costs:
  feasible=all(lower<=0<=upper for _,lower,upper in rows)
  return {'summary':{**metadata,'solver_status':0 if feasible else 2,'optimality_proven':feasible,
   'infeasibility_proven':not feasible,'incumbent':selected_statistics(images,[set(i['fixed_columns']) for i in images]) if feasible else None,
   'matching_upper_bound':0 if feasible else None,'elapsed_seconds':0.},
   '_selection':[set(i['fixed_columns']) for i in images] if feasible else None}
 rr=[];cc=[];data=[]
 for r,(coefficients,_,_) in enumerate(rows):
  for c,value in coefficients.items():rr.append(r);cc.append(c);data.append(value)
 matrix=coo_matrix((np.asarray(data,float),(rr,cc)),shape=(len(rows),len(costs))).tocsc()
 lower=np.array([row[1] for row in rows]);upper=np.array([row[2] for row in rows])
 started=time.monotonic()
 result=milp(np.asarray(costs),integrality=np.ones(len(costs)),bounds=Bounds(0,1),
  constraints=LinearConstraint(matrix,lower,upper),options={'time_limit':time_limit,'mip_rel_gap':0.,'presolve':True})
 elapsed=time.monotonic()-started
 selection=None;incumbent=None
 if getattr(result,'x',None) is not None:
  raw=np.asarray(result.x);rounded=np.rint(raw)
  if np.max(np.abs(raw-rounded))>1e-5:raise ValueError('Solver returned nonintegral incumbent')
  values=matrix@rounded
  if np.any(values<lower-1e-6) or np.any(values>upper+1e-6):raise ValueError('Solver incumbent failed independent constraint readback')
  selection=[set(image['fixed_columns'])|{column for column,x in variables.items() if rounded[x]==1}
   for image,variables in zip(images,x_variables)]
  incumbent=selected_statistics(images,selection)
  if incumbent['points']<minimum_points or incumbent['person_points']>maximum_person_points:raise ValueError('True point union failed readback')
  if result.status==0 and incumbent['matched']!=round(-result.fun):raise ValueError('Assignment objective failed matching readback')
 dual=getattr(result,'mip_dual_bound',None)
 bound=min(total_objects,math.floor(-float(dual)+1e-6)) if dual is not None and math.isfinite(dual) else None
 if result.status==0:bound=incumbent['matched']
 if incumbent is not None and bound is not None and incumbent['matched']>bound:raise ValueError('Invalid integer objective bound')
 summary={**metadata,'solver_status':int(result.status),'solver_message':str(result.message),
  'optimality_proven':result.status==0,'infeasibility_proven':result.status==2,
  'incumbent':incumbent,'matching_upper_bound':bound,'elapsed_seconds':round(elapsed,4),
  'mip_gap':float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None}
 return {'summary':summary,'_selection':selection}
