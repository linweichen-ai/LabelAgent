"""Raw-point coverage selection; no reference labels or semantic point targets."""
import math,time
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_matrix


def point_person_guard(candidate_points,person_points,anchor_points):
 """Allow old predicted-person overlap, prevent newly covering those raw points."""
 return not bool(set(candidate_points)&(set(person_points)-set(anchor_points)))


def solve_raw_cover(point_sets,required_points,priorities,conflicts,budget,time_limit=10.,feature_sets=None,feature_weights=None):
 """Maximize fixed unlabelled priorities while covering every required raw point."""
 sets=list(map(set,point_sets));required=set(required_points);weights=np.asarray(priorities,dtype=float)
 n=len(sets)
 if weights.shape!=(n,) or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
  raise ValueError('Aligned finite priorities in [0,1] required')
 if type(budget) is not int or budget<0 or not math.isfinite(time_limit) or time_limit<=0:raise ValueError('Valid budget and finite positive time limit required')
 if any(any(not isinstance(p,(int,np.integer)) or p<0 for p in s) for s in sets+[required]):raise ValueError('Nonnegative raw point indices required')
 pairs=list(conflicts)
 if any(type(a) is not int or type(b) is not int or not 0<=a<n or not 0<=b<n or a==b for a,b in pairs):raise ValueError('Valid pairwise conflicts required')
 available=set().union(*sets);missing=required-available
 features=[set() for _ in sets] if feature_sets is None else list(map(set,feature_sets))
 feature_weights={} if feature_weights is None else dict(feature_weights)
 if len(features)!=n or any(any(not isinstance(p,(int,np.integer)) or p<0 for p in s) for s in features):raise ValueError('Aligned raw component point sets required')
 if any(not isinstance(p,(int,np.integer)) or p<0 or not math.isfinite(w) or w<0 for p,w in feature_weights.items()):raise ValueError('Finite nonnegative component weights required')
 if set(feature_weights)!=set().union(*features):raise ValueError('Every raw component point needs a weight')
 metadata={'candidates':n,'required_raw_points':len(required),'uncoverable_raw_points':len(missing),'addition_budget':budget,'time_limit_seconds':time_limit}
 if missing:return {'summary':{**metadata,'status':'infeasible','infeasibility_proven':True,'certificate_satisfied':False,'reason':'raw_anchor_points_unavailable'},'selected_indices':None}
 if not n:
  return {'summary':{**metadata,'status':'optimal','optimality_proven':True,'certificate_satisfied':True,'selected':0,'priority_sum':0.,'elapsed_seconds':0.},'selected_indices':[]}
 feature_keys=sorted(feature_weights);feature_variables={p:n+i for i,p in enumerate(feature_keys)}
 costs=-np.r_[weights,[feature_weights[p] for p in feature_keys]]
 rows=[{i:1. for i in range(n)}];lower=[-np.inf];upper=[budget]
 for a,b in pairs:rows.append({a:1.,b:1.});lower.append(-np.inf);upper.append(1.)
 incidence={int(p):[] for p in required}
 for i,points in enumerate(sets):
  for p in points&required:incidence[int(p)].append(i)
 for p in sorted(incidence):rows.append({i:1. for i in incidence[p]});lower.append(1.);upper.append(np.inf)
 for p,z in feature_variables.items():
  rows.append({z:1.,**{i:-1. for i,s in enumerate(features) if p in s}});lower.append(-np.inf);upper.append(0.)
 rr=[];cc=[];values=[]
 for r,row in enumerate(rows):
  for c,v in row.items():rr.append(r);cc.append(c);values.append(v)
 matrix=coo_matrix((np.asarray(values,float),(rr,cc)),shape=(len(rows),len(costs))).tocsc()
 started=time.monotonic()
 result=milp(costs,integrality=np.ones(len(costs)),bounds=Bounds(0,1),
  constraints=LinearConstraint(matrix,np.asarray(lower),np.asarray(upper)),options={'time_limit':time_limit,'mip_rel_gap':0.,'presolve':True})
 selected=None;verified=False
 if getattr(result,'x',None) is not None:
  raw=np.asarray(result.x);binary=np.rint(raw)
  if np.max(np.abs(raw-binary))>1e-5 or np.any((binary<0)|(binary>1)):raise ValueError('Invalid integral incumbent')
  actual_constraints=matrix@binary
  if np.any(actual_constraints<np.asarray(lower)-1e-6) or np.any(actual_constraints>np.asarray(upper)+1e-6):raise ValueError('Solver constraint readback failed')
  selected=[i for i in range(n) if binary[i]==1]
  covered=set().union(*(sets[i] for i in selected))
  verified=required<=covered and len(selected)<=budget and not any(a in selected and b in selected for a,b in pairs)
  if not verified:raise ValueError('Raw coverage/NMS/budget independent readback failed')
  objective=float(weights[selected].sum())+sum(feature_weights[p] for p in set().union(*(features[i] for i in selected)))
  if result.status==0 and not math.isclose(objective,-float(result.fun),abs_tol=1e-6):raise ValueError('Unique component objective readback failed')
 summary={**metadata,'status':{0:'optimal',1:'time_limit',2:'infeasible',3:'unbounded',4:'solver_error'}.get(int(result.status),'unknown'),
  'solver_status':int(result.status),'optimality_proven':bool(result.status==0),'infeasibility_proven':bool(result.status==2),
  'certificate_satisfied':bool(verified),'selected':len(selected) if selected is not None else None,
  'priority_sum':float(weights[selected].sum()) if selected is not None else None,
  'component_coverage_value':sum(feature_weights[p] for p in set().union(*(features[i] for i in selected))) if selected is not None else None,
  'elapsed_seconds':round(time.monotonic()-started,4)}
 return {'summary':summary,'selected_indices':selected}
