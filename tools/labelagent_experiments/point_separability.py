"""Reference-only geometric relaxation; never a proposal/mask generator."""
import math
import numpy as np
import cv2
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def analyze(pixels,owners,target_owner,radius,width=1920,height=1536):
 pixels=np.asarray(pixels,dtype=int);owners=np.asarray(owners);target=np.flatnonzero(owners==target_owner);other=np.flatnonzero(owners!=target_owner)
 ordered=np.r_[target,other];p=pixels[ordered];n=len(target);m=len(other)
 if n<5:raise ValueError('Original eligible target requires at least five visible reference points')
 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1));dy,dx=np.where(kernel);offsets=list(zip(dx-radius,dy-radius));allowed=set()
 for x,y in pixels[target]:
  for ox,oy in offsets:
   if 0<=x+ox<width and 0<=y+oy<height:allowed.add((int(x+ox),int(y+oy)))
 signatures={};pure_covered=set();per_target_pure=np.zeros(n,int)
 for x,y in sorted(allowed):
  delta=p-np.array([x,y]);inside=(abs(delta[:,0])<=radius)&(abs(delta[:,1])<=radius);indices=np.flatnonzero(inside);indices=indices[kernel[delta[indices,1]+radius,delta[indices,0]+radius]!=0]
  signature=tuple(map(int,indices))
  if not any(i<n for i in signature):continue
  signatures[signature]=signatures.get(signature,0)+1
  if all(i<n for i in signature):
   pure_covered.update(signature)
   for i in signature:per_target_pure[i]+=1
 sigs=sorted(signatures);s=len(sigs);row=[];col=[];values=[];lower=[];upper=[]
 def constraint(items,lo=-np.inf,hi=np.inf):
  r=len(lower)
  for c,v in items:row.append(r);col.append(c);values.append(v)
  lower.append(lo);upper.append(hi)
 incident=[[] for _ in range(n+m)]
 for j,sig in enumerate(sigs):
  for i in sig:
   incident[i].append(j);constraint([(j,1),(s+i,-1)],hi=0)
 for i,js in enumerate(incident):constraint([(s+i,1)]+[(j,-1) for j in js],hi=0)
 constraint([(s+i,1) for i in range(n)],lo=math.ceil(.5*n))
 constraint([(s+i,33) for i in range(n)]+[(s+n+i,-67) for i in range(m)],lo=0)
 matrix=coo_matrix((values,(row,col)),shape=(len(lower),s+n+m)).tocsc()
 cost=np.r_[np.zeros(s+n),np.ones(m)]
 result=milp(cost,integrality=np.ones(s+n+m),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,np.array(lower),np.array(upper)),options={'time_limit':60,'mip_rel_gap':0.0})
 record={'radius':radius,'target_points':n,'other_PAL_points':m,'candidate_pixel_count':len(allowed),'unique_hit_signatures':s,'target_points_reachable_without_any_other_PAL':len(pure_covered),'pure_pixel_counts_per_target':per_target_pure.tolist(),'minimum_required_target_hits':math.ceil(.5*n),'solver_status':int(result.status),'solver_message':str(result.message),'relaxation_feasible':None,'allows_arbitrary_disconnected_pixels':True,'ignores_image_semantics_area_and_people':True,'never_eligible_for_annotation_or_adoption':True}
 if result.status==0:
  x=result.x[:s]>.5;actual=set(i for j,sig in enumerate(sigs) if x[j] for i in sig);a=sum(i<n for i in actual);b=len(actual)-a
  assert a>=math.ceil(.5*n) and 33*a>=67*b
  assert all(bool(result.x[s+i]>.5)==(i in actual) for i in range(n+m))
  record.update(relaxation_feasible=True,optimal_other_hits=b,target_hits_in_solver_solution=a,solution_purity=a/(a+b),solver_success=True,integer_solution_recomputed=True)
 elif result.status==2:record.update(relaxation_feasible=False,solver_success=True)
 else:record.update(solver_success=False,proof_incomplete=True)
 return record
