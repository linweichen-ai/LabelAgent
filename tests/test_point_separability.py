"""Independent small-case raster enumeration checks the MILP relaxation."""
import sys,itertools,unittest
from pathlib import Path
import numpy as np,cv2
here=Path(__file__).resolve()
for root in [here.parent]+[p/'tools/labelagent_experiments' for p in here.parents]:
 if (root/'point_separability.py').exists():sys.path.insert(0,str(root));break
from point_separability import analyze

def brute(px,owners,radius):
 target=np.flatnonzero(owners==1);other=np.flatnonzero(owners!=1);footprints=set();kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2)
 for y in range(9):
  for x in range(9):
   mask=np.zeros((9,9),np.uint8);mask[y,x]=1;dilated=cv2.dilate(mask,kernel);hits=dilated[px[:,1],px[:,0]]!=0
   if hits[target].any():footprints.add(tuple(np.flatnonzero(hits)))
 assert len(footprints)<=16
 best=None
 for choices in itertools.product((False,True),repeat=len(footprints)):
  hit=set(i for include,sig in zip(choices,sorted(footprints)) if include for i in sig);a=sum(i in hit for i in target);b=sum(i in hit for i in other)
  if a*2>=len(target) and 33*a>=67*b:best=b if best is None else min(best,b)
 return best

class Separability(unittest.TestCase):
 def test_coincident_references_infeasible_at_zero_and_three(self):
  p=np.array([[1,1],[3,1],[5,1],[1,5],[5,5]]*2);owners=np.array([1]*5+[2]*5)
  for radius in (0,3):self.assertFalse(analyze(p,owners,1,radius,9,9)['relaxation_feasible'])
 def test_separated_target_and_other_points_have_pure_sites(self):
  p=np.array([[1,1],[2,1],[3,1],[4,1],[5,1],[7,7]]);r=analyze(p,np.array([1]*5+[2]),1,0,9,9)
  self.assertTrue(r['relaxation_feasible']);self.assertEqual(r['optimal_other_hits'],0);self.assertEqual(r['target_points_reachable_without_any_other_PAL'],5)
 def test_mixed_case_exactly_matches_exhaustive_raster_search(self):
  p=np.array([[1,1],[2,1],[3,1],[4,1],[5,1],[1,1],[2,1],[3,1]]);owners=np.array([1]*5+[2]*3);r=analyze(p,owners,1,0,9,9)
  self.assertEqual(brute(p,owners,0),1);self.assertEqual(r['optimal_other_hits'],1)
 def test_duplicate_target_samples_count_like_original_metric(self):
  p=np.array([[1,1],[1,1],[2,1],[3,1],[4,1],[1,1],[2,1]]);owners=np.array([1]*5+[2]*2);r=analyze(p,owners,1,0,9,9)
  self.assertEqual(r['optimal_other_hits'],brute(p,owners,0));self.assertTrue(r['integer_solution_recomputed'])
 def test_ineligible_target_is_not_redefined(self):
  with self.assertRaises(ValueError):analyze(np.array([[1,1]]),np.array([1]),1,3,9,9)
if __name__=='__main__':unittest.main()
