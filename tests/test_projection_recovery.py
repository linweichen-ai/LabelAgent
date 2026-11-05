import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools/labelagent_experiments"))
import sys,unittest,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from incumbent_append_policy import append_complete_masks
from projection_audit_core import other_box_front_hits
class ProjectionRecoveryTests(unittest.TestCase):
 def test_append_keeps_incumbent_and_rejects_whole_oversized_prediction(self):
  original=np.zeros((32,32),np.uint8);original[5:15,5:15]=1
  inc=original.copy();inc[15:17,5:17]=1
  self.assertEqual(int(inc.sum()),124)
  big=inc.copy();big[17,5:7]=1;small=inc.copy();small[17,5]=1
  before=[original.copy(),inc.copy(),big.copy(),small.copy()]
  selected,rejected,summary=append_complete_masks({'a':[{'key':'big','mask':big,'score':1},{'key':'small','mask':small,'score':.9}]},{'a':[original]*4+[inc]},{'a':[original]*4},{'a':[]})
  self.assertEqual([r['key'] for r in selected['a']],['small']);self.assertEqual(summary['PAL_count'],6);self.assertEqual(summary['PAL_union_pixels'],125)
  self.assertTrue(any(r['key']=='big' and 'area' in r['reason'] for r in rejected))
  for a,b in zip(before,[original,inc,big,small]):self.assertTrue(np.array_equal(a,b))
 def test_human_guard_rejects_entire_mask_without_erasing_conflict(self):
  base=np.zeros((64,64),np.uint8);base[5:15,5:15]=1;person=np.zeros_like(base);person[30:35,30:35]=1;candidate=base.copy();candidate[27:31,30:33]=1
  original=candidate.copy();selected,rejected,summary=append_complete_masks({'a':[{'key':'touches_person','mask':candidate,'score':1}]},{'a':[base]*4},{'a':[base]*4},{'a':[person]})
  self.assertEqual(selected['a'],[]);self.assertIn('person',rejected[0]['reason']);self.assertTrue(np.array_equal(candidate,original))
 def test_ray_box_proxy_checks_actual_intersection_and_excludes_own_box(self):
  def box(center,size):return SimpleNamespace(center=np.array(center,float),size=np.array(size,float),rotation=np.eye(3),label='Pallet')
  data={'xyz':np.array([[10.,0,0],[10.,4,0]]),'owner':np.array([1,1]),'boxes':[{'obj_id':1,'box':box([10,2,0],[1,6,1])},{'obj_id':2,'box':box([5,0,0],[2,2,2])}]}
  transform=np.eye(4);transform[:3,:3]=[[0,1,0],[0,0,1],[1,0,0]];rows=other_box_front_hits(data,{'transform':transform},np.array([0,1]))
  self.assertEqual([v['object_id'] for v in rows[0]],[2]);self.assertEqual(rows[1],[]);self.assertGreater(rows[0][0]['front_depth_delta_m'],.15)
if __name__=='__main__':unittest.main(verbosity=2)
