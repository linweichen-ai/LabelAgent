import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/labelagent_experiments'))
import unittest,numpy as np
from preserved_model_policy import select_complete_masks
class PreservedPolicyTests(unittest.TestCase):
 def test_complete_masks_and_camera_reserves_cannot_starve_second_camera(self):
  base=np.zeros((32,32),np.uint8);base[2:12,2:12]=1
  bank={};bases={};people={};originals={}
  for case in ['a','b']:
   bases[case]=[base.copy(),base.copy()];people[case]=[];mask=base.copy();mask[2:12,12:14]=1
   bank[case]=[{'key':case,'mask':mask,'shape':{'original':'unchanged'},'trained_score':.9}];originals[case]=mask.copy()
  selected,rejected,summary=select_complete_masks(bank,bases,people)
  self.assertEqual(len(selected['a']),1);self.assertEqual(len(selected['b']),1)
  for case in bank:self.assertTrue(np.array_equal(originals[case],bank[case][0]['mask']));self.assertEqual(summary[case]['new_union_pixels'],120)
 def test_oversized_mask_is_rejected_entirely_and_person_mask_is_protected(self):
  base=np.zeros((32,32),np.uint8);base[2:12,2:12]=1;big=np.ones_like(base);person=np.zeros_like(base);person[20:24,20:24]=1
  candidates=[{'key':'big','mask':big,'trained_score':1.0},{'key':'person','mask':person,'trained_score':.9}]
  selected,rejected,summary=select_complete_masks({'a':candidates},{'a':[base,base]},{'a':[person]})
  self.assertEqual(selected['a'],[]);self.assertEqual(summary['a']['new_union_pixels'],100);self.assertEqual(len(rejected),2)
if __name__=='__main__':unittest.main(verbosity=2)
