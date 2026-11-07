from copy import deepcopy
import unittest
from tools.replay_multiscale_candidates import select_replay_frames


class ReplaySubsetTests(unittest.TestCase):
 def test_default_identical_cohort_is_preserved_without_input_mutation(self):
  plans=[{'frames':[1,3],'cameras':[0,1]} for _ in range(2)];before=deepcopy(plans)
  selected=select_replay_frames(plans);self.assertEqual(selected,[1,3]);selected.append(7)
  self.assertEqual(plans,before)

 def test_explicit_intersection_reuses_unequal_complete_source_cohorts(self):
  plans=[{'frames':[1,2,3,4],'cameras':[0,1]},{'frames':[2,3],'cameras':[0,1]}];before=deepcopy(plans)
  self.assertEqual(select_replay_frames(plans,'3,2'),[2,3]);self.assertEqual(plans,before)
  with self.assertRaises(ValueError):select_replay_frames(plans)

 def test_missing_duplicate_empty_or_camera_mismatch_cannot_silently_expand_data(self):
  plans=[{'frames':[1,2],'cameras':[0]},{'frames':[2,3],'cameras':[0]}]
  for value in ('1','2,2','','1,3'):
   with self.assertRaises(ValueError):select_replay_frames(plans,value)
  plans[1]['cameras']=[1]
  with self.assertRaises(ValueError):select_replay_frames(plans,'2')


if __name__=='__main__':unittest.main()
