from copy import deepcopy
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from tools.raw_coverage_selection import solve_raw_cover,point_person_guard


class RawCoverageSelectionTests(unittest.TestCase):
 def test_required_coverage_can_displace_the_highest_priority_candidate(self):
  r=solve_raw_cover([{1},{2}],{2},[.9,.1],[],1)
  self.assertEqual(r['selected_indices'],[1]);self.assertTrue(r['summary']['certificate_satisfied'])

 def test_duplicate_points_do_not_replace_a_missing_anchor(self):
  r=solve_raw_cover([{1},{1}],{1,2},[.9,.8],[],2)
  self.assertTrue(r['summary']['infeasibility_proven']);self.assertEqual(r['summary']['uncoverable_raw_points'],1)

 def test_budget_and_pairwise_nms_can_make_full_coverage_infeasible(self):
  for budget,conflicts in ((1,[]),(2,[(0,1)])):
   r=solve_raw_cover([{1},{2}],{1,2},[.9,.8],conflicts,budget)
   self.assertTrue(r['summary']['infeasibility_proven']);self.assertIsNone(r['selected_indices'])

 def test_point_person_guard_keeps_old_overlap_and_rejects_new_overlap(self):
  self.assertTrue(point_person_guard({1,2},{2,3},{1,2}))
  self.assertFalse(point_person_guard({1,2,3},{2,3},{1,2}))
  self.assertTrue(point_person_guard(set(),{2,3},{1,2}))

 def test_finite_small_problems_match_independent_enumeration(self):
  rng=np.random.default_rng(0)
  for _ in range(6):
   sets=[set(np.flatnonzero(rng.random(5)>.45).tolist()) for _ in range(5)]
   weights=rng.random(5);required={0,1};conflicts=[(0,1)];best=None
   for size in range(3):
    for chosen in combinations(range(5),size):
     if 0 in chosen and 1 in chosen:continue
     if required<=set().union(*(sets[i] for i in chosen)):
      score=sum(weights[i] for i in chosen);best=score if best is None else max(best,score)
   actual=solve_raw_cover(sets,required,weights,conflicts,2)
   if best is None:self.assertTrue(actual['summary']['infeasibility_proven'])
   else:self.assertAlmostEqual(actual['summary']['priority_sum'],best)

 def test_timeout_without_incumbent_is_not_a_coverage_certificate(self):
  with patch('tools.raw_coverage_selection.milp',return_value=SimpleNamespace(status=1,x=None)):
   r=solve_raw_cover([{1}],{1},[.9],[],1)
  self.assertEqual(r['summary']['status'],'time_limit');self.assertFalse(r['summary']['certificate_satisfied'])

 def test_unique_component_coverage_can_displace_duplicate_high_priorities(self):
  linear=solve_raw_cover([{1},{1},{2}],set(),[.5,.49,.45],[],2)
  self.assertEqual(linear['selected_indices'],[0,1])
  diverse=solve_raw_cover([{1},{1},{2}],set(),[.5,.49,.45],[],2,
   feature_sets=[{1,2},{1,2},{3,4}],feature_weights={1:.2,2:.2,3:.2,4:.2})
  self.assertEqual(diverse['selected_indices'],[0,2])
  self.assertAlmostEqual(diverse['summary']['component_coverage_value'],.8)

 def test_inputs_are_immutable_and_invalid_priorities_fail_closed(self):
  points=[{1},{2}];before=deepcopy(points);weights=np.array([.9,.8]);copy=weights.copy()
  solve_raw_cover(points,{1},weights,[],1);self.assertEqual(points,before);np.testing.assert_array_equal(weights,copy)
  with self.assertRaises(ValueError):solve_raw_cover(points,{1},[float('nan'),.8],[],1)
  with self.assertRaises(ValueError):solve_raw_cover(points,{1},[1.1,.8],[],1)

 def test_unique_component_objective_matches_finite_enumeration(self):
  rng=np.random.default_rng(4)
  for _ in range(5):
   points=[set(np.flatnonzero(rng.random(4)>.4).tolist()) for _ in range(4)]
   features=[set(np.flatnonzero(rng.random(5)>.5).tolist()) for _ in range(4)]
   all_features=set().union(*features);fw={p:float(rng.random()/5) for p in all_features};weights=rng.random(4)/2
   required={0};best=None
   for size in range(3):
    for chosen in combinations(range(4),size):
     if not required<=set().union(*(points[i] for i in chosen)):continue
     score=sum(weights[i] for i in chosen)+sum(fw[p] for p in set().union(*(features[i] for i in chosen)))
     best=score if best is None else max(best,score)
   actual=solve_raw_cover(points,required,weights,[],2,feature_sets=features,feature_weights=fw)
   if best is None:self.assertTrue(actual['summary']['infeasibility_proven'])
   else:self.assertAlmostEqual(actual['summary']['priority_sum']+actual['summary']['component_coverage_value'],best)


if __name__=='__main__':unittest.main()
