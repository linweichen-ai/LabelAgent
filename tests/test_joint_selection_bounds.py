from copy import deepcopy
from itertools import combinations
from unittest.mock import patch
from types import SimpleNamespace
import unittest
import numpy as np
from tools.joint_selection_bounds import solve_joint_selection


def problem(matrix,points,people=None,budget=2,fixed=(),conflicts=()):
 return {'matrix':np.array(matrix,dtype=bool),'point_sets':list(map(set,points)),
  'person_sets':list(map(set,people if people is not None else [[] for _ in points])),
  'fixed_columns':list(fixed),'budget':budget,'conflicts':list(conflicts)}


def exhaustive(image,minimum_points,maximum_people):
 fixed=set(image['fixed_columns']);new=[i for i in range(image['matrix'].shape[1]) if i not in fixed];best=None
 def matching(row,used,selected):
  if row==image['matrix'].shape[0]:return 0
  return max([matching(row+1,used,selected)]+[1+matching(row+1,used|{c},selected)
   for c in selected-used if image['matrix'][row,c]])
 for count in range(min(len(new),image['budget'])+1):
  for addition in combinations(new,count):
   selected=fixed|set(addition)
   if any(a in selected and b in selected for a,b in image['conflicts']):continue
   pts=set().union(*(image['point_sets'][c] for c in selected));people=set().union(*(image['person_sets'][c] for c in selected))
   if len(pts)<minimum_points or len(people)>maximum_people:continue
   matches=matching(0,set(),selected);best=matches if best is None else max(best,matches)
 return best


class JointSelectionTests(unittest.TestCase):
 def check_exhaustive(self,image,points,people):
  expected=exhaustive(image,points,people);result=solve_joint_selection([image],points,people,time_limit=5)
  if expected is None:self.assertTrue(result['summary']['infeasibility_proven']);self.assertIsNone(result['_selection'])
  else:
   self.assertTrue(result['summary']['optimality_proven']);self.assertEqual(result['summary']['incumbent']['matched'],expected)
   self.assertEqual(result['summary']['matching_upper_bound'],expected)
  return result

 def test_same_reference_point_cannot_be_counted_twice(self):
  image=problem([[1,0],[0,1]],[[1],[1]])
  self.check_exhaustive(image,1,0);self.check_exhaustive(image,2,0)

 def test_person_union_is_unique_and_cap_is_enforced(self):
  image=problem([[1,0],[0,1]],[[1],[2]],people=[[9],[9]])
  r=self.check_exhaustive(image,0,1);self.assertEqual(r['summary']['incumbent']['matched'],2)
  r=self.check_exhaustive(image,0,0);self.assertEqual(r['summary']['incumbent']['matched'],0)

 def test_one_to_one_matching_nms_and_budget(self):
  image=problem([[1,1],[1,1]],[[1],[2]],conflicts=[(0,1)])
  self.assertEqual(self.check_exhaustive(image,0,0)['summary']['incumbent']['matched'],1)
  image['conflicts']=[];self.assertEqual(self.check_exhaustive(image,0,0)['summary']['incumbent']['matched'],2)
  image['budget']=1;self.assertEqual(self.check_exhaustive(image,0,0)['summary']['incumbent']['matched'],1)

 def test_joint_coverage_requirement_can_reduce_best_instance_count(self):
  image=problem([[1,0,0],[0,1,0]],[[1],[2],[3,4,5]])
  self.assertEqual(self.check_exhaustive(image,0,0)['summary']['incumbent']['matched'],2)
  self.assertEqual(self.check_exhaustive(image,4,0)['summary']['incumbent']['matched'],1)

 def test_fixed_candidates_and_input_data_are_preserved(self):
  image=problem([[1,0],[0,1]],[[1,2],[3]],people=[[9],[9]],budget=1,fixed=[0]);before=deepcopy(image)
  result=self.check_exhaustive(image,3,1);self.assertIn(0,result['_selection'][0])
  np.testing.assert_array_equal(image['matrix'],before['matrix']);self.assertEqual(image['point_sets'],before['point_sets'])
  self.check_exhaustive(image,4,1)

 def test_point_ids_from_different_images_stay_distinct_under_global_caps(self):
  first=problem([[1]],[[1]],people=[[9]],budget=1)
  second=deepcopy(first)
  result=solve_joint_selection([first,second],2,2,time_limit=5)
  self.assertEqual(result['summary']['incumbent']['matched'],2)
  self.assertEqual(result['summary']['incumbent']['points'],2)
  self.assertEqual(result['summary']['incumbent']['person_points'],2)
  result=solve_joint_selection([first,second],2,1,time_limit=5)
  self.assertTrue(result['summary']['infeasibility_proven'])

 def test_timeout_is_not_optimal_or_infeasible_and_invalid_points_are_rejected(self):
  image=problem([[1,0,0],[0,1,0],[0,0,1]],[[1],[2],[3]])
  with patch('tools.joint_selection_bounds.milp',return_value=SimpleNamespace(status=1,message='time limit',x=None,mip_dual_bound=-2.7)):
   summary=solve_joint_selection([image],0,0)['summary']
  self.assertFalse(summary['optimality_proven']);self.assertFalse(summary['infeasibility_proven']);self.assertEqual(summary['matching_upper_bound'],2)
  with self.assertRaises(ValueError):solve_joint_selection([image],-1,0)
  with self.assertRaises(ValueError):solve_joint_selection([image],0,0,time_limit=float('inf'))


if __name__=='__main__':unittest.main()
