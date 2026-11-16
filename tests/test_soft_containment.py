"""Soft containment uses mask pixels and never deletes contained proposals."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('soft_containment_utils',Path(__file__).resolve().parents[1]/'tools/tiled_candidate_utils.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def square(i,x,y,size=8,score=.95):
    return {'id':i,'label':'pallet','points':[[x,y],[x+size,y],[x+size,y+size],[x,y+size]],'holes':[],'score':score}

class SoftContainmentTest(unittest.TestCase):
    def test_small_nested_fragment_does_not_outrank_disjoint_candidate_only_due_to_low_iou(self):
        seed=[square('large',10,10,30)]
        proposals=[square('nested',20,20,4,.99),square('new',60,60,8,.6)]
        before=deepcopy((seed,proposals))
        old,_=m.augment_candidates(seed,proposals,100,100,minimum_score=.5)
        soft,info=m.augment_candidates(seed,proposals,100,100,minimum_score=.5,soft_containment=True)
        self.assertEqual(old[-1]['id'],'nested')
        self.assertEqual(soft[-1]['id'],'new')
        self.assertEqual(soft[-1]['score'],.6)
        self.assertEqual((seed,proposals),before)
        self.assertFalse(info['soft_containment']['union_updated_during_ranking'])

    def test_contained_fragment_remains_eligible(self):
        seed=[square('large',10,10,30)]
        proposal=square('nested',20,20,4)
        result,_=m.augment_candidates(seed,[proposal],100,100,soft_containment=True)
        self.assertEqual(result[-1],proposal)

    def test_hole_inside_seed_box_is_outside_actual_seed_mask(self):
        ring=square('ring',10,10,60)
        ring['holes']=[[[25,25],[55,25],[55,55],[25,55]]]
        proposals=[square('in-solid',15,15,4,.8),square('in-hole',35,35,4,.6)]
        result,_=m.augment_candidates([ring],proposals,100,100,minimum_score=.5,soft_containment=True)
        self.assertEqual(result[-1]['id'],'in-hole')

    def test_dynamic_union_updates_after_selection_without_changing_hard_nms_or_budget(self):
        seed=[square(str(i),2+i*12,2,5) for i in range(4)]
        proposals=[square('first',50,50,20,.99),square('fragment',55,55,4,.98),square('new',80,50,8,.7)]
        old,_=m.augment_candidates(seed,proposals,100,100,minimum_score=.5,dynamic_diversity=True)
        soft,info=m.augment_candidates(seed,proposals,100,100,minimum_score=.5,dynamic_diversity=True,soft_containment=True)
        self.assertEqual([s['id'] for s in old[-2:]],['first','fragment'])
        self.assertEqual([s['id'] for s in soft[-2:]],['first','new'])
        self.assertEqual(info['added_pallets'],2)
        self.assertTrue(info['soft_containment']['union_updated_during_ranking'])
        self.assertEqual(soft[:len(seed)],seed)

if __name__=='__main__':unittest.main()
