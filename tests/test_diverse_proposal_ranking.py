"""Novelty and dynamic selection retain original geometry and budget."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('diverse_utils',Path(__file__).resolve().parents[1]/'tools/tiled_candidate_utils.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def square(i,x,y,size=8,score=.95):
    return {'id':i,'label':'pallet','points':[[x,y],[x+size,y],[x+size,y+size],[x,y+size]],'holes':[],'score':score}

class DiverseProposalRankingTest(unittest.TestCase):
    def test_cross_scale_agreement_does_not_overrule_exact_seed_duplicate(self):
        seed=[square('seed',20,20)]
        first=[square('duplicate',20,20),square('new',60,60,score=.9)]
        second=[square('duplicate-witness',20,20)]
        priority=m.proposal_priorities(first,second,100,100,'novel_consensus',baseline=seed)
        self.assertEqual(priority['duplicate'],0.)
        self.assertGreater(priority['new'],0.)
        with self.assertRaises(ValueError):m.proposal_priorities(first,second,100,100,'novel_consensus')

    def test_dynamic_novelty_reranks_near_duplicates_below_nms_threshold(self):
        seed=[square(str(i),2+i*12,2,size=5) for i in range(4)]
        proposals=[square('first',50,50,score=.99),square('near',52,50,score=.98),square('distant',75,50,score=.9)]
        before=deepcopy((seed,proposals))
        old,_=m.augment_candidates(seed,proposals,100,100,minimum_score=.5)
        diverse,info=m.augment_candidates(seed,proposals,100,100,minimum_score=.5,dynamic_diversity=True)
        self.assertEqual([s['id'] for s in old[-2:]],['first','near'])
        self.assertEqual([s['id'] for s in diverse[-2:]],['first','distant'])
        self.assertEqual(diverse[:len(seed)],seed)
        self.assertEqual(info['added_pallets'],2)
        self.assertEqual((seed,proposals),before)

    def test_dynamic_mode_keeps_hard_nms_and_zero_addition_case(self):
        p=square('a',20,20)
        copy=deepcopy(p);copy['id']='copy'
        result,_=m.augment_candidates([], [p,copy],100,100,dynamic_diversity=True)
        self.assertEqual(len(result),1)
        result,_=m.augment_candidates([],[],100,100,dynamic_diversity=True)
        self.assertEqual(result,[])

if __name__=='__main__':unittest.main()
