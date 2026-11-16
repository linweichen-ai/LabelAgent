"""Ranking experiments retain real scores, budget, masks and person protection."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('priority_utils',Path(__file__).resolve().parents[1]/'tools/tiled_candidate_utils.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def square(i,x,y,score=.9,label='pallet'):
    return {'id':i,'label':label,'points':[[x,y],[x+8,y],[x+8,y+8],[x,y+8]],'holes':[],'score':score}

class ProposalPrioritiesTest(unittest.TestCase):
    def test_cross_scale_support_is_geometric_and_does_not_use_same_scale_votes(self):
        a,b=square('a',20,20),square('b',50,50,.99)
        witness=square('w',20,20,.85)
        before=deepcopy(([a,b],[witness]))
        confidence=m.proposal_priorities([a,b],[witness],100,100,'confidence')
        consensus=m.proposal_priorities([a,b],[witness],100,100,'consensus')
        self.assertGreater(confidence['b'],confidence['a'])
        self.assertGreater(consensus['a'],consensus['b'])
        self.assertEqual(consensus['a'],.9)
        self.assertEqual(m.proposal_priorities([a,witness],[],100,100,'consensus')['a'],.45)
        self.assertEqual(([a,b],[witness]),before)

    def test_priority_override_changes_order_without_changing_model_confidence_or_seed(self):
        seed=[square('seed',2,2)]
        proposals=[square('a',20,20,.9),square('b',50,50,.99)]
        before=deepcopy((seed,proposals))
        result,info=m.augment_candidates(seed,proposals,100,100,ranking_scores={'a':.9,'b':.2})
        self.assertEqual(result[-1],proposals[0])
        self.assertEqual(result[:1],seed)
        self.assertEqual(info['added_pallets'],1)
        self.assertEqual((seed,proposals),before)

    def test_person_guard_precedes_any_external_priority(self):
        seed=[square('p',20,20,label='person')]
        proposals=[square('bad',20,20,.99),square('good',50,50,.8)]
        result,info=m.augment_candidates(seed,proposals,100,100,maximum_person_overlap=0,
            ranking_scores={'bad':1.,'good':0.})
        self.assertEqual(result[-1]['id'],'good')
        self.assertEqual(info['proposals_rejected_by_predicted_person'],1)

    def test_nonfinite_unbound_duplicate_or_wrong_class_priorities_are_rejected(self):
        proposal=square('a',20,20)
        for scores in [{},{'b':.9},{'a':float('nan')},{'a':1.1},{'a':-.1}]:
            with self.assertRaises(ValueError):m.augment_candidates([],[proposal],100,100,ranking_scores=scores)
        for first,second in [([proposal],[proposal]),([square('a',20,20,label='person')],[])]:
            with self.assertRaises(ValueError):m.proposal_priorities(first,second,100,100,'consensus')

    def test_nms_and_input_tie_breaking_still_apply(self):
        proposals=[square('first',20,20),square('copy',20,20),square('other',50,50)]
        result,_=m.augment_candidates([],proposals,100,100,ranking_scores={s['id']:.5 for s in proposals})
        self.assertEqual(result[-1]['id'],'first')
        with self.assertRaises(ValueError):m.proposal_priorities([],[],100,100,'unknown')

if __name__=='__main__':unittest.main()
