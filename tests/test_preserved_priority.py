"""Preserving prior masks also preserves the original budget under new weights."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('preserved_priority_utils',Path(__file__).resolve().parents[1]/'tools/tiled_candidate_utils.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def square(i,x,y,score=.95,label='pallet'):
    return {'id':i,'label':label,'points':[[x,y],[x+4,y],[x+4,y+4],[x,y+4]],'holes':[],'score':score}

class PreservedPriorityTest(unittest.TestCase):
    def test_explicit_priority_uses_only_original_remaining_slot(self):
        original=[square(str(i),2+i*10,2) for i in range(4)]
        selected=original+[square('old-add',50,2)]
        proposals=[square('lower-model',60,50,.6),square('higher-model',75,50,.99)]
        result,info=m.augment_preserving_selected(original,selected,proposals,100,80,
            ranking_scores={'lower-model':.9,'higher-model':.1})
        self.assertEqual(result[:len(selected)],selected)
        self.assertEqual(result[-1]['id'],'lower-model')
        self.assertEqual(result[-1]['score'],.6)
        self.assertEqual(len(result)-len(selected),1)
        self.assertEqual(info['original_addition_budget'],2)

    def test_person_rejection_audit_propagates_from_real_mask_filter(self):
        original=[square(str(i),2+i*10,2) for i in range(4)]+[square('person',50,40,label='person')]
        selected=original+[square('old-add',50,2)]
        result,info=m.augment_preserving_selected(original,selected,[square('bad',50,40),square('good',75,50)],
            100,80,maximum_person_overlap=0,dynamic_diversity=True)
        self.assertEqual(info['proposals_rejected_by_predicted_person'],1)
        self.assertEqual(result[-1]['id'],'good')
        self.assertEqual(result[:len(selected)],selected)

if __name__=='__main__':unittest.main()
