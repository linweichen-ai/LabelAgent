"""Predicted people protect only newly appended masks, without GT access."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('tile_utils_guard',Path(__file__).resolve().parents[1]/'tools/tiled_candidate_utils.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def square(i,x,y,size=8,label='pallet',score=.95):
    return {'id':i,'label':label,'points':[[x,y],[x+size,y],[x+size,y+size],[x,y+size]],'holes':[],'score':score}

class TilePersonGuardTest(unittest.TestCase):
    def test_guard_rejects_person_overlap_and_keeps_seed_and_novel_pallet(self):
        seed=[square('b',2,2),square('p',30,30,label='person')]
        proposals=[square('bad',30,30,score=.99),square('good',55,55)]
        before=deepcopy((seed,proposals))
        legacy,info=m.augment_candidates(seed,proposals,80,80)
        guarded,stats=m.augment_candidates(seed,proposals,80,80,maximum_person_overlap=0)
        self.assertEqual(legacy[-1]['id'],'bad')
        self.assertEqual(guarded[-1]['id'],'good')
        self.assertEqual(guarded[:len(seed)],seed)
        self.assertEqual((seed,proposals),before)
        self.assertEqual(stats['proposals_rejected_by_predicted_person'],1)
        self.assertNotIn('maximum_predicted_person_overlap',info)

    def test_holes_use_actual_pixels_not_person_bounding_box(self):
        person=square('person',10,10,50,'person')
        person['holes']=[[[20,20],[50,20],[50,50],[20,50]]]
        proposed=square('inside-hole',30,30,5)
        result,_=m.augment_candidates([person],[proposed],80,80,maximum_person_overlap=0)
        self.assertEqual(result[-1]['id'],'inside-hole')

    def test_union_does_not_double_count_overlapping_people(self):
        people=[square('p1',20,20,8,'person'),square('p2',20,20,8,'person')]
        proposal=square('a',20,20,12)
        result,_=m.augment_candidates(people,[proposal],80,80,maximum_person_overlap=.5)
        self.assertEqual(result[-1]['id'],'a')

    def test_invalid_overlap_and_unlabelled_person_metadata_do_not_bypass_validation(self):
        for limit in [float('nan'),float('inf'),-.1,1.01]:
            with self.assertRaises(ValueError):m.augment_candidates([],[],80,80,maximum_person_overlap=limit)

if __name__=='__main__':unittest.main()
