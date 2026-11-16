"""Adding another scale cannot displace already selected masks or expand budget."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('preserve_tile_utils',Path(__file__).resolve().parents[1]/'tools/tiled_candidate_utils.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def square(i,x,y,label='pallet',score=.95):
    return {'id':i,'label':label,'points':[[x,y],[x+4,y],[x+4,y+4],[x,y+4]],'holes':[],'score':score}

class PreserveSelectedTilesTest(unittest.TestCase):
    def test_original_budget_not_recomputed_from_augmented_seed(self):
        original=[square(str(i),i*10,2) for i in range(4)]+[square('person',2,50,'person')]
        selected=deepcopy(original)+[square('old-add',50,2)]
        proposals=[square('new',60,20),square('extra',70,20)]
        before=deepcopy((original,selected,proposals))
        result,info=m.augment_preserving_selected(original,selected,proposals,100,80)
        self.assertEqual(result[:len(selected)],selected)
        self.assertEqual(len(result),len(original)+2)
        self.assertEqual(info['remaining_addition_budget'],1)
        self.assertEqual((original,selected,proposals),before)

    def test_exhausted_budget_returns_selected_without_changes(self):
        original=[square('a',2,2),square('b',20,2)]
        selected=deepcopy(original)+[square('old',40,2)]
        result,info=m.augment_preserving_selected(original,selected,[square('new',60,2)],100,80)
        self.assertEqual(result,selected)
        self.assertEqual(info['added_pallets'],0)

    def test_changed_original_or_already_exceeded_budget_is_rejected(self):
        original=[square('a',2,2),square('b',20,2)]
        for selected in [original[:1],original+[square('old1',40,2),square('old2',60,2)],
                original+[square('wrong',40,2,'person')]]:
            with self.assertRaises(ValueError):m.augment_preserving_selected(original,selected,[],100,80)

if __name__=='__main__':unittest.main()
