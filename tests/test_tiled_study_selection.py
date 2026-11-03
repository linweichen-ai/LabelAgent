"""Keep validation fixed to the tuning-selected method and complete split."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('collect_tiled_candidates', Path(__file__).resolve().parents[1]/'tools/collect_tiled_candidates.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrozenStudyTest(unittest.TestCase):
    def setUp(self):
        self.methods = module.method_grid('.5', '.5', .8)
        self.selection = {'selected_using_tuning_only': True, 'metric_version': 2, 'metric_sha256': 'metric',
            'winner': 'tiles_s50_b50', 'frozen_parameters': self.methods['tiles_s50_b50'], 'heldout_frames': [0,1,3],
            'results': [{'method': 'tiles_s50_b50', 'eligible': True, 'checks': {'gain': True, 'cross_class': True}}]}

    def test_frozen_method_accepted(self):
        module.check_frozen_selection(self.selection,self.methods,[0,1,3],'metric')

    def test_parameter_metric_or_split_drift_rejected(self):
        for methods,frames,metric in [(module.method_grid('.6','.5',.8),[0,1,3],'metric'),
                (self.methods,[0,1],'metric'),(self.methods,[0,1,3],'changed')]:
            with self.assertRaises(ValueError):
                module.check_frozen_selection(self.selection,methods,frames,metric)

    def test_unsuccessful_or_non_tuning_selection_rejected(self):
        for field in ('eligible','cross_class','scope'):
            value = deepcopy(self.selection)
            if field == 'eligible': value['results'][0]['eligible'] = False
            elif field == 'cross_class': value['results'][0]['checks']['cross_class'] = False
            else: value['selected_using_tuning_only'] = False
            with self.assertRaises(ValueError):
                module.check_frozen_selection(value,self.methods,[0,1,3],'metric')

    def test_grid_does_not_silently_alias_values(self):
        for scores,budgets in [('.505','.5'),('.5,.5','.5'),('nan','.5'),('.5','0'),('.5','1.1')]:
            with self.assertRaises(ValueError):
                module.method_grid(scores,budgets,.8)


if __name__ == '__main__':
    unittest.main()
