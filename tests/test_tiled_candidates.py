"""Geometry, truncation, deduplication and preservation invariants."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('tiled_candidate_utils', Path(__file__).resolve().parents[1]/'tools/tiled_candidate_utils.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def square(identifier, x0=10, y0=10, size=10, label='pallet', score=.95):
    points = [[x0,y0],[x0+size,y0],[x0+size,y0+size],[x0,y0+size]]
    return {'id': identifier, 'label': label, 'points': points, 'holes': [], 'score': score}


class TiledCandidatesTest(unittest.TestCase):
    def test_grid_is_reference_independent_and_covers_corners(self):
        self.assertEqual(module.tile_grid(1920,1536), [(0,0,1152,922),(768,0,1920,922),(0,614,1152,1536),(768,614,1920,1536)])

    def test_remap_moves_parts_and_holes_without_mutating_input(self):
        result = {'score': .9, 'parts': [{'points': square('a')['points'], 'holes': [[[12,12],[14,12],[14,14],[12,14]]]},
            {'points': square('b',30,20)['points'], 'holes': []}]}
        before = deepcopy(result)
        mapped = module.remap_tile_result(result,(40,30,100,80),100,80,'tile')
        self.assertEqual(mapped['points'][0],[50,40])
        self.assertEqual(mapped['parts'][1]['points'][0],[70,50])
        self.assertEqual(mapped['holes'][0][0],[52,42])
        self.assertEqual(result,before)

    def test_internal_edge_is_rejected_but_original_edge_is_allowed(self):
        result = {'score': .9, 'parts': [{'points':square('a',0,10)['points']}]}
        self.assertIsNone(module.remap_tile_result(result,(40,0,100,60),100,80,'a'))
        self.assertIsNotNone(module.remap_tile_result(result,(0,0,60,60),100,80,'a'))

    def test_baseline_and_person_masks_are_preserved_and_nms_suppresses_duplicates(self):
        baseline = [square('base'),square('person',20,40,label='person')]
        proposals = [square('copy'),square('new',60,20),square('new-copy',60,20)]
        old = deepcopy(baseline)
        output, report = module.augment_candidates(baseline,proposals,100,80,addition_ratio=1)
        self.assertEqual(output[:2],old)
        self.assertEqual(baseline,old)
        self.assertEqual([shape['id'] for shape in output],['base','person','new'])
        self.assertEqual(report['added_pallets'],1)

    def test_budget_confidence_and_novelty_order(self):
        baseline = [square(str(i),5+i*20,5,5) for i in range(4)]
        proposals = [square('low',5,50,score=.7),square('second',25,50,score=.91),square('first',50,50,score=.99)]
        output, report = module.augment_candidates(baseline,proposals,100,80,minimum_score=.9,addition_ratio=.25)
        self.assertEqual(report['addition_budget'],1)
        self.assertEqual(output[-1]['id'],'first')

    def test_iou_matches_full_raster(self):
        a,b = [module._mask_record(shape,100,80) for shape in [square('a'),square('b',15,15)]]
        self.assertAlmostEqual(module._iou(a,b),36/(121*2-36))
        self.assertEqual(module._iou(a,a),1)

    def test_invalid_confidence_and_wrong_class_are_rejected(self):
        with self.assertRaises(ValueError):
            module.augment_candidates([], [square('a',score=float('nan'))],100,80)
        with self.assertRaises(ValueError):
            module.augment_candidates([], [square('a',label='person')],100,80)


if __name__ == '__main__':
    unittest.main()
