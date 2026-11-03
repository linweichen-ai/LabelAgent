"""Overlap hypotheses must not erase pixels or collapse original instances."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
spec=importlib.util.spec_from_file_location('merge_candidate_overlaps',Path(__file__).resolve().parents[1]/'tools/merge_candidate_overlaps.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def square(identifier,x=10,label='pallet'):
    return {'id':identifier,'label':label,'points':[[x,10],[x+20,10],[x+20,30],[x,30]],'holes':[]}


class MergeOverlapTest(unittest.TestCase):
    def test_merge_preserves_union_and_inputs(self):
        baseline=[square('base')]
        additions=[square('new',12)]
        before=deepcopy([baseline,additions])
        output,report=module.merge_added_overlaps(baseline,additions,100,80,.7)
        self.assertEqual(len(output),1)
        self.assertEqual(output[0]['id'],'base')
        expected=module._rasterize(baseline[0],100,80)|module._rasterize(additions[0],100,80)
        self.assertTrue(np.array_equal(module._rasterize(output[0],100,80),expected))
        self.assertEqual([baseline,additions],before)
        self.assertEqual(report['merged_additions'],1)

    def test_holes_and_other_class_are_preserved(self):
        base=square('base')
        base['holes']=[[[15,15],[25,15],[25,25],[15,25]]]
        person=square('person',60,label='person')
        addition=deepcopy(base)
        addition['id']='addition'
        output,report=module.merge_added_overlaps([base,person],[addition],100,80,.7)
        self.assertEqual(output[1],person)
        self.assertEqual(module._rasterize(output[0],100,80)[20,20],0)
        self.assertEqual(report['merged_additions'],1)

    def test_two_original_candidates_never_collapse(self):
        output,report=module.merge_added_overlaps([square('a'),square('b',12)],[square('new',11)],100,80,.7)
        self.assertEqual(len(output),2)
        self.assertEqual({shape['id'] for shape in output},{'a','b'})

    def test_distinct_addition_is_retained(self):
        output,report=module.merge_added_overlaps([square('a')],[square('new',60)],100,80,.7)
        self.assertEqual(len(output),2)
        self.assertEqual(report['merged_additions'],0)

    def test_export_pixel_loss_falls_back_to_unmerged_inputs(self):
        baseline,additions=[square('a')],[square('new',12)]
        def lossy(mask,label,identifier):
            return {'id':identifier,'label':label,'points':[[1,1],[2,1],[2,2],[1,2]],'holes':[]}
        with patch.object(module,'_shape_from_mask',side_effect=lossy):
            output,report=module.merge_added_overlaps(baseline,additions,100,80,.7)
        self.assertEqual(output,baseline+additions)
        self.assertFalse(report['audit'][0]['merged'])

    def test_duplicate_ids_and_wrong_class_are_rejected(self):
        with self.assertRaises(ValueError):module.merge_added_overlaps([square('a')],[square('a')],100,80)
        with self.assertRaises(ValueError):module.merge_added_overlaps([], [square('person',label='person')],100,80)


if __name__=='__main__':
    unittest.main()
