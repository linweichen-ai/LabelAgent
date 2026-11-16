"""Ensure the challenge metric cannot pass by losing unrelated masks."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('evaluate_duplicate_recovery',Path(__file__).resolve().parents[1]/'tools/evaluate_duplicate_recovery.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def shape(identifier,left,label='person'):
    return {'id':identifier,'label':label,'points':[[left,2],[left+8,2],[left+8,15],[left,15]]}


def result(shapes):
    return {'shapes':shapes,'agent':{'rounds':2,'converged':True,
        'audit':[{'tool_results':[{'status':'applied'}]}]}}


class DuplicateMetricTests(unittest.TestCase):
    def test_id_changes_and_reordering_can_recover_the_same_labelled_masks(self):
        seed=[shape(1,2),shape(2,20)]
        changed=[{**seed[1],'id':91},{**seed[0],'id':92}]
        value=module.recovery_metrics(seed,result(changed),40,30,seed[0])
        self.assertTrue(value['known_injection_recovered'])
        self.assertTrue(value['unrelated_candidate_geometry_preserved'])
        self.assertFalse(value['pixel_accuracy_evaluated'])

    def test_equal_candidate_count_with_a_missing_other_instance_fails(self):
        seed=[shape(1,2),shape(2,20)]
        changed=[seed[0],{**seed[0],'id':3}]
        value=module.recovery_metrics(seed,result(changed),40,30,seed[0])
        self.assertFalse(value['known_injection_recovered'])
        self.assertFalse(value['unrelated_candidate_geometry_preserved'])

    def test_changed_class_or_boundary_does_not_count_as_recovery(self):
        seed=[shape(1,2),shape(2,20)]
        for change in ('class','geometry'):
            changed=deepcopy(seed)
            if change=='class':changed[0]['label']='pallet'
            else:changed[0]['points'][0][0]+=1
            self.assertFalse(module.recovery_metrics(seed,result(changed),40,30,seed[0])['known_injection_recovered'])


if __name__=='__main__':
    unittest.main()
