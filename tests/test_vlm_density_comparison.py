"""A requested ceiling without actual encoding evidence is not a density test."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('density_evaluation',Path(__file__).resolve().parents[1]/'tools/evaluate_vlm_density.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class VLMDensityComparisonTest(unittest.TestCase):
    def test_wrong_actual_tokens_or_unfinished_response_is_rejected(self):
        fixture={'model':'test','model_config_sha256':'expected'}
        raw={'model':'test','local_vlm':{'backend':'transformers_local_cuda','quantization':'nf4',
            'model_config_sha256':'expected','image_count':1,'requested_image_token_budget':1024,
            'effective_image_token_budget':1024,'actual_image_token_counts':[986],'image_grid_thw':[[1,58,68]]},
            'choices':[{'finish_reason':'stop','message':{'content':'{"category":"uncertain","reason":"No clear object"}'}}]}
        self.assertEqual(m.validate_response(raw,fixture,1024,986)['category'],'uncertain')
        wrong=deepcopy(raw);wrong['local_vlm']['actual_image_token_counts']=[238]
        with self.assertRaises(ValueError):m.validate_response(wrong,fixture,1024,986)
        wrong=deepcopy(raw);wrong['local_vlm']['effective_image_token_budget']=256
        with self.assertRaises(ValueError):m.validate_response(wrong,fixture,1024,986)
        wrong=deepcopy(raw);wrong['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):m.validate_response(wrong,fixture,1024,986)

    def test_only_budget_changes_in_paired_payload_and_role_metadata_is_not_sent(self):
        fixture={'model':'test','temperature':0,'max_tokens':192,'prompts':{'single':'Classify the image'},
            'schema':{'type':'object'},'cases':[{'id':'secret_case_id','role':'person_fixture'}]}
        first,a=m.build_payload(fixture,{},b'private image','single',256)
        second,b=m.build_payload(fixture,{},b'private image','single',1024)
        self.assertEqual(a,b)
        first.pop('image_token_budget');second.pop('image_token_budget')
        self.assertEqual(first,second)
        self.assertNotIn('secret_case_id',json.dumps(first))
        self.assertNotIn('person_fixture',json.dumps(first))

    def test_blank_failure_cannot_be_hidden_by_pallet_consensus_count(self):
        cases=[{'id':'p'+str(i),'role':'pallet_fixture'} for i in range(5)]+[
            {'id':'person','role':'person_fixture'},{'id':'blank','role':'blank_control'},
            {'id':'suspect','role':'suspected_non_pallet'}]
        rows=[{'case_id':c['id'],'budget':1024,'view':v,'parsed':{'category':
            'pallet' if c['role']=='pallet_fixture' else 'person' if c['role']=='person_fixture' else 'uncertain'}}
            for c in cases for v in ('single','outline_context')]
        self.assertTrue(all(m.quality_checks({'cases':cases},rows,1024).values()))
        next(r for r in rows if r['case_id']=='blank')['parsed']['category']='pallet'
        checks=m.quality_checks({'cases':cases},rows,1024)
        self.assertTrue(checks['minimum_three_pallet_consensus'])
        self.assertFalse(checks['blank_views_uncertain'])

if __name__=='__main__':unittest.main()
