import copy
import hashlib
import json
import unittest
from tools.contour_visibility import classify_payload
from tools.source_last import source_last_payload,swap_image_roles,digest

class SourceLastTests(unittest.TestCase):
    def setup_data(self):
        data={'clean.jpg':b'clean-original','outlined.jpg':b'outlined-original'}
        assets={k:(v,'image/jpeg') for k,v in data.items()}
        refs=[{'path':k,'sha256':hashlib.sha256(v).hexdigest()} for k,v in data.items()]
        f={'model':'local','categories':['pallet','person','uncertain']}
        c={'id':'PRIVATE_ID','views':{v:refs for v in ('overview','linked')}}
        return f,assets,c

    def test_swaps_image_bytes_and_corresponding_roles_only(self):
        f,a,c=self.setup_data()
        for view in ('overview','linked'):
            old,_=classify_payload(f,a,c,view,None)
            new,h=source_last_payload(f,a,c,view)
            self.assertEqual(new['messages'][0]['content'][1:],old['messages'][0]['content'][1:][::-1])
            self.assertIn('Image 2 is clean',new['messages'][0]['content'][0]['text'])
            self.assertIn('Image 1 repeats',new['messages'][0]['content'][0]['text'])
            self.assertIn('If image 2 is only a uniform empty',new['messages'][0]['content'][0]['text'])
            self.assertEqual({k:v for k,v in old.items() if k!='messages'},{k:v for k,v in new.items() if k!='messages'})
            self.assertEqual(digest(new),h)

    def test_inverse_recovers_baseline_exactly(self):
        f,a,c=self.setup_data()
        old,_=classify_payload(f,a,c,'overview',None)
        self.assertEqual(swap_image_roles(swap_image_roles(old)),old)

    def test_baseline_and_fixture_not_mutated(self):
        f,a,c=self.setup_data();before=copy.deepcopy((f,a,c))
        old,_=classify_payload(f,a,c,'linked',None);saved=copy.deepcopy(old)
        swap_image_roles(old);source_last_payload(f,a,c,'linked')
        self.assertEqual(old,saved);self.assertEqual((f,a,c),before)

    def test_no_ground_truth_case_id_or_inspection_injection(self):
        f,a,c=self.setup_data();expected=source_last_payload(f,a,c,'overview')
        c.update(id='OTHER_PRIVATE',human_category='DIFFERENT_TRUTH',visibility='empty',features='UNRELIABLE')
        f['expected_category']='ANOTHER_TRUTH'
        self.assertEqual(source_last_payload(f,a,c,'overview'),expected)
        text=json.dumps(expected[0])
        for value in ('PRIVATE_ID','OTHER_PRIVATE','DIFFERENT_TRUTH','UNRELIABLE','ANOTHER_TRUTH'):
            self.assertNotIn(value,text)

    def test_multi_image_budget_override_stays_absent(self):
        f,a,c=self.setup_data();body,_=source_last_payload(f,a,c,'overview')
        self.assertNotIn('image_token_budget',body)
        self.assertEqual(body['temperature'],0)
        self.assertEqual(body['max_tokens'],192)

    def test_missing_or_ambiguous_role_and_shape_fail(self):
        f,a,c=self.setup_data();body,_=classify_payload(f,a,c,'overview',None)
        bad=copy.deepcopy(body);bad['messages'][0]['content'][0]['text']='Only Image 1 exists'
        with self.assertRaises(ValueError):swap_image_roles(bad)
        bad=copy.deepcopy(body);bad['messages'][0]['content'].append({'type':'text','text':'hidden'})
        with self.assertRaises(ValueError):swap_image_roles(bad)

if __name__=='__main__':unittest.main()
