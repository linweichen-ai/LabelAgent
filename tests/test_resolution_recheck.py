import unittest
from tools.ascii_target_recheck import ascii_card_payload
from tools.resolution_recheck import resolution_needed,resolution_payload
from tests.test_ascii_target_recheck import ASCIIChecks

class ResolutionChecks(unittest.TestCase):
    def test_class_symmetric_and_not_a_pallet_answer_rule(self):
        self.assertTrue(resolution_needed(['person','pallet']))
        self.assertTrue(resolution_needed(['pallet','person']))
        self.assertTrue(resolution_needed(['uncertain','person']))
        self.assertFalse(resolution_needed(['person','person']))
        self.assertFalse(resolution_needed(['pallet','pallet']))

    def test_only_visual_encoding_budget_changes(self):
        f,a,c=ASCIIChecks().inputs();old,_=ascii_card_payload(f,a,c,'overview');new,_=resolution_payload(f,a,c,'overview')
        self.assertEqual(new['image_token_budget'],1024);new['image_token_budget']=old['image_token_budget'];self.assertEqual(new,old)

    def test_default_probe_has_no_budget_override(self):
        f,a,c=ASCIIChecks().inputs();body,_=resolution_payload(f,a,c,'guard',True)
        self.assertNotIn('image_token_budget',body)
        self.assertEqual(resolution_payload(f,a,c,'guard',True),ascii_card_payload(f,a,c,'guard',True))

    def test_invalid_category_cannot_trigger_a_retry(self):
        with self.assertRaises(ValueError):resolution_needed(['pallet','bad'])

if __name__=='__main__':unittest.main()
