import unittest
import tests.test_ascii_target_recheck as fixtures
from tools.vlm_proposal_review import review_payload,reviewed_category

class ProposalReviewChecks(unittest.TestCase):
    def test_reviewer_cannot_invent_an_alternative_class(self):
        for draft in ('person','pallet'):
            self.assertEqual(reviewed_category(draft,'supported'),draft)
            self.assertEqual(reviewed_category(draft,'unsupported'),'uncertain')
            self.assertEqual(reviewed_category(draft,'uncertain'),'uncertain')
    def test_expected_truth_and_case_ids_are_not_sent(self):
        f,a,c=fixtures.ASCIIChecks().inputs();before=review_payload(f,a,c,'overview','person');after=review_payload(f,a,{**c,'id':'private-id','expected_category':'pallet','role':'pallet_fixture'},'overview','person')
        self.assertEqual(before,after);self.assertIn('UNTRUSTED',before[0]['messages'][0]['content'][0]['text'])
    def test_review_receives_exact_same_target_image(self):
        from tools.ascii_target_recheck import ascii_card_payload
        f,a,c=fixtures.ASCIIChecks().inputs();old,_=ascii_card_payload(f,a,c,'overview');new,_=review_payload(f,a,c,'overview','pallet')
        self.assertEqual(old['messages'][0]['content'][1:],new['messages'][0]['content'][1:])
    def test_invalid_draft_is_rejected(self):
        f,a,c=fixtures.ASCIIChecks().inputs()
        with self.assertRaises(ValueError):review_payload(f,a,c,'overview','ground_truth_answer')

if __name__=='__main__':unittest.main()
