import unittest
import tests.test_ascii_target_recheck as ascii_tests
from tools.reference_semantics import descriptor_payload,reason_first_payload
from tools.ascii_target_recheck import ascii_card_payload

class ReferenceSemanticsChecks(unittest.TestCase):
    def test_query_builder_keeps_only_original_query_image(self):
        f,a,c=ascii_tests.ASCIIChecks().inputs();before,_=ascii_card_payload(f,a,c,'overview');after,_=reason_first_payload(f,a,c,'overview')
        self.assertEqual(before['messages'],after['messages'])
        self.assertEqual(list(after['response_format']['json_schema']['schema']['properties']),['reason','category'])
        self.assertEqual(after['response_format']['json_schema']['schema']['properties']['category']['enum'],['pallet','person','uncertain'])
    def test_reference_descriptor_is_not_a_query_classifier(self):
        p=descriptor_payload(b'approved-reference-only','local');s=p['response_format']['json_schema']['schema']
        self.assertEqual(set(s['properties']),{'pallet_cues','person_cues','uncertain_rule'});self.assertNotIn('category',s['properties'])
    def test_default_guard_request_is_preserved(self):
        f,a,c=ascii_tests.ASCIIChecks().inputs();self.assertEqual(ascii_card_payload(f,a,c,'guard',True),reason_first_payload(f,a,c,'guard',True))
    def test_query_expected_label_metadata_never_reaches_payload(self):
        f,a,c=ascii_tests.ASCIIChecks().inputs();before=reason_first_payload(f,a,c,'overview');after=reason_first_payload(f,a,{**c,'expected_category':'person','role':'pallet_fixture','id':'private-case-id'},'overview');self.assertEqual(before,after)

if __name__=='__main__':unittest.main()
