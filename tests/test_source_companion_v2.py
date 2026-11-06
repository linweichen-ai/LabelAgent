from types import SimpleNamespace
import unittest
from tests.test_source_companion import CompanionTests
from tools.source_companion_v2 import source_companion_payload
from tools.local_vlm_server import checked_single_image_budget

class MultiImageBudgetTests(unittest.TestCase):
    def test_server_rejects_single_image_override_for_two_images(self):
        with self.assertRaises(ValueError):checked_single_image_budget(256,SimpleNamespace(),2)

    def test_two_image_request_uses_global_default(self):
        f,a,c=CompanionTests().fixture();body,_=source_companion_payload(f,a,c,'overview')
        self.assertNotIn('image_token_budget',body)
        self.assertEqual(len(body['messages'][0]['content']),3)

    def test_guard_remains_single_image_and_no_override(self):
        f,a,c=CompanionTests().fixture();body,_=source_companion_payload(f,a,c,'guard',True)
        self.assertNotIn('image_token_budget',body)
        self.assertEqual(len(body['messages'][0]['content']),2)

if __name__=='__main__':unittest.main()
