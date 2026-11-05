import hashlib
import unittest
from copy import deepcopy

from lmformatenforcer import JsonSchemaParser
from tools.ascii_target_recheck import ascii_card_payload
from tools.evaluate_target_cards import card_payload

class ASCIIChecks(unittest.TestCase):
    def inputs(self):
        raw=b'local-image-bytes';ref={'path':'a.jpg','sha256':hashlib.sha256(raw).hexdigest()}
        fixture={'model':'local','categories':['pallet','person','uncertain'],'temperature':0,'max_tokens':192,
                 'image_token_budgets':{'overview':256},'guard_prompt':'original guard','guard_image':ref,'prompts':{'overview':'same neutral prompt'}}
        return fixture,{'a.jpg':(raw,'image/jpeg')},{'views':{'overview':[ref]}}

    def test_only_reason_character_constraint_changes(self):
        f,a,c=self.inputs();before,_=card_payload(f,a,c,'overview');after,_=ascii_card_payload(f,a,c,'overview')
        modified=deepcopy(after);modified['response_format']['json_schema']['schema']['properties']['reason']=before['response_format']['json_schema']['schema']['properties']['reason']
        self.assertEqual(before,modified)
        self.assertEqual(after['response_format']['json_schema']['schema']['properties']['category']['enum'],['pallet','person','uncertain'])

    def test_default_control_request_is_byte_identical(self):
        f,a,c=self.inputs();self.assertEqual(card_payload(f,a,c,'guard',True),ascii_card_payload(f,a,c,'guard',True))

    def test_real_parser_rejects_problematic_unicode_inside_reason(self):
        f,a,c=self.inputs();body,_=ascii_card_payload(f,a,c,'overview')
        parser=JsonSchemaParser(body['response_format']['json_schema']['schema'])
        for ch in '{"category":"pallet","reason":"visible ':parser=parser.add_character(ch)
        self.assertNotIn('\u202f',parser.get_allowed_characters())
        self.assertIn('d',parser.get_allowed_characters())

    def test_real_parser_closes_at_original_100_character_limit(self):
        f,a,c=self.inputs();body,_=ascii_card_payload(f,a,c,'overview');parser=JsonSchemaParser(body['response_format']['json_schema']['schema'])
        for ch in '{"category":"person","reason":"'+'a'*100:parser=parser.add_character(ch)
        self.assertEqual(parser.get_allowed_characters(),'"')
        for ch in '"}':parser=parser.add_character(ch)
        self.assertTrue(parser.can_end())

if __name__=='__main__':unittest.main()
