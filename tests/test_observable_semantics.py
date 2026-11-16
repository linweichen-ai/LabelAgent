import json
import unittest
from lmformatenforcer import JsonSchemaParser
import tests.test_ascii_target_recheck as fixtures
from tools.observable_semantics import FIELDS,observable_schema,observable_payload,validated_observable
from tools.ascii_target_recheck import ascii_card_payload

class ObservableChecks(unittest.TestCase):
    def parser_at(self,text):
        p=JsonSchemaParser(observable_schema())
        for c in text:
            self.assertIn(c,p.get_allowed_characters());p=p.add_character(c)
        return p
    def test_marker_only_branch_cannot_emit_pallet(self):
        p=self.parser_at('{"content":"empty_or_markers","human":"not_supported","pallet":"not_supported","reason":"only marks","category":"')
        self.assertNotIn('p',p.get_allowed_characters());self.assertIn('u',p.get_allowed_characters())
    def test_real_parser_allows_both_positive_classes(self):
        for h,p,c in [('supported','not_supported','person'),('not_supported','supported','pallet')]:
            s=json.dumps(dict(content='photo',human=h,pallet=p,reason='visible structure',category=c),separators=(',',':'))
            self.assertTrue(self.parser_at(s).can_end())
    def test_ambiguous_class_evidence_abstains(self):
        p=self.parser_at('{"content":"photo","human":"supported","pallet":"supported","reason":"overlap","category":"')
        self.assertIn('u',p.get_allowed_characters());self.assertNotIn('p',p.get_allowed_characters())
    def test_no_images_or_label_metadata_are_added(self):
        f,a,c=fixtures.ASCIIChecks().inputs();before,_=ascii_card_payload(f,a,c,'overview');after,_=observable_payload(f,a,{**c,'expected_category':'pallet','role':'test','id':'hidden'},'overview')
        self.assertEqual(before['messages'],after['messages']);self.assertEqual(before['image_token_budget'],after['image_token_budget'])
    def test_default_probe_is_identical(self):
        f,a,c=fixtures.ASCIIChecks().inputs();self.assertEqual(ascii_card_payload(f,a,c,'guard',True),observable_payload(f,a,c,'guard',True))
    def test_validator_never_repairs_an_inconsistent_model_class(self):
        parsed=dict(content='empty_or_markers',human='not_supported',pallet='not_supported',reason='no object',category='pallet')
        raw={'local_vlm':{'backend':'transformers_local_cuda','image_count':1,'requested_image_token_budget':256},'choices':[{'finish_reason':'stop','message':{'content':json.dumps(parsed)}}]}
        with self.assertRaises(ValueError):validated_observable(raw,['pallet','person','uncertain'],1)

if __name__=='__main__':unittest.main()
