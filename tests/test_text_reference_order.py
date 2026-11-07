import json
import unittest
from tools.evaluate_text_reference import validated_reason_first

class EvidenceOrderChecks(unittest.TestCase):
    def response(self,content,override=256):
        return {'local_vlm':{'backend':'transformers_local_cuda','image_count':1,'requested_image_token_budget':override},
                'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}]}
    def test_actual_category_first_quality_reply_is_rejected(self):
        with self.assertRaises(ValueError):validated_reason_first(self.response({'category':'pallet','reason':'platform'}),['pallet','person','uncertain'],1)
        self.assertEqual(validated_reason_first(self.response({'reason':'platform','category':'pallet'}),['pallet','person','uncertain'],1)['category'],'pallet')
    def test_original_default_control_order_is_accepted(self):
        raw=self.response({'category':'person','reason':'visible body'},None)
        self.assertEqual(validated_reason_first(raw,['pallet','person','uncertain'],1)['category'],'person')

if __name__=='__main__':unittest.main()
