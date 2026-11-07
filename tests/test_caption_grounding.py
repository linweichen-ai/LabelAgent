"""Image observations must not cross-bind assets or bypass original protocol."""
from copy import deepcopy
import hashlib
import unittest
from tools.evaluate_caption_grounding import classification_payload, bound_observation, OBSERVATION_PREFIX


class CaptionGroundingTests(unittest.TestCase):
    def facts(self):
        data=b'private image bytes'
        fixture={'model':'m','model_config_sha256':'weights','temperature':0,'max_tokens':192,
            'schema':{'type':'object'},'prompts':{'single':'Original target rules.'}}
        view={'sha256':hashlib.sha256(data).hexdigest(),'expected_actual_image_tokens':{'256':238}}
        observation={'image_sha256':view['sha256'],'model_config_sha256':'weights',
            'actual_image_tokens':238,'description':'A visible object.'}
        return data,fixture,view,observation

    def test_only_observation_text_is_added_and_private_role_is_never_sent(self):
        data,fixture,view,obs=self.facts()
        baseline=classification_payload(fixture,view,data,'single')
        intervention=classification_payload(fixture,view,data,'single',obs)
        text=intervention['messages'][0]['content'][0]['text']
        self.assertTrue(text.startswith(baseline['messages'][0]['content'][0]['text']+OBSERVATION_PREFIX))
        intervention['messages'][0]['content'][0]['text']=baseline['messages'][0]['content'][0]['text']
        self.assertEqual(intervention,baseline)

    def test_cross_image_model_encoding_and_changed_source_are_rejected(self):
        data,fixture,view,obs=self.facts()
        for key in ('image_sha256','model_config_sha256','actual_image_tokens'):
            wrong=deepcopy(obs);wrong[key]='incorrect'
            with self.assertRaises(ValueError):classification_payload(fixture,view,data,'single',wrong)
        with self.assertRaises(ValueError):classification_payload(fixture,view,b'changed','single',obs)

    def test_unfinished_or_repaired_caption_cannot_enter_next_model_request(self):
        data,fixture,view,obs=self.facts()
        raw={'model':'m','local_vlm':{'backend':'transformers_local_cuda','quantization':'nf4',
            'model_config_sha256':'weights','image_count':1,'requested_image_token_budget':256,
            'effective_image_token_budget':256,'actual_image_token_counts':[238],
            'image_grid_thw':[[1,28,34]],'inference_stats':{'structured_output':True}},
            'choices':[{'finish_reason':'stop','message':{'content':'{"description":"A visible object."}'}}]}
        self.assertEqual(bound_observation(raw,fixture,view['sha256'],238),obs)
        for value in ('```json\n{"description":"object"}\n```','{"caption":"object"}'):
            wrong=deepcopy(raw);wrong['choices'][0]['message']['content']=value
            with self.assertRaises(ValueError):bound_observation(wrong,fixture,view['sha256'],238)
        wrong=deepcopy(raw);wrong['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):bound_observation(wrong,fixture,view['sha256'],238)


if __name__=='__main__':unittest.main()
