"""Protect exact image bytes and prevent partial or uncertain probes passing."""
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_vlm_grounding import build_payload,checked_fixture,fixture_checks,sha,validated_response


def fixture(root):
    stream=io.BytesIO()
    Image.new('RGB',(20,16),'blue').save(stream,'JPEG')
    raw=stream.getvalue()
    (root/'image.jpg').write_bytes(raw)
    ref={'path':'image.jpg','sha256':sha(raw)}
    cases=[{'id':role,'role':role,'views':{'single':[ref],'context':[ref]}}
        for role in ('pallet_fixture','blank_control','suspected_non_pallet')]
    return {'asset_root':'.','cases':cases,'minimum_positive_accepts':1,'categories':['pallet','cargo','uncertain'],
        'temperature':0,'max_tokens':192,'prompts':{'single':'Target only','context':'Target and context'}},raw


class VLMGroundingTests(unittest.TestCase):
    def test_payload_preserves_JPEG_bytes_and_omits_fixture_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            spec,raw=fixture(root)
            (root/'fixture.json').write_text(json.dumps(spec))
            spec,assets=checked_fixture(root/'fixture.json')
            payload,binding=build_payload(spec,assets,spec['cases'][0],'single','real-model')
            data=payload['messages'][0]['content'][1]['image_url']['url'].split(',',1)[1]
            self.assertEqual(base64.b64decode(data),raw)
            self.assertNotIn('pallet_fixture',json.dumps(payload))
            self.assertEqual(binding,build_payload(spec,assets,spec['cases'][0],'single','other-model')[1])

    def test_changed_image_rejected_before_any_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            spec,_=fixture(root)
            (root/'fixture.json').write_text(json.dumps(spec))
            (root/'image.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes changed'):checked_fixture(root/'fixture.json')

    def test_asset_cannot_escape_declared_image_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            spec,_=fixture(root)
            spec['cases'][0]['views']['single']=[{'path':'../outside.jpg','sha256':'unused'}]
            (root/'fixture.json').write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'escapes'):checked_fixture(root/'fixture.json')

    def test_all_uncertain_is_not_useful_success(self):
        with tempfile.TemporaryDirectory() as directory:
            spec,_=fixture(Path(directory))
            rows=[{'id':row['id'],'decision':'uncertain'} for row in spec['cases']]
            checks=fixture_checks(spec,rows,6)
            self.assertFalse(checks['minimum_qualitative_pallet_accepts'])
            self.assertFalse(all(checks.values()))

    def test_missing_result_cannot_pass_on_vacuous_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            spec,_=fixture(Path(directory))
            self.assertFalse(any(fixture_checks(spec,[],0).values()))

    def test_test_backend_or_wrong_image_count_cannot_be_real_response(self):
        raw={'local_vlm':{'backend':'contract_test_double','image_count':1},'choices':[
            {'finish_reason':'stop','message':{'content':'{"category":"pallet","reason":"test"}'}}]}
        with self.assertRaises(ValueError):validated_response(raw,['pallet'],1)
        raw['local_vlm']['backend']='transformers_local_cuda'
        with self.assertRaises(ValueError):validated_response(raw,['pallet'],2)

    def test_complete_JSON_with_length_finish_is_still_unfinished(self):
        raw={'local_vlm':{'backend':'transformers_local_cuda','image_count':1},'choices':[
            {'finish_reason':'length','message':{'content':'{"category":"pallet","reason":"test"}'}}]}
        with self.assertRaisesRegex(ValueError,'unfinished'):validated_response(raw,['pallet'],1)


if __name__=='__main__':unittest.main()
