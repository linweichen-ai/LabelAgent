import hashlib,json,tempfile,unittest
from pathlib import Path
from neutral_caption_cache import lookup,payload


class ExactCaptionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.raw=b'fixed-image-bytes';self.image=self.root/'input.png';self.image.write_bytes(self.raw);self.response=self.root/'reply.json';self.response.write_text(json.dumps({'local_vlm':{'model_config_sha256':'fixed-model','actual_image_token_counts':[256]},'choices':[{'message':{'content':'{"object":"a surface"}'}}]}));self.row={'request_sha256':hashlib.sha256(json.dumps(payload(self.raw,256),sort_keys=True).encode()).hexdigest(),'image':str(self.image),'response':str(self.response)}
    def tearDown(self):self.temp.cleanup()
    def test_exact_payload_and_model_reuse_without_request(self):self.assertFalse(lookup(self.raw,'fixed-model',[self.row])['new_model_call'])
    def test_model_or_image_change_cannot_reuse(self):
        self.assertIsNone(lookup(self.raw,'other-model',[self.row]));self.assertIsNone(lookup(b'changed','fixed-model',[self.row]));self.image.write_bytes(b'changed');self.assertIsNone(lookup(self.raw,'fixed-model',[self.row]))
    def test_actual_encoding_or_response_digest_change_rejected(self):
        r=json.loads(self.response.read_text());r['local_vlm']['actual_image_token_counts']=[1024];self.response.write_text(json.dumps(r));self.assertIsNone(lookup(self.raw,'fixed-model',[self.row]));r['local_vlm']['actual_image_token_counts']=[256];self.response.write_text(json.dumps(r));row={**self.row,'response_sha256':'wrong'};self.assertIsNone(lookup(self.raw,'fixed-model',[row]))
    def test_operational_hit_precedes_any_network(self):
        from unittest.mock import patch
        from review_material_cached import review_one
        proposal={'semantic_image':str(self.image),'request_sha256':self.row['request_sha256']}
        with patch('urllib.request.urlopen',side_effect=AssertionError('network must not be called')):
            r=review_one(proposal,'fixed-model',[self.row],self.root,allow_network=True)
        self.assertFalse(r['new_model_call']);self.assertTrue(r['cache_verified'])
    def test_operational_cache_only_miss_never_calls_network(self):
        from unittest.mock import patch
        from review_material_cached import review_one
        proposal={'semantic_image':str(self.image),'request_sha256':self.row['request_sha256']}
        with patch('urllib.request.urlopen',side_effect=AssertionError('network must not be called')):
            with self.assertRaises(RuntimeError):review_one(proposal,'fixed-model',[],self.root,allow_network=False)


if __name__=='__main__':unittest.main()
