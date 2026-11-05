import json
from pathlib import Path
import tempfile
import unittest
from tools.evaluate_vlm_review_v2 import save_review_response

class ReviewArchiveChecks(unittest.TestCase):
    def test_actual_response_roundtrip_uses_path_for_writer_and_string_for_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            raw={'number':1,'raw_response':{'id':'test-only','choices':[{'message':{'content':'evidence'}}]}}
            result=save_review_response(Path(d),1,raw)
            self.assertIsInstance(result,str)
            self.assertEqual(json.loads(Path(result).read_bytes()),raw)
            self.assertEqual(Path(result).name,'response_001.json')

if __name__=='__main__':unittest.main()
