import hashlib,json,tempfile,unittest
from pathlib import Path
from tools.validate_cvat_zoom_triage import validate
class ZoomTriageTests(unittest.TestCase):
    def fixture(self,root):
        pack=root/"p.json"; pack.write_text(json.dumps({"schema_version":"labelagent.cvat-triage-zoom-pack.v1","status":"human_review_pending","items":[{"candidate_id":"1:2"}]})); inp=root/"i.json"; inp.write_text(json.dumps({"zoom_pack_manifest_sha256":hashlib.sha256(pack.read_bytes()).hexdigest(),"decisions":[{"candidate_id":"1:2","zoom_triage":"likely_ok_after_zoom","reason":"No obvious issue."}]})); return pack,inp
    def test_zoom_likely_ok_never_authorizes_training(self):
        with tempfile.TemporaryDirectory() as d: private,public=validate(*self.fixture(Path(d)))
        self.assertEqual(private["human_signoff_count"],0); self.assertFalse(public["training_authorized"])
    def test_wrong_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); pack,inp=self.fixture(root); v=json.loads(inp.read_text()); v["decisions"][0]["candidate_id"]="wrong"; inp.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,"exactly cover"): validate(pack,inp)
if __name__=="__main__": unittest.main()
