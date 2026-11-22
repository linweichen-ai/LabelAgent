import hashlib,json,tempfile,unittest
from pathlib import Path
from tools.validate_cvat_temporal_triage import validate
class TemporalTriageTests(unittest.TestCase):
    def fixture(self,root):
        p=root/"p.json"; p.write_text(json.dumps({"schema_version":"labelagent.cvat-uncertain-temporal-context.v1","status":"assistant_review_pending","items":[{"candidate_id":"1:2"}]})); i=root/"i.json"; i.write_text(json.dumps({"temporal_pack_manifest_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"decisions":[{"candidate_id":"1:2","temporal_triage":"likely_ok_with_temporal","reason":"Neighbor frames support the occlusion."}]})); return p,i
    def test_likely_ok_remains_non_authoritative(self):
        with tempfile.TemporaryDirectory() as d: private,public=validate(*self.fixture(Path(d)))
        self.assertEqual(private["human_signoff_count"],0); self.assertFalse(public["training_authorized"])
    def test_wrong_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p,i=self.fixture(root); v=json.loads(i.read_text()); v["decisions"][0]["candidate_id"]="wrong"; i.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,"exactly cover"): validate(p,i)
if __name__=="__main__": unittest.main()
