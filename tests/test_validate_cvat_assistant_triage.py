import hashlib, json, tempfile, unittest
from pathlib import Path
from tools.validate_cvat_assistant_triage import validate
class AssistantTriageValidationTests(unittest.TestCase):
    def fixture(self, root: Path):
        pack=root/"pack.json"; pack.write_text(json.dumps({"schema_version":"labelagent.cvat-geometry-priority-review-pack.v1","status":"assistant_triage_pending","pages":[{"priority":"high","candidate_ids":["0:1"]}]})); triage=root/"triage.json"; triage.write_text(json.dumps({"review_pack_manifest_sha256":hashlib.sha256(pack.read_bytes()).hexdigest(),"decisions":[{"candidate_id":"0:1","assistant_triage":"likely_ok","reason":"Boundary appears complete."}]})); return pack,triage
    def test_likely_ok_never_authorizes_training(self):
        with tempfile.TemporaryDirectory() as directory: private,public=validate(*self.fixture(Path(directory)))
        self.assertEqual(private["decision_counts"]["likely_ok"],1); self.assertEqual(private["human_signoff_count"],0); self.assertFalse(public["training_authorized"])
    def test_missing_candidate_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); pack,triage=self.fixture(root); value=json.loads(triage.read_text()); value["decisions"]=[]; triage.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"exactly cover"): validate(pack,triage)
if __name__ == "__main__": unittest.main()
