import hashlib,json,tempfile,unittest
from pathlib import Path
from tools.prepare_cvat_human_signoff_template import prepare
class HumanSignoffTemplateTests(unittest.TestCase):
    def fixture(self,root):
        items=[{"candidate_id":f"{i}:{i}","frame_id":i,"source_image_sha256":hashlib.sha256(f"i{i}".encode()).hexdigest(),"canonical_geometry_sha256":hashlib.sha256(f"g{i}".encode()).hexdigest(),"focused_image_sha256":hashlib.sha256(f"f{i}".encode()).hexdigest()} for i in range(81)]; p=root/"review.json"; p.write_text(json.dumps({"schema_version":"labelagent.cvat-manual-instance-review-pack.v1","status":"pending_independent_review","test24_intersection_count":0,"training_authorized":False,"review_items":items})); return p
    def test_template_is_fully_pending_and_not_authorized(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); value,receipt=prepare(self.fixture(root),root/"template.json"); self.assertEqual(len(value["rows"]),81); self.assertTrue(all(r["decision"] is None for r in value["rows"])); self.assertFalse(receipt["training_authorized"])
    def test_test24_overlap_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=self.fixture(root); v=json.loads(p.read_text()); v["test24_intersection_count"]=1; p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,"isolation boundary"): prepare(p,root/"template.json")
if __name__=="__main__": unittest.main()
