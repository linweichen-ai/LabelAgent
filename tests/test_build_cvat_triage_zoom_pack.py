import hashlib,json,tempfile,unittest
from pathlib import Path
from PIL import Image
from tools.build_cvat_triage_zoom_pack import build
class ZoomPackTests(unittest.TestCase):
    def fixture(self,root):
        images=root/"images"; images.mkdir(); candidates=[]; decisions=[]
        for i in range(8):
            path=images/f"{i}.png"; Image.new("RGB",(64,48),(i,100,100)).save(path); digest=hashlib.sha256(path.read_bytes()).hexdigest(); cid=f"{i}:{i}"; candidates.append({"candidate_id":cid,"image":{"file":path.name,"sha256":digest},"shape":{"id":i,"label":"pallet","cvat_metadata":{"source":"manual"},"points":[[5,5],[40,5],[40,30],[5,30]],"parts":[]}}); decisions.append({"candidate_id":cid,"assistant_triage":"needs_human_review","reason":"Needs zoom."})
        cp=root/"c.json"; cp.write_text(json.dumps({"candidates":candidates})); rp=root/"r.json"; rp.write_text("{}"); tp=root/"t.json"; tp.write_text(json.dumps({"review_pack_manifest_sha256":hashlib.sha256(rp.read_bytes()).hexdigest(),"human_truth":False,"training_authorized":False,"decisions":decisions})); tr=root/"tr.json"; tr.write_text(json.dumps({"private_triage_sha256":hashlib.sha256(tp.read_bytes()).hexdigest()})); return cp,rp,tp,tr,images
    def test_builds_eight_private_zooms_without_authorization(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest,receipt=build(*self.fixture(root),root/"out"); self.assertEqual(manifest["item_count"],8); self.assertFalse(receipt["training_authorized"]); self.assertTrue((root/"out"/"zoom_01.jpg").is_file())
    def test_triage_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); args=self.fixture(root); args[2].write_text(args[2].read_text()+" ")
            with self.assertRaisesRegex(ValueError,"triage hash differs"): build(*args,root/"out")
if __name__=="__main__": unittest.main()
