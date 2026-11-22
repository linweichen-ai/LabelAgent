import hashlib,json,tempfile,unittest
from pathlib import Path
from PIL import Image
from tools.build_cvat_uncertain_temporal_context import build
class TemporalContextTests(unittest.TestCase):
    def fixture(self,root):
        images=root/"images"; images.mkdir(); frames=[]
        for i in range(9):
            p=images/f"{i}.png"; Image.new("RGB",(64,48),(i,100,100)).save(p); frames.append({"frame":i,"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
        candidates=[]; decisions=[]
        for i in (4,):
            cid=f"{i}:{i}"; candidates.append({"candidate_id":cid,"frame_id":i,"shape":{"points":[[5,5],[40,5],[40,30],[5,30]]}}); decisions.append({"candidate_id":cid,"zoom_triage":"uncertain_needs_human"})
        while len(decisions)<3: decisions.append(dict(decisions[0],candidate_id=f"x{len(decisions)}")); candidates.append({"candidate_id":decisions[-1]["candidate_id"],"frame_id":4,"shape":{"points":[[5,5],[40,5],[40,30],[5,30]]}})
        cp=root/"c.json"; cp.write_text(json.dumps({"candidates":candidates})); tp=root/"t.json"; tp.write_text(json.dumps({"decisions":decisions})); rp=root/"r.json"; rp.write_text(json.dumps({"private_triage_sha256":hashlib.sha256(tp.read_bytes()).hexdigest()})); ip=root/"i.json"; ip.write_text(json.dumps({"frames":frames})); return cp,tp,rp,ip,images
    def test_builds_same_parity_context_without_authorization(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest,receipt=build(*self.fixture(root),root/"out"); self.assertEqual(manifest["offsets"],[-4,-2,0,2,4]); self.assertFalse(receipt["training_authorized"])
    def test_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); args=self.fixture(root); args[1].write_text(args[1].read_text()+" ")
            with self.assertRaisesRegex(ValueError,"hash differs"): build(*args,root/"out")
if __name__=="__main__": unittest.main()
